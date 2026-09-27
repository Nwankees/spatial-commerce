from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.dimension_extraction import extract_from_features, extract_from_html
from app.dimension_parsing import parse_field, parse_labeled_dimensions, parse_length
from app.dimension_resolver import DimensionResolver, ResolutionCache
from app.main import app, get_dimension_resolver, get_product_cache, get_resolution_cache
from app.page_fetcher import FetchedPage, HttpPageFetcher, PageFetchError, is_public_http_url
from app.product_cache import ProductSearchCache
from app.product_details import ProductDetail, SerpApiImmersiveProductDetailSource, StoreLink
from app.product_models import ProductCandidate
from app.product_providers import ProductProviderTimeoutError

IN = 0.0254


def run(coro):
    return asyncio.run(coro)


def candidate(**overrides) -> ProductCandidate:
    fields = dict(
        id="serpapi:1", provider="serpapi", providerProductId="1",
        title="Accent Chair with Ottoman, 40 inch Wide Oversized Chair", price=199.0, priceText="$199.00",
        currency="USD", retailer="Wayfair", productUrl="https://www.google.com/search?ibp=oshop&prds=productid:1",
        detailPageToken="token-1",
    )
    fields.update(overrides)
    return ProductCandidate(**fields)


def page(body: str, head: str = "") -> str:
    return f"<html><head>{head}</head><body>{body}</body></html>"


def jsonld(obj) -> str:
    return f'<script type="application/ld+json">{json.dumps(obj)}</script>'


# ---- Unit conversion ----------------------------------------------------

@pytest.mark.parametrize(
    ("text", "meters"),
    [("762 mm", 0.762), ("76.2 cm", 0.762), ("0.762 m", 0.762), ("30 in", 30 * IN), ('30"', 30 * IN),
     ("30 inches", 30 * IN), ("2.5 ft", 0.762), ("2 ft 6 in", 0.762), ("30 1/2 in", 30.5 * IN), ("1,200 mm", 1.2)],
)
def test_units_normalize_to_meters(text: str, meters: float) -> None:
    assert parse_length(text) == pytest.approx(meters, abs=1e-4)


@pytest.mark.parametrize("text", ["30", "thirty inches", "30 px", "0.001 mm", "500 m", ""])
def test_lengths_without_explicit_unit_or_out_of_range_are_rejected(text: str) -> None:
    assert parse_length(text) is None


# ---- Text / label parsing -----------------------------------------------

def test_labeled_wxdxh_text() -> None:
    c = parse_labeled_dimensions('Product Dimensions: 30" W x 28" D x 35" H', require_keyword=True)
    assert (c.width, c.depth, c.height) == pytest.approx((30 * IN, 28 * IN, 35 * IN))


def test_labels_map_by_letter_not_position() -> None:
    c = parse_labeled_dimensions("Overall: 30.5'' H x 72'' W x 30'' D", require_keyword=True)
    assert (c.width, c.depth, c.height) == pytest.approx((72 * IN, 30 * IN, 30.5 * IN))


def test_declared_order_in_label_is_used() -> None:
    c = parse_field("Dimensions (W x D x H)", "76 x 71 x 89 cm")
    assert (c.width, c.depth, c.height) == pytest.approx((0.76, 0.71, 0.89))


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("Dimensions", "30 x 28 x 35 in"),                                   # unlabeled order
        ("Assembled Product Dimensions (L x W x H)", "28.70 x 25.20 x 38.20 in"),  # L is ambiguous
        ("Package Dimensions", '30" W x 28" D x 35" H'),                    # not the product
        ("Seat Width", "22 in"),                                             # a part, not overall
        ("Length", '38"L (Package Dimensions)'),
        ("Width", "30"),                                                     # no unit
        ("Width", "30 in D"),                                                # contradicting designator
        ("Adjustable Height", "Yes"),
    ],
)
def test_ambiguous_or_non_footprint_values_are_rejected(label: str, value: str) -> None:
    assert parse_field(label, value).known_axes == 0


def test_page_text_requires_dimension_keyword_and_rejects_fragments() -> None:
    assert parse_labeled_dimensions('Fits laptops up to 12" W x 9" D', require_keyword=True).known_axes == 0
    assert parse_labeled_dimensions("Dimensions: 2 ft 6 in W x 700 mm D", require_keyword=True).known_axes == 0
    assert parse_labeled_dimensions('Dimensions: 30" W x 28" D x 35" H x 40" L', require_keyword=True).known_axes == 0


def test_conflicting_values_for_an_axis_are_dropped() -> None:
    findings = extract_from_features([("Width", "30 in"), ("Width", "32 in"), ("Depth", "20 in")], None)
    c = findings[0].candidate
    assert c.width is None and c.depth == pytest.approx(20 * IN)


# ---- HTML extraction ----------------------------------------------------

def test_jsonld_quantitative_values() -> None:
    html = page("", jsonld({
        "@context": "https://schema.org", "@type": "Product", "name": "Chair",
        "width": {"@type": "QuantitativeValue", "value": 30, "unitCode": "INH"},
        "depth": {"@type": "QuantitativeValue", "value": 71.1, "unitCode": "CMT"},
        "height": "35 in",
    }))
    (finding,) = [f for f in extract_from_html(html, "https://shop.example.com/p") if f.source_type == "json_ld"]
    assert (finding.candidate.width, finding.candidate.depth, finding.candidate.height) == pytest.approx(
        (30 * IN, 0.711, 35 * IN), abs=1e-4
    )


def test_jsonld_additional_property_in_graph() -> None:
    html = page("", jsonld({"@graph": [{"@type": "Product", "additionalProperty": [
        {"@type": "PropertyValue", "name": "Overall Width", "value": 120, "unitCode": "CMT"},
        {"@type": "PropertyValue", "name": "Overall Depth", "value": "60 cm"},
        {"@type": "PropertyValue", "name": "Seat Height", "value": "45 cm"},
    ]}]}))
    c = next(f for f in extract_from_html(html, None) if f.source_type == "json_ld").candidate
    assert (c.width, c.depth, c.height) == (pytest.approx(1.2), pytest.approx(0.6), None)


def test_jsonld_bare_numbers_are_not_trusted() -> None:
    html = page("", jsonld({"@type": "Product", "width": 30, "depth": {"value": 28}}))
    assert extract_from_html(html, None) == []


def test_spec_table_and_definition_list() -> None:
    html = page("""
      <table><tr><th>Overall Width</th><td>1200 mm</td></tr>
             <tr><th>Overall Depth</th><td>600 mm</td></tr>
             <tr><th>Seat Height</th><td>450 mm</td></tr></table>
      <dl><dt>Height (in)</dt><dd>29.5</dd></dl>""")
    c = next(f for f in extract_from_html(html, None) if f.source_type == "spec_table").candidate
    assert (c.width, c.depth, c.height) == pytest.approx((1.2, 0.6, 29.5 * IN))


def test_page_text_dimensions() -> None:
    html = page("<p>Solid oak.</p><p>Overall dimensions: 48 in W x 24 in D x 30 in H</p>")
    (finding,) = extract_from_html(html, "https://shop.example.com/desk")
    assert finding.source_type == "page_text"
    assert finding.candidate.width == pytest.approx(48 * IN)


def test_malformed_and_empty_pages_yield_nothing() -> None:
    assert extract_from_html("<html><body><table><tr><td>Width<td>", None) == []
    assert extract_from_html("\x00\x01 not html <<<>>>", None) == []
    assert extract_from_html(page("", '<script type="application/ld+json">{broken json</script>'), None) == []


def test_title_and_image_are_never_used_as_dimensions() -> None:
    html = page('<h1>Legahome 40 inch Wide Oversized Chair</h1><img src="c.jpg" width="800" height="600">'
                '<span itemprop="width">800</span>')
    assert extract_from_html(html, None) == []


# ---- Resolver -----------------------------------------------------------

class FakeDetailSource:
    name = "fake"

    def __init__(self, detail: ProductDetail | None = None, error: Exception | None = None) -> None:
        self.detail, self.error, self.calls = detail, error, 0

    @property
    def configured(self) -> bool:
        return True

    async def fetch(self, candidate):
        self.calls += 1
        if self.error:
            raise self.error
        return self.detail


class FakeFetcher:
    def __init__(self, pages: dict[str, str | Exception]) -> None:
        self.pages, self.requested = pages, []

    async def fetch(self, url: str) -> FetchedPage:
        self.requested.append(url)
        outcome = self.pages[url]
        if isinstance(outcome, Exception):
            raise outcome
        return FetchedPage(url, outcome)


STORE = "https://www.wayfair.com/chair-123.html"


def test_provider_features_resolve_partial_dimensions() -> None:
    detail = ProductDetail(features=[("Width", "40 in wide"), ("Height", "23 in high"), ("Color", "Black")],
                           stores=[StoreLink("Wayfair", STORE)], source_name="provider")
    result = run(DimensionResolver(FakeDetailSource(detail), FakeFetcher({STORE: page("<p>No specs</p>")})).resolve(candidate()))
    assert result.status == "partial"
    assert result.sourceType == "structured_metadata"
    assert result.widthMeters == pytest.approx(40 * IN) and result.depthMeters is None
    assert result.heightMeters == pytest.approx(23 * IN)
    assert "Width: 40 in wide" in result.rawDimensions


def test_jsonld_on_retailer_page_outranks_provider_features() -> None:
    detail = ProductDetail(features=[("Width", "38 in"), ("Depth", "37 in"), ("Height", "33.5\"")],
                           stores=[StoreLink("Other", "https://other.example.com/p"), StoreLink("Wayfair", STORE)])
    html = page("", jsonld({"@type": "Product", "width": "39 in", "depth": "36 in", "height": "34 in"}))
    fetcher = FakeFetcher({STORE: html, "https://other.example.com/p": PageFetchError("blocked", retryable=False)})
    result = run(DimensionResolver(FakeDetailSource(detail), fetcher).resolve(candidate()))
    assert fetcher.requested[0] == STORE  # the result's own retailer is tried first
    assert result.status == "verified" and result.sourceType == "json_ld"
    assert result.sourceUrl == STORE and result.sourceName == "Wayfair"
    assert result.widthMeters == pytest.approx(39 * IN)


def test_verified_lower_priority_beats_partial_higher_priority() -> None:
    detail = ProductDetail(features=[("Width", "38 in"), ("Depth", "37 in"), ("Height", "33 in")], stores=[StoreLink("Wayfair", STORE)])
    html = page("", jsonld({"@type": "Product", "height": "34 in"}))
    result = run(DimensionResolver(FakeDetailSource(detail), FakeFetcher({STORE: html})).resolve(candidate()))
    assert result.status == "verified" and result.sourceType == "structured_metadata"


def test_direct_retailer_url_is_used_without_provider_details() -> None:
    url = "https://shop.example.com/desk"
    fetcher = FakeFetcher({url: page("<p>Dimensions: 120 cm W x 60 cm D x 75 cm H</p>")})
    result = run(DimensionResolver(None, fetcher).resolve(candidate(productUrl=url, detailPageToken=None)))
    assert result.status == "verified" and result.sourceType == "page_text" and result.sourceUrl == url


def test_missing_dimensions_are_unavailable_not_guessed() -> None:
    detail = ProductDetail(features=[("Color", "Black")], stores=[StoreLink("Wayfair", STORE)])
    result = run(DimensionResolver(FakeDetailSource(detail), FakeFetcher({STORE: page("<h1>40 inch Wide Chair</h1>")})).resolve(candidate()))
    assert result.status == "unavailable" and result.sourceType == "unavailable"
    assert result.widthMeters is None and result.depthMeters is None and result.heightMeters is None
    assert result.retryable is False and result.message


def test_timeouts_and_provider_failures_are_retryable_unavailable() -> None:
    fetcher = FakeFetcher({})
    result = run(DimensionResolver(FakeDetailSource(error=ProductProviderTimeoutError("timed out")), fetcher).resolve(candidate()))
    assert result.status == "unavailable" and result.retryable is True
    detail = ProductDetail(stores=[StoreLink("Wayfair", STORE)])
    result = run(DimensionResolver(FakeDetailSource(detail), FakeFetcher({STORE: PageFetchError("timed out", retryable=True)})).resolve(candidate()))
    assert result.retryable is True


def test_blocked_page_is_unavailable_not_retryable() -> None:
    detail = ProductDetail(stores=[StoreLink("west elm", STORE)])
    fetcher = FakeFetcher({STORE: PageFetchError("Retailer blocked automated access (HTTP 403).", retryable=False)})
    result = run(DimensionResolver(FakeDetailSource(detail), fetcher).resolve(candidate()))
    assert result.status == "unavailable" and result.retryable is False and "403" in result.message


def test_unexpected_errors_never_escape() -> None:
    class Exploding:
        async def fetch(self, url):
            raise RuntimeError("boom")

    result = run(DimensionResolver(None, Exploding()).resolve(candidate(productUrl="https://shop.example.com/p", detailPageToken=None)))
    assert result.status == "unavailable" and result.retryable is True


def test_non_public_or_google_urls_are_never_fetched() -> None:
    assert not is_public_http_url("http://127.0.0.1:8000/admin")
    assert not is_public_http_url("http://localhost/x")
    assert not is_public_http_url("file:///etc/passwd")
    detail = ProductDetail(stores=[StoreLink("x", "http://10.0.0.1/p"), StoreLink("g", "https://www.google.com/x")])
    fetcher = FakeFetcher({})
    run(DimensionResolver(FakeDetailSource(detail), fetcher).resolve(candidate()))
    assert fetcher.requested == []


def test_http_fetcher_maps_errors_without_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "blocked" in request.url.path:
            return httpx.Response(403, text="denied")
        if "slow" in request.url.path:
            raise httpx.ReadTimeout("slow", request=request)
        if "json" in request.url.path:
            return httpx.Response(200, json={"a": 1})
        return httpx.Response(200, html=page("<p>ok</p>"))

    fetcher = HttpPageFetcher(transport=httpx.MockTransport(handler))
    assert "ok" in run(fetcher.fetch("https://shop.example.com/page")).html
    for path, retryable in (("blocked", False), ("slow", True), ("json", False)):
        with pytest.raises(PageFetchError) as error:
            run(fetcher.fetch(f"https://shop.example.com/{path}"))
        assert error.value.retryable is retryable


def test_serpapi_immersive_details_are_normalized() -> None:
    body = {"product_results": {
        "about_the_product": {"features": [{"title": "Width", "value": "72\""}, {"title": "Depth", "value": "30\""},
                                           {"title": "Height", "value": "30.5\""}, {"title": "Bad"}]},
        "stores": [{"name": "west elm", "link": STORE}, {"name": "No link"}],
    }}
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body)

    source = SerpApiImmersiveProductDetailSource("secret", transport=httpx.MockTransport(handler))
    detail = run(source.fetch(candidate()))
    assert seen[0].url.params["engine"] == "google_immersive_product"
    assert seen[0].url.params["page_token"] == "token-1"
    assert detail.features == [("Width", '72"'), ("Depth", '30"'), ("Height", '30.5"')]
    assert detail.stores == [StoreLink("west elm", STORE)]
    assert run(source.fetch(candidate(detailPageToken=None))) is None


# ---- Endpoint -----------------------------------------------------------

def remembered_cache(tmp_path: Path, product: ProductCandidate) -> ProductSearchCache:
    cache = ProductSearchCache(tmp_path / "cache.json")
    cache.put("serpapi", "chair", "chair", [product])
    return cache


def post_dimensions(cache, resolver, body):
    app.dependency_overrides[get_product_cache] = lambda: cache
    app.dependency_overrides[get_dimension_resolver] = lambda: resolver
    app.dependency_overrides[get_resolution_cache] = lambda: ResolutionCache()
    try:
        return TestClient(app).post("/api/v1/products/dimensions", json=body)
    finally:
        app.dependency_overrides.clear()


def test_endpoint_resolves_a_previously_returned_product(tmp_path: Path) -> None:
    product = candidate()
    detail = ProductDetail(features=[("Width", "38 in"), ("Depth", "37 in"), ("Height", "33.5\"")])
    # A fresh cache instance proves the provider reference is persisted server-side.
    remembered_cache(tmp_path, product)
    cache = ProductSearchCache(tmp_path / "cache.json")
    source = FakeDetailSource(detail)
    response = post_dimensions(cache, DimensionResolver(source, FakeFetcher({})),
                               {"productId": product.id, "productUrl": product.productUrl})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified" and body["sourceType"] == "structured_metadata"
    assert body["widthMeters"] == pytest.approx(38 * IN)
    assert "detailPageToken" not in body
    assert source.calls == 1


def test_endpoint_rejects_unknown_or_mismatched_products(tmp_path: Path) -> None:
    product = candidate()
    cache = remembered_cache(tmp_path, product)
    resolver = DimensionResolver(FakeDetailSource(), FakeFetcher({}))
    assert post_dimensions(cache, resolver, {"productId": "serpapi:nope", "productUrl": product.productUrl}).status_code == 404
    response = post_dimensions(cache, resolver, {"productId": product.id, "productUrl": "http://127.0.0.1/evil"})
    assert response.status_code == 404
    assert post_dimensions(cache, resolver, {"productId": product.id, "productUrl": product.productUrl,
                                             "widthMeters": 1}).status_code == 422


def test_search_response_never_exposes_provider_token() -> None:
    assert "detailPageToken" not in candidate().model_dump()

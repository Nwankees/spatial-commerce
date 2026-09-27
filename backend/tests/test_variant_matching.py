from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.dimension_evidence import build_dimension_evidence
from app.dimension_llm import DimensionExtraction
from app.dimension_resolver import DimensionResolver
from app.page_fetcher import FetchedPage
from app.product_cache import ProductSearchCache
from app.product_details import ProductDetail, StoreLink, _selected_options
from app.product_models import ProductCandidate
from app.product_search_service import ProductSearchService
from app.query_builder import QueryPlanner
from app.variant_context import build_variant_context, choose_offer, extract_retailer_ids, parse_google_ids
from app.variant_scope import find_variant_records, ids_for_selected_options, record_evidence

IN = 0.0254
GRAY_URL = "https://www.target.com/p/chair-gray/-/A-93144009?TCID=OGS&AFID=google"
GRAY_LINE = "Dimensions (Overall): 39.4 inches (H) x 21.65 inches (W) x 22 inches (D)"
BEIGE_LINE = "Dimensions (Overall): 39.4 inches (H) x 22 inches (W) x 22 inches (D)"
GOOGLE_LINK = ("https://www.google.com/search?ibp=oshop&q=chair&prds=catalogid:3128998902942934176,"
               "productid:3281142218968418669,headlineOfferDocid:16214864048579896457,gpcid:6481,mid:5764")


def run(coro):
    return asyncio.run(coro)


def L(value, unit):
    return {"value": value, "unit": unit}


def extraction(evidence, **axes) -> DimensionExtraction:
    return DimensionExtraction.model_validate({"evidence_text": evidence, "raw_dimensions": evidence,
                                               "confidence": 0.9, "status": "verified", **axes})


def target_like_page(extra_child: dict | None = None) -> str:
    def child(tcin, color, line):
        return {"tcin": tcin, "item": {"product_description": {
            "title": f"Chair, {color}",
            "bullet_descriptions": [f"<B>{line.split(':')[0]}:</B>{line.split(':', 1)[1]}", "<B>Seat Dimensions:</B> 22.4 inches [W]"]},
            "package_dimensions": {"width": 21.65, "depth": 22.44, "dimension_unit_of_measure": "INCH"}}}
    product = {
        "tcin": "93143997",
        "variation_hierarchy": [{"name": "Color", "value": "Beige", "tcin": "93328976"},
                                {"name": "Color", "value": "Gray", "tcin": "93144009"}],
        "item": {"product_description": {"bullet_descriptions": ["<B>" + GRAY_LINE.replace(":", ":</B>", 1)]}},
        "children": [child("93328976", "Beige", BEIGE_LINE), child("93144009", "Gray", GRAY_LINE)]
                    + ([extra_child] if extra_child else []),
        "genAiConciseSummary": {"items": [{"description": "Overall dimensions: 50W x 50D x 50H inches"}]},
    }
    return f'<html><body><script type="application/json">{json.dumps({"props": {"product": product}})}</script></body></html>'


def candidate(**kw) -> ProductCandidate:
    fields = dict(id="serpapi:3128998902942934176", provider="serpapi", providerProductId="3128998902942934176",
                  title="VECELO Mid-Back Chair", price=71.99, retailer="Target", productUrl=GOOGLE_LINK,
                  detailPageToken="tok")
    fields.update(kw)
    return ProductCandidate(**fields)


class FakeDetail:
    name = "fake"

    def __init__(self, detail):
        self.detail = detail

    @property
    def configured(self):
        return True

    async def fetch(self, candidate):
        return self.detail


class FakeFetcher:
    def __init__(self, pages: dict[str, Any]):
        self.pages, self.requested = pages, []

    async def fetch(self, url):
        self.requested.append(url)
        return FetchedPage(url, self.pages[url])


class RecordingExtractor:
    """Returns a (correct) extraction for whatever single labeled line it is shown."""

    def __init__(self):
        self.evidence: list[str] = []

    async def extract(self, evidence: str) -> DimensionExtraction:
        self.evidence.append(evidence)
        line = next((l for l in evidence.split("\n") if l.startswith("Dimensions (Overall)")), None)
        if line is None:
            return extraction(None, status="unavailable") if False else DimensionExtraction.model_validate(
                {"confidence": 0.1, "status": "unavailable"})
        width = 21.65 if "21.65 inches (W)" in line else 22
        return extraction(line, width=L(width, "in"), depth=L(22, "in"), height=L(39.4, "in"))


FAMILY_FEATURES = [("Width", "22 in"), ("Depth", "22 in"), ("Height", "40 in")]


def gray_detail(features=FAMILY_FEATURES, **store_kw):
    store = dict(name="Target", url=GRAY_URL, title="VECELO Mid-Back Chair, Gray", price=71.99)
    store.update(store_kw)
    return ProductDetail(features=list(features), stores=[StoreLink(**store)], source_name="Google Shopping product details")


# ---- Identity extraction -------------------------------------------------

@pytest.mark.parametrize(("url", "slot", "value"), [
    (GRAY_URL, "itemId", "93144009"),
    ("https://www.walmart.com/ip/Arlopu-Lazy-Chair/17743069195?selectedSellerId=101&selectedOfferId=56AAC6D7", "itemId", "17743069195"),
    ("https://www.walmart.com/ip/Arlopu-Lazy-Chair/17743069195?selectedSellerId=101&selectedOfferId=56AAC6D7", "offerId", "56AAC6D7"),
    ("https://www.bestbuy.com/product/x/JXTH/sku/11047701?ref=212", "sku", "11047701"),
    ("https://belfurniture.com/products/alana-chair?variant=54823991771508&utm_source=google", "variantId", "54823991771508"),
    ("https://www.kohls.com/product/prd-8506928/loheer.jsp?skuid=67758630&CID=seo", "sku", "67758630"),
    ("https://www.wayfair.com/Desk-X119352385-L32-K~W008241274.html?refid=FR49&PiID%5B%5D=667945213", "sku", "W008241274"),
])
def test_retailer_ids_from_store_urls(url, slot, value) -> None:
    _, ids = extract_retailer_ids(url)
    assert getattr(ids, slot) == value


def test_google_ids_parsed_from_product_link() -> None:
    ids = parse_google_ids(GOOGLE_LINK)
    assert ids["headlineOfferDocid"] == "16214864048579896457" and ids["catalogid"] == "3128998902942934176"


def test_offer_chosen_only_when_unambiguous() -> None:
    stores = [StoreLink("Target", "https://t/1", None, 71.99), StoreLink("Target", "https://t/2", None, 89.99),
              StoreLink("Walmart", "https://w/1", None, 71.99)]
    offer, _ = choose_offer(stores, "Target", 71.99)
    assert offer.storeUrl == "https://t/1"
    assert choose_offer(stores, "Target", 50.0)[0] is None  # two Target offers, neither at this price
    assert choose_offer(stores, "Kohl's", 71.99)[0] is None


def test_selected_options_only_when_explicit() -> None:
    variants = [{"title": "Color", "items": [{"name": "Any Color"}, {"name": "Black", "selected": True}, {"name": "Gray"}]},
                {"title": "Size", "items": [{"name": "Any Size", "selected": True}]}]
    assert _selected_options(variants) == {"Color": "Black"}
    context = build_variant_context(GOOGLE_LINK, "Wayfair", 10.0,
                                    [StoreLink("Wayfair", "https://www.wayfair.com/x.html", "Hambrook Desk Color: Off-White", 10.0)])
    assert context.selectedOptions == {"color": "Off-White"} and context.identity == "options_only"


# ---- Variant scoping -----------------------------------------------------

def test_record_scoping_keeps_only_selected_variant_and_drops_ai_and_package_data() -> None:
    records = find_variant_records(target_like_page(), ["93144009"])
    evidence = record_evidence(records)
    assert GRAY_LINE in evidence
    assert "22 inches (W)" not in evidence and "50W" not in evidence and "22.44" not in evidence


def test_ai_generated_summaries_excluded_from_page_evidence() -> None:
    page = ('<html><body><div class="ai-summary">Overall dimensions: 43W x 19D x 29H inches</div>'
            '<script>window.__D={"genAiConciseSummary":{"items":[{"description":"Overall dimensions: 43.03W x 19.09D x 29.76H inches"}]}}</script>'
            '<p>Product size: 30 W x 20 D x 29 H in</p></body></html>')
    evidence = build_dimension_evidence(page)
    assert "30 W x 20 D x 29 H in" in evidence and "43" not in evidence


# ---- Resolver policy -----------------------------------------------------

def test_exact_item_record_outranks_family_specs_without_conflict() -> None:
    extractor = RecordingExtractor()
    fetcher = FakeFetcher({GRAY_URL: target_like_page()})
    resolver = DimensionResolver(FakeDetail(gray_detail()), fetcher, extractor=extractor)
    result = run(resolver.resolve(candidate()))
    assert fetcher.requested == [GRAY_URL]  # exact offer fetched even though family specs had W+D
    assert result.status == "verified" and result.variantScope == "exact_variant"
    assert (result.widthMeters, result.depthMeters, result.heightMeters) == pytest.approx((0.54991, 0.5588, 1.00076))
    assert "93144009" in result.variantIdentity
    assert len(extractor.evidence) >= 1 and all("22 inches (W)" not in e for e in extractor.evidence)
    assert "conflict" not in (result.message or "")


def test_structured_pairs_in_variant_record_need_no_llm() -> None:
    page = {"variants": [{"id": 111, "title": "Gray", "specs": [{"name": "Width", "value": "30 in"}, {"name": "Depth", "value": "20 in"}]},
                         {"id": 222, "title": "Black", "specs": [{"name": "Width", "value": "40 in"}, {"name": "Depth", "value": "25 in"}]}]}
    url = "https://shop.example.com/products/chair?variant=111"
    html = f'<script type="application/json">{json.dumps(page)}</script>'
    detail = ProductDetail(features=FAMILY_FEATURES, stores=[StoreLink("Shop", url, "Chair", 50.0)])
    extractor = RecordingExtractor()
    result = run(DimensionResolver(FakeDetail(detail), FakeFetcher({url: html}), extractor=extractor)
                 .resolve(candidate(retailer="Shop", price=50.0)))
    assert result.sourceType == "structured_metadata" and result.variantScope == "exact_variant"
    assert result.widthMeters == pytest.approx(30 * IN) and extractor.evidence == []


def test_exact_offer_id_scopes_record() -> None:
    page = {"offers": [{"offerId": "OFF1", "attributes": [{"name": "Width", "value": "30 in"}, {"name": "Depth", "value": "18 in"}]},
                       {"offerId": "OFF2", "attributes": [{"name": "Width", "value": "36 in"}, {"name": "Depth", "value": "18 in"}]}]}
    url = "https://shop.example.com/item?offerId=OFF2"
    html = f'<script type="application/json">{json.dumps(page)}</script>'
    detail = ProductDetail(stores=[StoreLink("Shop", url, None, 50.0)])
    result = run(DimensionResolver(FakeDetail(detail), FakeFetcher({url: html})).resolve(candidate(retailer="Shop", price=50.0)))
    assert result.widthMeters == pytest.approx(36 * IN) and result.variantScope == "exact_variant"


def test_selected_options_identify_single_variant() -> None:
    url = "https://www.target.com/p/chair/-/A-93143997"  # parent item page; the option picks the child
    detail = ProductDetail(stores=[StoreLink("Target", "https://shop.example.com/chair", "Chair Color: Gray", 71.99)])
    fetcher = FakeFetcher({"https://shop.example.com/chair": target_like_page()})
    result = run(DimensionResolver(FakeDetail(detail), fetcher, extractor=RecordingExtractor()).resolve(candidate()))
    assert result.widthMeters == pytest.approx(0.54991) and result.variantScope == "exact_variant"
    assert url  # (documentation only)


def test_ambiguous_option_match_stays_unresolved() -> None:
    extra = {"tcin": "99999999", "item": {"product_description": {"bullet_descriptions": [BEIGE_LINE]}}}
    html = target_like_page().replace('{"name": "Color", "value": "Beige", "tcin": "93328976"}',
                                      '{"name": "Color", "value": "Gray", "tcin": "93328976"}')
    detail = ProductDetail(stores=[StoreLink("Target", "https://shop.example.com/chair", "Chair Color: Gray", 71.99)])
    assert ids_for_selected_options(html, {"color": "Gray"}) == {"93328976", "93144009"}
    result = run(DimensionResolver(FakeDetail(detail), FakeFetcher({"https://shop.example.com/chair": html}),
                                   extractor=RecordingExtractor()).resolve(candidate()))
    assert result.status == "unavailable" and "several retailer items" in result.message
    assert extra


def test_no_matching_record_falls_back_to_page_with_sibling_conflicts() -> None:
    url = "https://www.target.com/p/chair/-/A-11111111"  # not in the page data
    detail = ProductDetail(stores=[StoreLink("Target", url, None, 71.99)])
    resolver = DimensionResolver(FakeDetail(detail), FakeFetcher({url: target_like_page()}), extractor=RecordingExtractor())
    result = run(resolver.resolve(candidate()))
    assert result.widthMeters is None  # 21.65 vs 22 across siblings: never picked
    assert result.status in ("partial", "unavailable")


def test_without_exact_identity_family_specs_short_circuit() -> None:
    detail = ProductDetail(features=FAMILY_FEATURES, stores=[StoreLink("Target", "https://shop.example.com/c", None, 99.0)])
    fetcher = FakeFetcher({})
    result = run(DimensionResolver(FakeDetail(detail), fetcher, extractor=RecordingExtractor()).resolve(candidate()))
    assert fetcher.requested == [] and result.variantScope == "product_family" and result.widthMeters == pytest.approx(22 * IN)


def test_exact_page_without_dimensions_falls_back_to_family() -> None:
    fetcher = FakeFetcher({GRAY_URL: "<html><body><h1>Chair</h1></body></html>"})
    result = run(DimensionResolver(FakeDetail(gray_detail()), fetcher, extractor=RecordingExtractor()).resolve(candidate()))
    assert result.variantScope == "product_family" and result.widthMeters == pytest.approx(22 * IN)


# ---- Variant-aware dedupe ------------------------------------------------

class QueryProvider:
    name = "serpapi"

    def __init__(self, outcomes):
        self.outcomes = outcomes

    @property
    def configured(self):
        return True

    async def search(self, query, limit):
        return self.outcomes.get(query.text, [])


def offer(pid: str, doc: str | None, position: int, token: str) -> ProductCandidate:
    prds = f"productid:{pid}" + (f",headlineOfferDocid:{doc}" if doc else "")
    return ProductCandidate(id=f"serpapi:{pid}", provider="serpapi", providerProductId=pid, position=position,
                            title="VECELO Mid-Back Chair", price=71.99, retailer="Target",
                            productUrl=f"https://www.google.com/search?prds={prds}", detailPageToken=token)


def test_different_offers_with_same_title_are_not_merged(tmp_path) -> None:
    from app.models import VisualProductAnalysis
    analysis = VisualProductAnalysis(objectDetected=True, confidence=0.9, category="chair", searchQueries=["chair a", "chair b"])
    gray, beige = offer("1", "DOC-GRAY", 2, "tok-gray"), offer("2", "DOC-BEIGE", 1, "tok-beige")
    gray_again = offer("3", "DOC-GRAY", 1, "tok-gray-2")
    svc = ProductSearchService(QueryProvider({"chair a": [gray, beige], "chair b": [gray_again]}),
                               ProductSearchCache(tmp_path / "c.json"), planner=QueryPlanner(2))
    products = run(svc.search(analysis, 5)).products
    by_doc = {parse_google_ids(p.productUrl)["headlineOfferDocid"]: p for p in products}
    assert set(by_doc) == {"DOC-GRAY", "DOC-BEIGE"}  # variants kept apart
    merged = by_doc["DOC-GRAY"]
    assert merged.position == 1 and merged.detailPageToken == "tok-gray-2"  # best-ranked concrete offer kept
    assert merged.matchedQueries == ["chair a", "chair b"]

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from app.dimension_evidence import build_dimension_evidence
from app.dimension_extraction import extract_from_html
from app.dimension_llm import (
    DimensionExtraction,
    DimensionExtractionError,
    OllamaDimensionExtractor,
    extraction_json_schema,
    validate_extraction,
)
from app.dimension_resolver import DimensionResolver
from app.page_fetcher import FetchedPage, PageFetchError
from app.product_details import ProductDetail, StoreLink
from app.product_models import ProductCandidate

IN = 0.0254
TARGET_TEXT = "Dimensions (Overall): 39.4 inches (H) x 21.65 inches (W) x 22 inches (D)"
STORE = "https://www.target.com/p/accent-chair/-/A-1"


def run(coro):
    return asyncio.run(coro)


def L(value, unit):
    return {"value": value, "unit": unit}


def extraction(evidence: str | None = TARGET_TEXT, status: str = "verified", **axes) -> DimensionExtraction:
    return DimensionExtraction.model_validate(
        {"width": None, "depth": None, "height": None, "raw_dimensions": evidence, "evidence_text": evidence,
         "confidence": 0.9, "status": status, "reason": None, **axes}
    )


def candidate(**overrides) -> ProductCandidate:
    fields = dict(id="serpapi:1", provider="serpapi", providerProductId="1", title="Accent Chair", price=129.0,
                  retailer="Target", productUrl="https://www.google.com/search?prds=1", detailPageToken="tok")
    fields.update(overrides)
    return ProductCandidate(**fields)


class FakeDetail:
    name = "fake"

    def __init__(self, detail: ProductDetail) -> None:
        self.detail = detail

    @property
    def configured(self) -> bool:
        return True

    async def fetch(self, candidate):
        return self.detail


class FakeFetcher:
    def __init__(self, pages: dict[str, Any]) -> None:
        self.pages, self.requested = pages, []

    async def fetch(self, url):
        self.requested.append(url)
        outcome = self.pages[url]
        if isinstance(outcome, Exception):
            raise outcome
        return FetchedPage(url, outcome)


class FakeExtractor:
    def __init__(self, result: DimensionExtraction | Exception) -> None:
        self.result, self.evidence = result, []

    async def extract(self, evidence: str) -> DimensionExtraction:
        self.evidence.append(evidence)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def target_page() -> str:
    return f"""<html><head><title>Chair : Target</title></head><body>
      <nav>Shop width departments 123</nav><div class="cookie-banner">We use cookies 1 2 3</div>
      <h1>Upholstered Accent Chair</h1>
      <div data-test="item-details-specifications"><div><b>Dimensions (Overall):</b> 39.4 inches (H) x 21.65 inches (W) x 22 inches (D)</div>
      <div><b>Weight:</b> 25 pounds</div></div>
      <section class="reviews"><p>Dimensions were 30 x 30 x 30 in my room</p></section></body></html>"""


def resolver(page_html: str, extractor, features=()) -> tuple[DimensionResolver, FakeFetcher]:
    fetcher = FakeFetcher({STORE: page_html})
    detail = ProductDetail(features=list(features), stores=[StoreLink("Target", STORE)], source_name="provider")
    return DimensionResolver(FakeDetail(detail), fetcher, extractor=extractor), fetcher


# ---- Root cause & Target case ------------------------------------------

def test_deterministic_parser_misses_target_format() -> None:
    """Documents the Milestone 5 gap: '(H)'-style labels were not recognized."""
    assert extract_from_html(target_page(), STORE) == []


def test_target_style_hxwxd_resolves_verified_through_llm_extraction() -> None:
    extractor = FakeExtractor(extraction(width=L(21.65, "in"), depth=L(22, "in"), height=L(39.4, "in")))
    r, _ = resolver(target_page(), extractor)
    result = run(r.resolve(candidate()))
    assert result.status == "verified" and result.sourceType == "page_text_llm"
    assert result.heightMeters == pytest.approx(1.00076)
    assert result.widthMeters == pytest.approx(0.54991)
    assert result.depthMeters == pytest.approx(0.5588)
    assert result.sourceName == "Target" and result.sourceUrl == STORE
    assert result.rawDimensions == TARGET_TEXT
    evidence = extractor.evidence[0]
    assert TARGET_TEXT in evidence
    assert "cookies" not in evidence and "30 x 30 x 30" not in evidence and "Shop width" not in evidence


def test_structured_sources_take_priority_over_llm() -> None:
    extractor = FakeExtractor(extraction(width=L(21.65, "in"), depth=L(22, "in"), height=L(39.4, "in")))
    r, fetcher = resolver(target_page(), extractor, features=[("Width", "21 in"), ("Depth", "20 in")])
    result = run(r.resolve(candidate()))
    assert result.sourceType == "structured_metadata" and result.widthMeters == pytest.approx(21 * IN)
    assert extractor.evidence == [] and fetcher.requested == []

    jsonld = '<script type="application/ld+json">{"@type":"Product","width":"20 in","depth":"19 in"}</script>'
    r, _ = resolver(target_page().replace("</head>", jsonld + "</head>"), extractor)
    result = run(r.resolve(candidate()))
    assert result.sourceType == "json_ld" and extractor.evidence == []


# ---- Deterministic validation ------------------------------------------

@pytest.mark.parametrize(
    ("source", "axes", "expected"),
    [
        ('30"W x 28"D x 35"H', dict(width=L(30, "in"), depth=L(28, "in"), height=L(35, "in")), (30 * IN, 28 * IN, 35 * IN)),
        ("Width: 120 cm\nDepth: 60.5 cm\nHeight: 75 cm", dict(width=L(120, "cm"), depth=L(60.5, "cm"), height=L(75, "cm")), (1.2, 0.605, 0.75)),
        ("Overall: 1.2 m W x 600 mm D; 2.5 ft H", dict(width=L(1.2, "m"), depth=L(600, "mm"), height=L(2.5, "ft")), (1.2, 0.6, 0.762)),
        ("Dimensions (W x D x H): 76 x 71 x 89 cm", dict(width=L(76, "cm"), depth=L(71, "cm"), height=L(89, "cm")), (0.76, 0.71, 0.89)),
        ("Dimensions: 60cmW x 40cmD x 75cmH", dict(width=L(60, "cm"), depth=L(40, "cm"), height=L(75, "cm")), (0.6, 0.4, 0.75)),
        ("Product size: 30 W x 28 D x 35 H in", dict(width=L(30, "in"), depth=L(28, "in"), height=L(35, "in")), (30 * IN, 28 * IN, 35 * IN)),
    ],
)
def test_explicit_formats_and_units_validate(source, axes, expected) -> None:
    result = validate_extraction(extraction(source, **axes), source)
    assert (result.width, result.depth, result.height) == pytest.approx(expected)


def test_partial_dimensions_stay_partial() -> None:
    source = "Width 24.5 in; Height 30 in"
    result = validate_extraction(extraction(source, "partial", width=L(24.5, "in"), height=L(30, "in")), source)
    assert result.width == pytest.approx(24.5 * IN) and result.depth is None and result.height == pytest.approx(0.762)


@pytest.mark.parametrize("source", [
    "Dimensions: 39.4 x 21.65 x 22 in",                                           # unlabeled
    "Assembled Product Dimensions (L x W x H): 21.65 x 22 x 39.4 Inches",         # L is ambiguous
    'Package Dimensions: 21.65"W x 22"D x 39.4"H',                                # not the product
])
def test_unlabeled_ambiguous_and_package_dimensions_rejected(source: str) -> None:
    result = validate_extraction(
        extraction(source, width=L(21.65, "in"), depth=L(22, "in"), height=L(39.4, "in")), source)
    assert result.known_axes == 0


def test_hallucinated_number_axis_and_unit_are_rejected() -> None:
    hallucinated = validate_extraction(extraction(width=L(21.65, "in"), depth=L(24, "in"), height=L(39.4, "in")), TARGET_TEXT)
    assert hallucinated.depth is None and hallucinated.width is not None
    swapped = validate_extraction(extraction(width=L(22, "in"), depth=L(21.65, "in"), height=L(39.4, "in")), TARGET_TEXT)
    assert swapped.width is None and swapped.depth is None and swapped.height is not None
    wrong_unit = validate_extraction(extraction(width=L(21.65, "cm"), depth=L(22, "in")), TARGET_TEXT)
    assert wrong_unit.width is None and wrong_unit.depth == pytest.approx(0.5588)
    seat = "Width: 120 cm\nSeat Height: 45 cm"
    assert validate_extraction(extraction(seat, height=L(45, "cm")), seat).height is None


def test_evidence_not_in_source_is_not_trusted_and_unavailable_is_respected() -> None:
    fake = validate_extraction(extraction('Width: 99 in (W)', width=L(99, "in")), TARGET_TEXT)
    assert fake.width is None and fake.rejected
    assert validate_extraction(extraction(None, "unavailable", width=L(21.65, "in")), TARGET_TEXT).known_axes == 0


def test_non_positive_values_rejected() -> None:
    assert validate_extraction(extraction(width=L(0, "in"), height=L(-39.4, "in")), TARGET_TEXT).known_axes == 0


# ---- Ollama extractor ---------------------------------------------------

def ollama(handler) -> OllamaDimensionExtractor:
    return OllamaDimensionExtractor("http://ollama.test", "qwen3:4b-instruct", transport=httpx.MockTransport(handler))


def reply(content: Any, status: int = 200):
    def handler(request):
        if status != 200:
            return httpx.Response(status, json=content)
        return httpx.Response(200, json={"message": {"content": content if isinstance(content, str) else json.dumps(content)}})
    return handler


def test_request_format_is_text_only_structured_and_deterministic() -> None:
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return reply(extraction(width=L(21.65, "in")).model_dump())(request)

    result = run(ollama(handler).extract(TARGET_TEXT))
    body = seen[0]
    assert body["model"] == "qwen3:4b-instruct" and body["stream"] is False and body["think"] is False
    assert body["options"]["temperature"] == 0 and body["format"] == extraction_json_schema()
    assert all("images" not in m for m in body["messages"]) and TARGET_TEXT in body["messages"][1]["content"]
    assert result.width.value == 21.65


@pytest.mark.parametrize(("handler", "retryable", "match"), [
    (reply("not json"), True, "malformed"),
    (reply({"width": {"value": 1}, "status": "verified"}), True, "malformed"),
    (reply({"error": "model 'qwen3:4b-instruct' not found"}, status=404), False, "not installed"),
    (reply({"error": "boom"}, status=500), True, "HTTP 500"),
])
def test_extractor_errors(handler, retryable: bool, match: str) -> None:
    with pytest.raises(DimensionExtractionError, match=match) as error:
        run(ollama(handler).extract(TARGET_TEXT))
    assert error.value.retryable is retryable


def test_extractor_timeout_and_unreachable() -> None:
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    for handler, match in ((slow, "timed out"), (down, "not reachable")):
        with pytest.raises(DimensionExtractionError, match=match) as error:
            run(ollama(handler).extract(TARGET_TEXT))
        assert error.value.retryable


def test_resolver_reports_llm_failures_safely() -> None:
    r, _ = resolver(target_page(), FakeExtractor(DimensionExtractionError("The local dimension model timed out.", retryable=True)))
    result = run(r.resolve(candidate()))
    assert result.status == "unavailable" and result.retryable and "timed out" in result.message
    r, _ = resolver(target_page(), FakeExtractor(DimensionExtractionError("model not installed", retryable=False)))
    result = run(r.resolve(candidate()))
    assert result.status == "unavailable" and not result.retryable


def test_llm_hallucination_through_resolver_is_discarded() -> None:
    r, _ = resolver(target_page(), FakeExtractor(extraction(width=L(30, "in"), depth=L(30, "in"))))
    assert run(r.resolve(candidate())).status == "unavailable"


# ---- Fetch failures & preprocessing -------------------------------------

@pytest.mark.parametrize(("error", "retryable"), [
    (PageFetchError("Retailer blocked automated access (HTTP 403).", retryable=False), False),
    (PageFetchError("Retailer page timed out.", retryable=True), True),
])
def test_page_fetch_failures_are_safe(error, retryable: bool) -> None:
    extractor = FakeExtractor(extraction(width=L(21.65, "in")))
    fetcher = FakeFetcher({STORE: error})
    detail = ProductDetail(stores=[StoreLink("Target", STORE)])
    result = run(DimensionResolver(FakeDetail(detail), fetcher, extractor=extractor).resolve(candidate()))
    assert result.status == "unavailable" and result.retryable is retryable and extractor.evidence == []


def test_large_page_is_preprocessed_and_capped() -> None:
    filler = "".join(f"<p>Lorem ipsum filler paragraph {i} about comfort and style.</p>" for i in range(20000))
    reviews = "".join(f'<div class="review-item">Overall great, dimensions {i} x {i} x {i} in</div>' for i in range(500))
    script = '<script>window.__DATA__={"specs":"\\u003cB\\u003eDimensions (Overall):\\u003c/B\\u003e 39.4 inches (H) x 21.65 inches (W) x 22 inches (D)"}</script>'
    html = f"<html><body>{filler}{reviews}{script}</body></html>"
    assert len(html) > 1_000_000
    evidence = build_dimension_evidence(html, max_chars=2000)
    assert len(evidence) <= 2000
    assert TARGET_TEXT in evidence  # found inside embedded page data
    assert "Lorem ipsum" not in evidence and "great" not in evidence


def test_evidence_keeps_spec_tables_and_jsonld() -> None:
    html = """<html><head><script type="application/ld+json">{"@type":"Product","name":"Desk",
      "description":"Solid wood. Overall dimensions: 48 in W x 24 in D x 30 in H."}</script></head>
      <body><table><tr><th>Width</th><td>48 in</td></tr><tr><th>Color</th><td>Oak</td></tr></table></body></html>"""
    evidence = build_dimension_evidence(html)
    assert "Width: 48 in" in evidence and "48 in W x 24 in D x 30 in H" in evidence and "Oak" not in evidence


# ---- Regression: shape of a real Target product page (embedded page data) ----

TARGET_SCRIPT = (
    '<script>window.__DATA__={"images":[{"__typename":"ProductGalleryAssetImage","alt":"mesh backrest, and '
    'dimensions labeled as 20.9\\" height, 22\\" width, 18.1\\" depth","url":"https://target.scene7.com/is/image/Target/x"}],'
    '"reviews":[{"text":"Overall, great chair for the price.","author":{"nickname":"Alina"},"rating":{"value":5}}],'
    '"package_dimensions":{"depth":22.44,"dimension_unit_of_measure":"INCH","height":11.42,"width":21.65},'
    '"bullets":["\\u003cB\\u003eDimensions (Overall):\\u003c/B\\u003e 39.4 inches (H) x 21.65 inches (W) x 22 inches (D)",'
    '"\\u003cB\\u003eSeat Dimensions:\\u003c/B\\u003e 22.4 inches [W] \\u0026 20.9 inches [D]"]}</script>'
)


def test_target_embedded_data_evidence_drops_reviews_images_and_package_data() -> None:
    evidence = build_dimension_evidence(f"<html><body><h1>Office Chair</h1>{TARGET_SCRIPT}</body></html>")
    assert TARGET_TEXT in evidence
    assert "great chair" not in evidence and "labeled as" not in evidence and "22.44" not in evidence


def test_target_seat_dimensions_never_become_overall_dimensions() -> None:
    evidence = build_dimension_evidence(f"<html><body>{TARGET_SCRIPT}</body></html>")
    seat = validate_extraction(extraction(None, width=L(22.4, "in"), depth=L(20.9, "in")), evidence)
    assert seat.known_axes == 0
    overall = validate_extraction(
        extraction(TARGET_TEXT, width=L(21.65, "in"), depth=L(22, "in"), height=L(39.4, "in")), evidence)
    assert (overall.width, overall.depth, overall.height) == pytest.approx((0.54991, 0.5588, 1.00076))


def test_model_answering_null_axes_is_unavailable_not_guessed() -> None:
    """Observed with qwen3:4b-instruct before the schema required axis objects."""
    lazy = extraction(width=None, depth=None, height=None)
    assert validate_extraction(lazy, TARGET_TEXT).known_axes == 0
    assert extraction_json_schema()["properties"]["width"]["required"] == ["value", "unit"]


# ---- Regression from the live Target check -------------------------------

REAL_TARGET_EVIDENCE = (
    's up to: 1"," Dimensions (Overall): 39.4 inches (H) x 21.65 inches (W) x 22 inches (D)",'
    '" Seat Dimensions: 22.4 inches [W] & 20.9 inches [D]"," Seat Back Dimensions: 20.9 inches (H) x 16.9 inches (W)",'
    '" Arm Dimensions: 0 inches (H) x 0 inches (W)"," Weight: 21.42 pounds"'
)
VARIANT = '\n" Dimensions (Overall): 39.4 inches (H) x 22 inches (W) x 22 inches (D)"," Seat Dimensions: 17.3 inches [W]'


class SequenceExtractor:
    def __init__(self, *results: DimensionExtraction) -> None:
        self.results, self.evidence = list(results), []

    async def extract(self, evidence: str) -> DimensionExtraction:
        self.evidence.append(evidence)
        return self.results.pop(0)


def test_live_misread_is_caught_and_second_pass_on_quoted_line_recovers() -> None:
    from app.dimension_llm import extract_and_validate
    wrong = extraction(width=L(21.65, "in"), depth=L(39.4, "in"), height=L(39.4, "in"))  # what the model returned live
    right = extraction(width=L(21.65, "in"), depth=L(22, "in"), height=L(39.4, "in"))
    extractor = SequenceExtractor(wrong, right)
    result = run(extract_and_validate(extractor, REAL_TARGET_EVIDENCE))
    assert extractor.evidence[1] == TARGET_TEXT  # second pass sees only the quoted statement
    assert (result.width, result.depth, result.height) == pytest.approx((0.54991, 0.5588, 1.00076))


def test_variants_with_different_overall_sizes_drop_the_conflicting_axis() -> None:
    right = extraction(width=L(21.65, "in"), depth=L(22, "in"), height=L(39.4, "in"))
    result = validate_extraction(right, REAL_TARGET_EVIDENCE + VARIANT)
    assert result.width is None  # 21.65 in vs 22 in across variants: refuse to pick one
    assert result.depth == pytest.approx(0.5588) and result.height == pytest.approx(1.00076)
    assert any("conflicting" in r for r in result.rejected)


def test_second_pass_never_merges_with_first() -> None:
    from app.dimension_llm import extract_and_validate
    first = extraction(width=L(21.65, "in"), depth=L(39.4, "in"))
    second = extraction(height=L(39.4, "in"))
    result = run(extract_and_validate(SequenceExtractor(first, second), REAL_TARGET_EVIDENCE))
    assert (result.width, result.depth, result.height) == (pytest.approx(0.54991), None, None)

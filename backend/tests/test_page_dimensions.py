"""Milestone 6 dimension pipeline: page -> structured representation -> (fake) model ->
deterministic provenance validation -> normalized dimensions."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
import pytest

from app.ar_geometry import assign_axes_by_mesh, compute_scale
from app.dimension_resolver import DimensionResolver
from app.dimension_semantic import (
    DimensionExtractionError,
    OllamaSemanticDimensionExtractor,
    SemanticExtraction,
    semantic_json_schema,
    source_occurrences,
    validate_semantic,
)
from app.page_fetcher import FetchedPage
from app.page_representation import build_page_representation, build_provider_representation
from app.product_details import ProductDetail, StoreLink
from app.product_models import ProductCandidate

IN = 0.0254
WALMART_URL = "https://www.walmart.com/ip/Organizer-Box/17743069195"
TARGET_GRAY = "https://www.target.com/p/chair-gray/-/A-93144009"


def run(coro):
    return asyncio.run(coro)


# ---- fixtures ---------------------------------------------------------------------

def next_data(product: dict, idml: dict, extra: dict | None = None) -> str:
    data = {"props": {"pageProps": {"initialData": {"data": {"product": product, "idml": idml, **(extra or {})}}}}}
    return (f'<html><head><title>{product.get("name")} - Walmart.com</title>'
            f'<link rel="canonical" href="{WALMART_URL}"></head><body>'
            '<nav><a>Departments</a><a>Dimensions 99 x 99 x 99 in</a></nav>'
            f'<h1>{product.get("name")}</h1>'
            '<div id="cookie-banner">We use cookies 12 x 12 x 12 in</div>'
            '<section data-testid="reviews-section"><h2>Customer reviews</h2><p>Mine is 50 x 50 x 50 in</p></section>'
            f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script>'
            '<footer>About us 1 x 1 x 1 in</footer></body></html>')


def walmart_page(dims: str = "13.39 x 4.13 x 4.13 in", extra_specs: list | None = None) -> str:
    specs = [{"name": "Material", "value": "Plastic"}, {"name": "Dimensions", "value": dims},
             {"name": "Weight", "value": "0.47 lb"}] + (extra_specs or [])
    product = {"usItemId": "17743069195", "name": "Desk Organizer Box", "brand": "Acme",
               "variantsMap": {"999": {"usItemId": "55500011", "specs": [{"name": "Dimensions", "value": "20 x 5 x 5 in"}]}}}
    return next_data(product, {"specifications": specs})


def target_page() -> str:
    def child(tcin, color, line):
        return {"tcin": tcin, "item": {"product_description": {
            "title": f"Chair, {color}", "bullet_descriptions": [line, "Seat Dimensions: 22.4 inches (W)"]},
            "package_dimensions": {"width": 21.65, "depth": 22.44, "dimension_unit_of_measure": "INCH"}}}
    product = {"tcin": "93143997",
               "children": [child("93328976", "Beige", "Dimensions (Overall): 39.4 inches (H) x 22 inches (W) x 22 inches (D)"),
                            child("93144009", "Gray", "Dimensions (Overall): 39.4 inches (H) x 21.65 inches (W) x 22 inches (D)")],
               "genAiConciseSummary": {"items": [{"description": "Overall dimensions: 50W x 50D x 50H inches"}]}}
    return (f'<html><head><link rel="canonical" href="https://www.target.com/p/chair/-/A-93143997"></head><body>'
            f'<script type="application/json">{json.dumps({"props": {"product": product}})}</script></body></html>')


def html_page(body: str, head: str = "") -> str:
    return f"<html><head>{head}</head><body>{body}</body></html>"


def candidate(**kw) -> ProductCandidate:
    fields = dict(id="serpapi:1", provider="serpapi", providerProductId="1", title="Desk Organizer Box", price=9.99,
                  retailer="Walmart", productUrl=WALMART_URL)
    fields.update(kw)
    return ProductCandidate(**fields)


# ---- fake model ---------------------------------------------------------------------

def find(page: dict, name: str | None = None, contains: str | None = None) -> str:
    for section in page["sections"]:
        for entry in section.get("entries", []):
            if (name and (entry["name"] or "").lower() == name.lower()) or (contains and contains in entry["value"]):
                return entry["id"]
        for text in section.get("text", []):
            if contains and contains in text["text"]:
                return text["id"]
    raise AssertionError(f"entry {name or contains!r} not in the page data")


def answer(source: str, values: list[tuple[float, str]], raw: str | None = None, labels=None, mapping=None,
           status="verified") -> dict:
    return {"status": status, "rawText": raw,
            "dimensions": [{"value": v, "unit": u, "sourceId": source, "label": (labels or [None] * 6)[i]}
                           for i, (v, u) in enumerate(values)],
            "axisMapping": mapping or {"widthIndex": None, "depthIndex": None, "heightIndex": None,
                                       "confidence": 0.0, "reason": "not labeled"},
            "reason": None}


class ScriptedExtractor:
    def __init__(self, fn: Callable[[dict], dict]):
        self.fn = fn
        self.pages: list[dict] = []
        self.last_raw: str | None = None

    async def extract(self, rep, page_json: str) -> SemanticExtraction:
        page = json.loads(page_json)
        self.pages.append(page)
        out = self.fn(page)
        self.last_raw = json.dumps(out)
        return SemanticExtraction.model_validate(out)


class FakeFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages, self.requested = pages, []

    async def fetch(self, url):
        self.requested.append(url)
        return FetchedPage(url, self.pages[url])


class FakeDetail:
    name = "fake"

    def __init__(self, detail):
        self.detail = detail

    @property
    def configured(self):
        return True

    async def fetch(self, candidate):
        return self.detail


def resolve(pages: dict[str, str], fn, cand: ProductCandidate | None = None, detail=None):
    extractor = ScriptedExtractor(fn)
    resolver = DimensionResolver(FakeDetail(detail) if detail else None, FakeFetcher(pages), extractor=extractor)
    return run(resolver.resolve(cand or candidate())), extractor, resolver


# ---- representation -------------------------------------------------------------------

def test_walmart_spec_table_is_represented_structurally_without_noise() -> None:
    rep = build_page_representation(walmart_page(), WALMART_URL, product_title="Desk Organizer Box",
                                    retailer="Walmart", exact_ids=["17743069195"])
    page = json.loads(rep.to_llm_json())
    specs = next(s for s in page["sections"] if s["heading"].endswith("specifications"))
    assert {"name": "Dimensions", "value": "13.39 x 4.13 x 4.13 in"} in [
        {k: e[k] for k in ("name", "value")} for e in specs["entries"]]
    assert {"Material", "Weight"} <= {e["name"] for e in specs["entries"]}  # not filtered by "dimension-likeness"
    blob = json.dumps(page)
    for noise in ("99 x 99", "12 x 12", "50 x 50", "1 x 1 x 1", "20 x 5 x 5"):  # nav, cookie, review, footer, sibling
        assert noise not in blob
    assert page["pageProductTitle"] == "Desk Organizer Box" and not rep.variant_mismatch


def test_dl_headings_tables_and_unanticipated_labels_are_kept() -> None:
    body = ('<h2>Size Details</h2><dl><dt>Overall Size</dt><dd>60 x 30 x 75 cm</dd><dt>Color</dt><dd>Oak</dd></dl>'
            '<h3>Specs</h3><table><tr><th>Item Measurements</th><td>23.6 in wide, 11.8 in deep, 29.5 in tall</td></tr>'
            '<tr><td>Frobnication index</td><td>7</td></tr></table><p>Made of solid oak.</p>')
    page = json.loads(build_page_representation(html_page(body), "https://x.example/p").to_llm_json())
    entries = {e["name"]: e["value"] for s in page["sections"] for e in s.get("entries", [])}
    assert entries["Overall Size"] == "60 x 30 x 75 cm" and entries["Frobnication index"] == "7"
    assert "23.6 in wide" in entries["Item Measurements"]
    headings = [s["heading"] for s in page["sections"]]
    assert "Size Details" in headings and "Specs" in headings
    assert any(t["text"] == "Made of solid oak." for s in page["sections"] for t in s.get("text", []))


def test_budget_prefers_structured_sections_and_records_what_was_shown() -> None:
    body = "".join(f"<p>Paragraph {i} " + "lorem ipsum " * 20 + "</p>" for i in range(200))
    body += '<table><tr><td>Dimensions</td><td>10 x 20 x 30 in</td></tr></table>'
    rep = build_page_representation(html_page(body), "https://x.example/p")
    text = rep.to_llm_json(3000)
    assert len(text) <= 3000 and "10 x 20 x 30 in" in text and rep.truncated
    shown = rep.included_ids
    hidden = [e.id for e, _ in rep.all_entries() if e.id not in shown]
    assert hidden  # validation only accepts shown entries (tested below)


def test_exact_variant_scoping_excludes_siblings_and_ai_summaries() -> None:
    rep = build_page_representation(target_page(), TARGET_GRAY, exact_ids=["93144009"])
    blob = rep.to_llm_json()
    assert "21.65 inches (W)" in blob
    assert "22 inches (W)" not in blob and "50W" not in blob
    gray = next(s for s in rep.sections if any("21.65 inches" in e.value for e in s.entries + s.text))
    assert gray.scope == "exact_record"


def test_variant_mismatch_detected() -> None:
    rep = build_page_representation(walmart_page(), WALMART_URL.replace("17743069195", "11111111"),
                                    exact_ids=["22222222"])
    assert rep.variant_mismatch


# ---- validation -------------------------------------------------------------------------

def walmart_rep(dims="13.39 x 4.13 x 4.13 in", extra_specs=None):
    rep = build_page_representation(walmart_page(dims, extra_specs), WALMART_URL, exact_ids=["17743069195"])
    return rep, json.loads(rep.to_llm_json())


def test_unlabeled_triple_survives_with_uncertain_axes() -> None:
    rep, page = walmart_rep()
    result = validate_semantic(SemanticExtraction.model_validate(
        answer(find(page, "Dimensions"), [(13.39, "in"), (4.13, "in"), (4.13, "in")], "13.39 x 4.13 x 4.13 in")), rep)
    assert result.accepted and result.values_m == pytest.approx([13.39 * IN, 4.13 * IN, 4.13 * IN], abs=1e-5)
    assert result.axis.widthIndex is None and result.axis.heightIndex is None and result.axis.confidence == 0
    assert "not labeled" in result.axis.reason
    assert result.source_path.endswith("> Dimensions") and result.source_type == "embedded_json"


def test_unlabeled_choice_uses_only_an_exact_labeled_duplicate_for_axes() -> None:
    body = ('<h2>Highlights</h2><table><tr><td>Dimensions</td><td>28.70 x 25.20 x 38.20 in</td></tr></table>'
            '<p>Chair Dimensions: 25.20&quot;D x 28.70&quot;W x 38.20&quot;H. '
            'Ottoman Dimensions: 16.1&quot;D x 18.1&quot;W x 17.7&quot;H.</p>')
    rep = build_page_representation(html_page(body), "https://x.example/p")
    page = json.loads(rep.to_llm_json())
    result = validate_semantic(SemanticExtraction.model_validate(
        answer(find(page, "Dimensions"), [(28.70, "in"), (25.20, "in"), (38.20, "in")])), rep)
    assert result.accepted
    assert (result.axis.widthIndex, result.axis.depthIndex, result.axis.heightIndex) == (0, 1, 2)
    assert result.axis.source == "labels" and "corroborating source" in result.axis.reason
    assert any("duplicated with explicit axis labels" in note for note in result.notes)


@pytest.mark.parametrize(("values", "fragment"), [
    ([(13.4, "in"), (4.13, "in"), (4.13, "in")], "does not occur"),           # altered number
    ([(13.39, "cm"), (4.13, "in"), (4.13, "in")], "source unit is in"),       # wrong unit
    ([(13.39, "in"), (4.13, "in"), (4.13, "in"), (4.13, "in")], "more than three"),
    ([(13.39, "in"), (4.13, "in"), (9.0, "in")], "does not occur"),           # extra invented value
])
def test_hallucinated_or_altered_values_rejected(values, fragment) -> None:
    rep, page = walmart_rep()
    extraction = SemanticExtraction.model_validate(answer(find(page, "Dimensions"), values))
    result = validate_semantic(extraction, rep)
    assert not result.accepted and fragment in " ".join(result.errors)


def test_wrong_source_paths_rejected() -> None:
    rep, page = walmart_rep()
    for source in ("E9999", find(page, "Material")):
        extraction = SemanticExtraction.model_validate(answer(source, [(13.39, "in"), (4.13, "in"), (4.13, "in")]))
        assert not validate_semantic(extraction, rep).accepted
    # values split across two different sections
    body = ('<h2>A</h2><table><tr><td>Width</td><td>30 in</td></tr></table>'
            '<h2>B</h2><table><tr><td>Depth</td><td>20 in</td></tr></table>')
    rep2 = build_page_representation(html_page(body), "https://x.example/p")
    page2 = json.loads(rep2.to_llm_json())
    mixed = {"status": "partial", "dimensions": [
        {"value": 30, "unit": "in", "sourceId": find(page2, "Width"), "label": "W"},
        {"value": 20, "unit": "in", "sourceId": find(page2, "Depth"), "label": "D"}]}
    result = validate_semantic(SemanticExtraction.model_validate(mixed), rep2)
    assert not result.accepted and "different sections" in result.errors[0]


def test_entry_not_shown_to_model_is_rejected() -> None:
    body = "".join(f"<p>Paragraph {i} " + "lorem " * 50 + "</p>" for i in range(100)) + "<p>Size: 10 x 20 x 30 in</p>"
    rep = build_page_representation(html_page(body), "https://x.example/p")
    rep.to_llm_json(1500)
    hidden = next(e for e, _ in rep.all_entries() if "10 x 20 x 30" in e.value)
    assert hidden.id not in rep.included_ids
    ext = SemanticExtraction.model_validate(answer(hidden.id, [(10, "in"), (20, "in"), (30, "in")]))
    assert not validate_semantic(ext, rep).accepted


def test_package_and_part_fields_rejected_after_choice_but_visible_to_model() -> None:
    body = ('<table><tr><td>Package Dimensions</td><td>30 x 20 x 10 in</td></tr>'
            '<tr><td>Seat Dimensions</td><td>20 x 18 x 4 in</td></tr>'
            '<tr><td>Product Dimensions</td><td>28 x 18 x 8 in</td></tr></table>')
    rep = build_page_representation(html_page(body), "https://x.example/p")
    page = json.loads(rep.to_llm_json())
    for name, vals in (("Package Dimensions", (30, 20, 10)), ("Seat Dimensions", (20, 18, 4))):
        ext = SemanticExtraction.model_validate(answer(find(page, name), [(v, "in") for v in vals]))
        result = validate_semantic(ext, rep)
        assert not result.accepted and "not an overall product dimension" in result.errors[0]
    ok = validate_semantic(SemanticExtraction.model_validate(
        answer(find(page, "Product Dimensions"), [(28, "in"), (18, "in"), (8, "in")])), rep)
    assert ok.accepted


def test_package_dimensions_only_resolve_unavailable() -> None:
    body = '<table><tr><td>Package Dimensions</td><td>30 x 20 x 10 in</td></tr></table>'

    def choose_package(page):
        return answer(find(page, "Package Dimensions"), [(30, "in"), (20, "in"), (10, "in")])

    result, _, _ = resolve({WALMART_URL: html_page(body)}, choose_package)
    assert result.status == "unavailable"
    assert result.dimensionsMeters is None and "not an overall product dimension" in result.message


def test_uncertain_and_unavailable_are_respected() -> None:
    rep, page = walmart_rep()
    for status in ("uncertain", "unavailable"):
        ext = SemanticExtraction.model_validate(
            answer(find(page, "Dimensions"), [(13.39, "in")], status=status) | {"reason": "two overall sizes"})
        result = validate_semantic(ext, rep)
        assert not result.accepted and status in result.errors[0]


@pytest.mark.parametrize(("value", "expected"), [
    ('30"W x 20"D x 35"H', (0, 1, 2)),
    ("39.4 inches (H) x 21.65 inches (W) x 22 inches (D)", (1, 2, 0)),
    ("Width 30 in, Depth 20 in, Height 35 in", (0, 1, 2)),
])
def test_axis_mapping_only_from_source_labels(value, expected) -> None:
    body = f"<table><tr><td>Dimensions</td><td>{value}</td></tr></table>"
    rep = build_page_representation(html_page(body), "https://x.example/p")
    page = json.loads(rep.to_llm_json())
    nums = [o.value for o in source_occurrences(next(e for e, _ in rep.all_entries() if e.name == "Dimensions"))]
    ext = SemanticExtraction.model_validate(answer(find(page, "Dimensions"), [(n, "in") for n in nums]))
    axis = validate_semantic(ext, rep).axis
    assert (axis.widthIndex, axis.depthIndex, axis.heightIndex) == expected and axis.confidence == 1.0


def test_declared_orders_and_length_labels() -> None:
    body = ('<table><tr><td>Dimensions (W x D x H)</td><td>76 x 71 x 89 cm</td></tr>'
            '<tr><td>Assembled Product Dimensions (L x W x H)</td><td>28.70 x 25.20 x 38.20 Inches</td></tr></table>')
    rep = build_page_representation(html_page(body), "https://x.example/p")
    page = json.loads(rep.to_llm_json())
    wdh = validate_semantic(SemanticExtraction.model_validate(
        answer(find(page, "Dimensions (W x D x H)"), [(76, "cm"), (71, "cm"), (89, "cm")])), rep).axis
    assert (wdh.widthIndex, wdh.depthIndex, wdh.heightIndex) == (0, 1, 2)
    lwh = validate_semantic(SemanticExtraction.model_validate(
        answer(find(page, "Assembled Product Dimensions (L x W x H)"), [(28.7, "in"), (25.2, "in"), (38.2, "in")],
               mapping={"widthIndex": 0, "depthIndex": 1, "heightIndex": 2, "confidence": 0.9, "reason": "guess"})), rep).axis
    assert (lwh.widthIndex, lwh.depthIndex, lwh.heightIndex) == (1, 0, 2)
    assert lwh.source == "labels"


def test_model_axis_guess_on_unlabeled_values_is_dropped() -> None:
    rep, page = walmart_rep("28.70 x 25.20 x 38.20 in")
    ext = SemanticExtraction.model_validate(answer(find(page, "Dimensions"), [(28.7, "in"), (25.2, "in"), (38.2, "in")],
                                                   mapping={"widthIndex": 0, "depthIndex": 1, "heightIndex": 2,
                                                            "confidence": 0.8, "reason": "typical order"}))
    result = validate_semantic(ext, rep)
    assert result.accepted and result.axis.heightIndex is None and "ignored" in result.axis.reason


# ---- resolver --------------------------------------------------------------------------

def test_walmart_unlabeled_dimensions_resolve_verified() -> None:
    def fn(page):
        return answer(find(page, "Dimensions"), [(13.39, "in"), (4.13, "in"), (4.13, "in")], "13.39 x 4.13 x 4.13 in")
    result, extractor, _ = resolve({WALMART_URL: walmart_page()}, fn)
    assert result.status == "verified" and result.variantScope == "exact_variant_page"
    assert result.dimensionsMeters == pytest.approx([0.34011, 0.1049, 0.1049], abs=1e-5)
    assert result.widthMeters is None and result.axisMapping.heightIndex is None
    assert result.sourceType == "embedded_json" and result.extractionMethod == "llm"
    assert result.sourcePath.endswith("Dimensions") and result.rawDimensions == "13.39 x 4.13 x 4.13 in"
    assert len(extractor.pages) == 1


def test_walmart_seat_width_coincidence_does_not_confuse_provenance() -> None:
    extra = [{"name": "Seat width", "value": "28.7 in"}, {"name": "Seat back width", "value": "25.2 in"}]

    def fn(page):
        return answer(find(page, "Dimensions"), [(28.70, "in"), (25.20, "in"), (38.20, "in")])
    result, _, _ = resolve({WALMART_URL: walmart_page("28.70 x 25.20 x 38.20 in", extra)}, fn)
    assert result.status == "verified" and result.dimensionsMeters == pytest.approx([0.72898, 0.64008, 0.97028], abs=1e-5)


def test_target_exact_variant_resolves_labeled_axes() -> None:
    cand = candidate(retailer="Target", productUrl=TARGET_GRAY, title="VECELO Chair")

    def fn(page):
        return answer(find(page, contains="21.65 inches (W)"), [(39.4, "in"), (21.65, "in"), (22, "in")])
    result, extractor, _ = resolve({TARGET_GRAY: target_page()}, fn, cand)
    assert result.status == "verified" and result.variantScope == "exact_variant"
    assert (result.widthMeters, result.depthMeters, result.heightMeters) == pytest.approx((0.54991, 0.5588, 1.00076))
    assert "22 inches (W)" not in json.dumps(extractor.pages)


def test_json_ld_explicit_fields_use_fast_path_without_model() -> None:
    ld = {"@context": "https://schema.org", "@type": "Product", "name": "Desk", "sku": "17743069195",
          "width": {"@type": "QuantitativeValue", "value": 47.2, "unitCode": "INH"},
          "depth": {"@type": "QuantitativeValue", "value": 23.6, "unitCode": "INH"},
          "height": {"@type": "QuantitativeValue", "value": 29.5, "unitCode": "INH"}}
    page = html_page("<h1>Desk</h1>", f'<script type="application/ld+json">{json.dumps(ld)}</script>')
    result, extractor, _ = resolve({WALMART_URL: page}, lambda p: pytest.fail("model must not run"))
    assert result.status == "verified" and result.extractionMethod == "structured" and extractor.pages == []
    assert result.widthMeters == pytest.approx(47.2 * IN, abs=1e-4) and result.axisMapping.source == "structured_fields"


def test_json_ld_additional_property_goes_through_model() -> None:
    ld = {"@type": "Product", "name": "Desk", "additionalProperty": [
        {"@type": "PropertyValue", "name": "Overall Size", "value": "120 x 60 x 75 cm"}]}
    page = html_page("", f'<script type="application/ld+json">{json.dumps(ld)}</script>')
    result, _, _ = resolve({WALMART_URL: page},
                           lambda p: answer(find(p, "Overall Size"), [(120, "cm"), (60, "cm"), (75, "cm")]))
    assert result.status == "verified" and result.sourceType == "json_ld"
    assert result.dimensionsMeters == pytest.approx([1.2, 0.6, 0.75])


def test_embedded_product_json_flat_object() -> None:
    state = {"product": {"sku": "17743069195", "details": {"assembledSize": "24 in W x 18 in D x 30 in H",
                                                           "weightLbs": 12}}}
    page = html_page(f'<script>window.__STATE__ = {json.dumps(state)};</script>')
    result, _, _ = resolve({WALMART_URL: page},
                           lambda p: answer(find(p, contains="24 in W"), [(24, "in"), (18, "in"), (30, "in")]))
    assert result.status == "verified" and result.sourceType == "embedded_json"
    assert (result.widthMeters, result.depthMeters, result.heightMeters) == pytest.approx((24 * IN, 18 * IN, 30 * IN), abs=1e-5)


def test_multiple_plausible_sets_become_unavailable_with_reason() -> None:
    body = ('<table><tr><td>Dimensions</td><td>30 x 20 x 35 in</td></tr>'
            '<tr><td>Dimensions (large)</td><td>36 x 24 x 40 in</td></tr></table>')

    def fn(page):
        return answer(find(page, "Dimensions"), [(30, "in")], status="uncertain") | {"reason": "two overall sizes listed"}
    result, _, _ = resolve({WALMART_URL: html_page(body)}, fn)
    assert result.status == "unavailable" and "uncertain" in result.message


def test_selected_variant_mismatch_skips_page() -> None:
    wrong = walmart_page().replace(f'href="{WALMART_URL}"', 'href="https://www.walmart.com/ip/x/99999999"')
    url = "https://www.walmart.com/ip/x/12345678"
    cand = candidate(productUrl=url)
    result, extractor, _ = resolve({url: wrong.replace("17743069195", "88888888")}, lambda p: pytest.fail("no model"), cand)
    assert result.status == "unavailable" and "different item" in result.message and extractor.pages == []


def test_malformed_model_output_is_retryable_unavailable() -> None:
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "{not json"}})
    extractor = OllamaSemanticDimensionExtractor("http://ollama.test", "qwen3:4b-instruct",
                                                 transport=httpx.MockTransport(handler))
    resolver = DimensionResolver(None, FakeFetcher({WALMART_URL: walmart_page()}), extractor=extractor)
    result = run(resolver.resolve(candidate()))
    assert result.status == "unavailable" and result.retryable and "malformed" in result.message


def test_request_format_is_structured_deterministic_and_page_based() -> None:
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(answer("E1", [], status="unavailable"))}})
    extractor = OllamaSemanticDimensionExtractor("http://ollama.test", "qwen3:4b-instruct", num_ctx=8192,
                                                 transport=httpx.MockTransport(handler))
    rep = build_page_representation(walmart_page(), WALMART_URL, product_title="Desk Organizer Box")
    run(extractor.extract(rep, rep.to_llm_json()))
    assert seen["model"] == "qwen3:4b-instruct" and seen["think"] is False and seen["stream"] is False
    assert seen["options"] == {"temperature": 0, "num_ctx": 8192} and seen["format"] == semantic_json_schema()
    prompt = seen["messages"][1]["content"]
    assert "13.39 x 4.13 x 4.13 in" in prompt and "Desk Organizer Box" in prompt and "images" not in seen
    assert 'status MUST be\n  "verified"' in prompt
    assert 'Never use "uncertain" merely because' in prompt
    assert "cite the labeled occurrence" in prompt


def test_extractor_http_errors() -> None:
    for status, retryable in ((404, False), (500, True)):
        extractor = OllamaSemanticDimensionExtractor("http://o.test", "m", transport=httpx.MockTransport(
            lambda r, s=status: httpx.Response(s, json={"error": "model not found" if s == 404 else "boom"})))
        rep = build_page_representation(walmart_page(), WALMART_URL)
        with pytest.raises(DimensionExtractionError) as info:
            run(extractor.extract(rep, rep.to_llm_json()))
        assert info.value.retryable is retryable


def test_provider_family_specs_go_through_the_same_pipeline() -> None:
    detail = ProductDetail(features=[("Product Size", "47 x 24 x 30 inches"), ("Color", "Black")],
                           stores=[], source_name="Google Shopping product details")
    cand = candidate(productUrl="https://www.google.com/search?ibp=oshop&q=x", detailPageToken="tok", retailer=None)
    result, _, _ = resolve({}, lambda p: answer(find(p, "Product Size"), [(47, "in"), (24, "in"), (30, "in")]),
                           cand, detail)
    assert result.status == "verified" and result.variantScope == "product_family"
    assert result.sourceType == "structured_metadata" and result.sourcePath.startswith("Google Shopping")
    rep = build_provider_representation(detail.features, product_title="x", retailer=None, source_name="P")
    assert rep.sections[0].scope == "family"


def test_nothing_is_invented_when_page_has_no_dimensions() -> None:
    body = "<table><tr><td>Color</td><td>Red</td></tr></table>"
    result, _, _ = resolve({WALMART_URL: html_page(body)},
                           lambda p: {"status": "unavailable", "reason": "no dimensions on the page"})
    assert result.status == "unavailable" and result.dimensionsMeters is None


# ---- M6 mesh-assisted axis assignment -------------------------------------------------------

def test_mesh_assisted_mapping_uses_only_proportions() -> None:
    # mesh: x 1.0, y 0.3, z 0.3 (a long box lying along x); verified unlabeled 13.39 x 4.13 x 4.13 in
    values = [13.39 * IN, 4.13 * IN, 4.13 * IN]
    width, depth, height = assign_axes_by_mesh((1.0, 0.3, 0.29), values)
    assert width == pytest.approx(13.39 * IN) and {depth, height} == {4.13 * IN}
    scale = compute_scale((1.0, 0.3, 0.29), width, depth, height)
    assert (1.0 * scale.scale_x, 0.3 * scale.scale_y, 0.29 * scale.scale_z) == pytest.approx((width, height, depth))
    # labeled height: only width vs depth is decided by the mesh
    w, d, h = assign_axes_by_mesh((0.5, 0.9, 0.7), [0.7290, 0.6401, 0.9703], height_index=2)
    assert h == pytest.approx(0.9703) and (w, d) == pytest.approx((0.6401, 0.7290))

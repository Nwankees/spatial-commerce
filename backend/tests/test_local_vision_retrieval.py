from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_analysis_service, get_product_search_service
from app.models import Hypothesis, VisibleSpecification, VisualProductAnalysis
from app.ollama_vision import OllamaVisionService, analysis_json_schema, enforce_conservative_analysis, ollama_health
from app.product_cache import ProductSearchCache
from app.product_models import ProductCandidate
from app.product_providers import (
    ProductProviderNotConfiguredError,
    ProductProviderTimeoutError,
    ProductProviderUnavailableError,
)
from app.product_reranker import ProductReranker, ScoredCandidate
from app.product_search_service import ProductSearchFailedError, ProductSearchService
from app.query_builder import QueryBuildError, QueryPlanner
from app.vision_errors import VisionMalformedResponseError, VisionTimeoutError, VisionUnavailableError


def run(coro):
    return asyncio.run(coro)


def jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 6), (30, 30, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def charger_analysis(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = dict(
        objectDetected=True,
        category="laptop charger",
        subcategory="usb-c laptop power adapter",
        brand={"value": "Lenovo", "confidence": 0.62, "evidence": "logo"},
        modelFamily=None,
        visibleText=["lenovo", "65W", "USB-C"],
        color="black",
        materials=["plastic"],
        style=[],
        shape="compact rectangular brick with detachable cable",
        distinctiveFeatures=["rounded corners", "USB-C cable"],
        visibleSpecifications=[{"name": "power", "value": "65W", "evidence": "printed '65W' on label"}],
        searchQueries=["Lenovo 65W USB-C laptop charger", "black USB-C laptop power adapter", "laptop charger"],
        confidence=0.8,
        uncertaintyNotes="Brand read from a partially visible logo.",
        message=None,
    )
    fields.update(overrides)
    return fields


def ollama_service(handler, **kwargs) -> OllamaVisionService:
    return OllamaVisionService("http://ollama.test", "qwen3-vl:30b", transport=httpx.MockTransport(handler), **kwargs)


def chat_reply(content: Any, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json=content)
        text = content if isinstance(content, str) else json.dumps(content)
        return httpx.Response(200, json={"model": "qwen3-vl:30b", "message": {"role": "assistant", "content": text},
                                         "done": True, "total_duration": 4_000_000_000, "eval_count": 120})
    return handler


# ---- Ollama -------------------------------------------------------------

def test_ollama_request_format_and_structured_success() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return chat_reply(charger_analysis())(request)

    analysis = run(ollama_service(handler).analyze(jpeg_bytes(), "something cheaper"))
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/chat"
    assert body["model"] == "qwen3-vl:30b" and body["stream"] is False and body["think"] is False
    assert body["format"]["type"] == "object" and "$defs" not in json.dumps(body["format"])
    assert base64.b64decode(body["messages"][1]["images"][0]).startswith(b"\xff\xd8")
    assert "shopping" in body["messages"][0]["content"] and "something cheaper" in body["messages"][1]["content"]
    assert analysis.objectDetected and analysis.brand.value == "Lenovo"
    assert analysis.brand.confidence == 0.62  # uncertainty preserved
    assert len(analysis.searchQueries) == 3


def test_schema_sent_to_ollama_requires_every_key() -> None:
    schema = analysis_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert "$ref" not in json.dumps(schema)


@pytest.mark.parametrize("content", ["not json at all", "```json\n{\"objectDetected\": tru", ""])
def test_malformed_json_is_retryable_error(content: str) -> None:
    with pytest.raises(VisionMalformedResponseError):
        run(ollama_service(chat_reply(content)).analyze(jpeg_bytes(), None))


def test_fenced_json_is_accepted() -> None:
    fenced = "```json\n" + json.dumps(charger_analysis()) + "\n```"
    assert run(ollama_service(chat_reply(fenced)).analyze(jpeg_bytes(), None)).objectDetected


@pytest.mark.parametrize(
    "bad",
    [charger_analysis(confidence=1.7), charger_analysis(unexpected="x"), charger_analysis(brand={"value": "Lenovo"}),
     {"category": "chair"}],
)
def test_invalid_schema_is_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(VisionMalformedResponseError):
        run(ollama_service(chat_reply(bad)).analyze(jpeg_bytes(), None))


def test_timeout_connection_failure_and_missing_model() -> None:
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(VisionTimeoutError):
        run(ollama_service(timeout).analyze(jpeg_bytes(), None))
    with pytest.raises(VisionUnavailableError, match="not reachable"):
        run(ollama_service(refused).analyze(jpeg_bytes(), None))
    with pytest.raises(VisionUnavailableError, match="not installed"):
        run(ollama_service(chat_reply({"error": "model 'qwen3-vl:30b' not found"}, status=404)).analyze(jpeg_bytes(), None))
    with pytest.raises(VisionUnavailableError, match="HTTP 500"):
        run(ollama_service(chat_reply({"error": "out of memory"}, status=500)).analyze(jpeg_bytes(), None))


def test_no_product_and_low_confidence_become_safe_no_object() -> None:
    none = run(ollama_service(chat_reply({**charger_analysis(), "objectDetected": False, "confidence": 0.05,
                                          "message": "Point at one product."})).analyze(jpeg_bytes(), None))
    assert not none.objectDetected and none.searchQueries == [] and none.message
    low = run(ollama_service(chat_reply(charger_analysis(confidence=0.1)), min_confidence=0.2).analyze(jpeg_bytes(), None))
    assert not low.objectDetected and low.searchQueries == [] and low.message


def test_incomplete_detection_is_retryable_error() -> None:
    incomplete = charger_analysis(category=None, subcategory=None, searchQueries=[])
    with pytest.raises(VisionMalformedResponseError, match="incomplete"):
        run(ollama_service(chat_reply(incomplete)).analyze(jpeg_bytes(), None))


def test_health_reports_model_availability() -> None:
    def tags(request):
        return httpx.Response(200, json={"models": [{"name": "qwen3-vl:30b"}, {"name": "gemma3:12b"}]})

    ok = run(ollama_health("http://ollama.test", "qwen3-vl:30b", transport=httpx.MockTransport(tags)))
    assert ok["reachable"] and ok["modelInstalled"]
    missing = run(ollama_health("http://ollama.test", "qwen3-vl:8b", transport=httpx.MockTransport(tags)))
    assert missing["reachable"] and not missing["modelInstalled"]

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    assert run(ollama_health("http://ollama.test", "x", transport=httpx.MockTransport(down)))["reachable"] is False


# ---- Analysis contract --------------------------------------------------

def test_optional_brand_and_model_may_be_absent() -> None:
    analysis = VisualProductAnalysis.model_validate(charger_analysis(brand=None, modelFamily=None, visibleText=[],
                                                                      visibleSpecifications=[]))
    assert analysis.brand is None and analysis.modelFamily is None


def test_unsupported_specs_and_text_claims_are_not_fabricated() -> None:
    analysis = VisualProductAnalysis.model_validate(charger_analysis(
        visibleText=["lenovo"],
        visibleSpecifications=[{"name": "power", "value": "65W", "evidence": "looks like a 65W brick"},
                               {"name": "brand text", "value": "lenovo", "evidence": "printed"}],
        modelFamily={"value": "ThinkPad", "confidence": 0.4, "evidence": "visible_text"},
    ))
    cleaned = enforce_conservative_analysis(analysis, 0.2)
    assert [s.value for s in cleaned.visibleSpecifications] == ["lenovo"]  # 65W not visible -> dropped
    assert cleaned.modelFamily.evidence == "design_resemblance"            # not in visible text -> downgraded
    assert cleaned.modelFamily.confidence == 0.4                           # uncertainty kept


def test_legacy_search_keywords_still_accepted() -> None:
    analysis = VisualProductAnalysis(objectDetected=True, confidence=0.9, category="chair",
                                     searchKeywords=["beige chair", "beige chair", " accent  chair "])
    assert analysis.searchQueries == ["beige chair", "accent chair"]
    assert analysis.searchKeywords == analysis.searchQueries


# ---- Query planning -----------------------------------------------------

def test_planner_caps_and_dedupes_queries() -> None:
    analysis = VisualProductAnalysis.model_validate(charger_analysis(searchQueries=[
        "Lenovo 65W USB-C laptop charger", "laptop charger lenovo 65w usb-c", "black USB-C laptop power adapter",
        "laptop charger", "usb c charger", "power brick"]))
    queries = [q.text for q in QueryPlanner(max_queries=3).plan(analysis)]
    assert queries == ["Lenovo 65W USB-C laptop charger", "black USB-C laptop power adapter", "laptop charger"]


def test_planner_falls_back_to_deterministic_query() -> None:
    analysis = VisualProductAnalysis.model_validate(charger_analysis(searchQueries=[]))
    assert [q.text for q in QueryPlanner(3).plan(analysis)] == ["black plastic usb-c laptop power adapter"]
    with pytest.raises(QueryBuildError):
        QueryPlanner(3).plan(VisualProductAnalysis(objectDetected=False, confidence=0.1))


# ---- Multi-query retrieval ----------------------------------------------

def product(pid: str, title: str, position: int, retailer: str = "Store", token: str | None = None, **kw) -> ProductCandidate:
    return ProductCandidate(
        id=f"serpapi:{pid}", provider="serpapi", providerProductId=pid, position=position, title=title,
        price=kw.pop("price", 39.99), priceText="$39.99", currency="USD", retailer=retailer,
        productUrl=f"https://www.google.com/shopping/product/{pid}", detailPageToken=token, **kw,
    )


class QueryProvider:
    name = "serpapi"

    def __init__(self, outcomes: dict[str, Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    @property
    def configured(self) -> bool:
        return True

    async def search(self, query, limit):
        self.calls.append(query.text)
        outcome = self.outcomes.get(query.text, [])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


Q1, Q2, Q3 = "Lenovo 65W USB-C laptop charger", "black USB-C laptop power adapter", "laptop charger"


def analysis() -> VisualProductAnalysis:
    return enforce_conservative_analysis(VisualProductAnalysis.model_validate(charger_analysis()), 0.2)


def service(provider, tmp_path: Path, **kw) -> ProductSearchService:
    return ProductSearchService(provider, ProductSearchCache(tmp_path / "cache.json"), planner=QueryPlanner(3), **kw)


def test_executes_capped_queries_merges_and_dedupes(tmp_path: Path) -> None:
    lenovo = product("1", "Lenovo 65W USB-C Laptop Charger Power Adapter", 3, "Lenovo", token="tok-1")
    lenovo_again = product("1", "Lenovo 65W USB-C Laptop Charger Power Adapter", 1, "Lenovo", token="tok-1")
    same_listing = product("9", "lenovo 65w usb-c laptop charger power adapter", 2, "lenovo")
    generic = product("2", "Universal 65W USB C Charger", 1, "Amazon")
    provider = QueryProvider({Q1: [generic, lenovo], Q2: [lenovo_again, same_listing], Q3: []})
    response = run(service(provider, tmp_path).search(analysis(), 5))

    assert provider.calls == [Q1, Q2, Q3]
    assert [q.status for q in response.queries] == ["ok", "ok", "empty"]
    ids = [p.providerProductId for p in response.products]
    assert sorted(ids) == ["1", "2"]  # id and title+retailer duplicates merged
    merged = next(p for p in response.products if p.providerProductId == "1")
    assert merged.position == 1                  # best-ranked occurrence kept
    assert merged.matchedQueries == [Q1, Q2]
    assert merged.detailPageToken == "tok-1"     # M5 metadata preserved


def test_reranking_prefers_better_semantic_match(tmp_path: Path) -> None:
    unrelated = product("2", "Stainless Steel Water Bottle 32 oz", 1, "Target")
    generic = product("3", "USB C Charger Block", 2, "Walmart")
    lenovo = product("1", "Lenovo 65W USB-C Laptop Charger Power Adapter Black", 6, "Best Buy")
    response = run(service(QueryProvider({Q1: [unrelated, generic, lenovo]}), tmp_path).search(analysis(), 5))
    assert [p.providerProductId for p in response.products] == ["1", "3", "2"]
    assert response.products[0].retrievalScore > response.products[1].retrievalScore


def test_reranker_ignores_low_confidence_brand() -> None:
    reranker = ProductReranker()
    weak = VisualProductAnalysis.model_validate(charger_analysis(brand={"value": "Lenovo", "confidence": 0.3, "evidence": "design_resemblance"}))
    item = ScoredCandidate(product("1", "Lenovo charger", 1), 1, [Q1])
    reranker.score(weak, item, Q1)
    assert "brand" not in item.breakdown


def test_one_failed_query_does_not_fail_request(tmp_path: Path) -> None:
    provider = QueryProvider({Q1: ProductProviderTimeoutError("slow"), Q2: [product("1", "Black USB-C Power Adapter", 1)]})
    response = run(service(provider, tmp_path).search(analysis(), 5))
    assert response.resultSource == "live" and len(response.products) == 1
    assert [q.status for q in response.queries] == ["failed", "ok", "empty"]
    assert "1 of 3 searches failed" in response.message


def test_all_queries_failing_without_cache_raises(tmp_path: Path) -> None:
    error = ProductProviderUnavailableError("down")
    with pytest.raises(ProductSearchFailedError):
        run(service(QueryProvider({Q1: error, Q2: error, Q3: error}), tmp_path).search(analysis(), 5))


def test_all_queries_failing_uses_cache(tmp_path: Path) -> None:
    lenovo = product("1", "Lenovo 65W USB-C Laptop Charger", 1, token="tok-1")
    run(service(QueryProvider({Q1: [lenovo]}), tmp_path).search(analysis(), 5))
    error = ProductProviderUnavailableError("down")
    cached = run(service(QueryProvider({Q1: error, Q2: error, Q3: error}), tmp_path).search(analysis(), 5))
    assert cached.resultSource == "cache" and cached.cachedAt is not None
    assert cached.products[0].providerProductId == "1" and cached.products[0].detailPageToken == "tok-1"


def test_per_query_cache_used_when_plan_changed(tmp_path: Path) -> None:
    run(service(QueryProvider({Q2: [product("1", "Black USB-C Power Adapter", 1)]}), tmp_path).search(analysis(), 5))
    error = ProductProviderUnavailableError("down")
    other = VisualProductAnalysis.model_validate(charger_analysis(searchQueries=[Q2, "usb c brick"]))
    cached = run(service(QueryProvider({Q2: error, "usb c brick": error, "black plastic usb-c laptop power adapter": error}),
                         tmp_path).search(other, 5))
    assert cached.resultSource == "cache" and cached.products[0].providerProductId == "1"


def test_missing_provider_key_is_not_masked(tmp_path: Path) -> None:
    with pytest.raises(ProductProviderNotConfiguredError):
        run(service(QueryProvider({Q1: ProductProviderNotConfiguredError("no key")}), tmp_path).search(analysis(), 5))


# ---- Endpoints ----------------------------------------------------------

def test_analyze_endpoint_uses_ollama_and_reports_timing() -> None:
    app.dependency_overrides[get_analysis_service] = lambda: ollama_service(chat_reply(charger_analysis()))
    try:
        response = TestClient(app).post("/api/v1/analyze", json={
            "imageBase64": base64.b64encode(jpeg_bytes()).decode(), "mimeType": "image/jpeg"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["brand"] == {"value": "Lenovo", "confidence": 0.62, "evidence": "logo"}
    assert "X-Analysis-Duration-Ms" in response.headers


@pytest.mark.parametrize(("handler", "status"), [
    (chat_reply("nope"), 502),
    (chat_reply({"error": "model not found"}, status=404), 502),
])
def test_analyze_endpoint_errors_are_retryable(handler, status: int) -> None:
    app.dependency_overrides[get_analysis_service] = lambda: ollama_service(handler)
    try:
        response = TestClient(app).post("/api/v1/analyze", json={
            "imageBase64": base64.b64encode(jpeg_bytes()).decode(), "mimeType": "image/jpeg"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == status and "retry" in response.json()["detail"].lower()


def test_search_endpoint_accepts_rich_analysis_and_m5_registry_still_works(tmp_path: Path) -> None:
    cache = ProductSearchCache(tmp_path / "cache.json")
    svc = ProductSearchService(QueryProvider({Q1: [product("1", "Lenovo 65W USB-C Laptop Charger", 1, token="tok-1")]}),
                               cache, planner=QueryPlanner(3))
    app.dependency_overrides[get_product_search_service] = lambda: svc
    try:
        response = TestClient(app).post("/api/v1/products/search", json={"analysis": charger_analysis()})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["products"][0]["matchedQueries"] == [Q1] and "detailPageToken" not in body["products"][0]
    assert len(body["queries"]) == 3
    found = ProductSearchCache(tmp_path / "cache.json").find_product("serpapi:1")
    assert found is not None and found.detailPageToken == "tok-1"

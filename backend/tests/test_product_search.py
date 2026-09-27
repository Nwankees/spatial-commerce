from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_product_search_service
from app.models import VisualProductAnalysis
from app.product_cache import ProductSearchCache
from app.product_providers import (
    ProductProviderMalformedResponseError,
    ProductProviderNotConfiguredError,
    ProductProviderTimeoutError,
    ProductProviderUnavailableError,
)
from app.product_search_service import ProductSearchFailedError, ProductSearchService
from app.query_builder import ProductQueryBuilder, QueryBuildError
from app.serpapi_provider import SerpApiProductSearchProvider

SECRET = "test-serpapi-secret-value"


def chair_analysis(**overrides: Any) -> VisualProductAnalysis:
    fields: dict[str, Any] = dict(
        objectDetected=True,
        category="lounge chair",
        subcategory="accent chair",
        color="beige",
        materials=["fabric", "wood"],
        style=["modern", "minimalist"],
        shape="rounded",
        searchKeywords=["beige modern lounge chair", "rounded fabric accent chair"],
        confidence=0.91,
    )
    fields.update(overrides)
    return VisualProductAnalysis(**fields)


def serpapi_payload() -> dict[str, Any]:
    return {
        "search_metadata": {"status": "Success"},
        "shopping_results": [
            {
                "position": 1,
                "title": "Mid-Century Upholstered Accent Chair, Beige",
                "product_id": "1111",
                "product_link": "https://www.google.com/shopping/product/1111",
                "source": "Wayfair",
                "price": "$149.99",
                "extracted_price": 149.99,
                "rating": 4.6,
                "reviews": 1320,
                "thumbnail": "https://encrypted-tbn0.gstatic.com/shopping?q=a",
            },
            {
                "position": 2,
                "title": "Boucle Barrel Chair",
                "product_id": "2222",
                "link": "https://www.walmart.com/ip/2222",
                "product_link": "https://www.google.com/shopping/product/2222",
                "source": "Walmart",
                "price": "$89.00",
                "extracted_price": 89.0,
                "thumbnail": "https://encrypted-tbn0.gstatic.com/shopping?q=b",
            },
            # Unusable: no price.
            {"position": 3, "title": "Chair without price", "product_id": "3333",
             "product_link": "https://www.google.com/shopping/product/3333", "source": "Etsy"},
            # Unusable: no title.
            {"position": 4, "product_id": "4444", "extracted_price": 10.0, "price": "$10.00",
             "product_link": "https://www.google.com/shopping/product/4444"},
            # Unusable: no product link.
            {"position": 5, "title": "Linkless chair", "product_id": "5555",
             "extracted_price": 20.0, "price": "$20.00"},
            # Duplicate of the first result.
            {"position": 6, "title": "Mid-Century Upholstered Accent Chair, Beige", "product_id": "1111",
             "product_link": "https://www.google.com/shopping/product/1111", "extracted_price": 149.99,
             "price": "$149.99", "source": "Wayfair"},
            # Same listing title from the same seller under a different product id.
            {"position": 6, "title": "mid-century upholstered accent chair,  Beige", "product_id": "6666",
             "product_link": "https://www.google.com/shopping/product/6666", "extracted_price": 139.99,
             "price": "$139.99", "source": "Wayfair"},
            {"position": 7, "title": "Linen Armchair", "product_id": "7777",
             "product_link": "https://www.google.com/shopping/product/7777", "source": "Target",
             "price": "$120.00", "extracted_price": 120.0},
        ],
    }


def provider_with(handler) -> SerpApiProductSearchProvider:
    return SerpApiProductSearchProvider(SECRET, transport=httpx.MockTransport(handler))


def json_handler(payload: Any, status_code: int = 200, seen: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status_code, json=payload)

    return handler


def run(coro):
    return asyncio.run(coro)


# ---- Query construction -------------------------------------------------

def test_query_uses_color_primary_style_material_and_subcategory() -> None:
    assert ProductQueryBuilder().build(chair_analysis()).text == "beige modern fabric accent chair"


def test_query_avoids_repeating_words_and_vague_colors() -> None:
    analysis = chair_analysis(color="multicolor", subcategory="wooden chair", materials=["wooden"])
    assert ProductQueryBuilder().build(analysis).text == "modern wooden chair"


def test_query_falls_back_to_category_then_keywords() -> None:
    builder = ProductQueryBuilder()
    assert builder.build(chair_analysis(subcategory=None, style=[], materials=[], color=None)).text == "lounge chair"
    keywords_only = chair_analysis(category=None, subcategory=None)
    assert builder.build(keywords_only).text == "beige modern lounge chair"


def test_query_is_short_enough_to_keep_recall() -> None:
    analysis = chair_analysis(color="dark walnut brown", style=["mid-century modern"], materials=["solid oak wood"])
    assert len(ProductQueryBuilder().build(analysis).text.split()) <= 7


def test_query_rejects_no_object_and_empty_analysis() -> None:
    builder = ProductQueryBuilder()
    with pytest.raises(QueryBuildError):
        builder.build(VisualProductAnalysis(objectDetected=False, confidence=0.1))
    with pytest.raises(QueryBuildError):
        builder.build(VisualProductAnalysis(objectDetected=True, confidence=0.5))


# ---- SerpApi provider ---------------------------------------------------

def test_serpapi_results_are_normalized_filtered_and_limited() -> None:
    seen: list[httpx.Request] = []
    provider = provider_with(json_handler(serpapi_payload(), seen=seen))
    query = ProductQueryBuilder().build(chair_analysis())

    products = run(provider.search(query, 5))

    assert [p.providerProductId for p in products] == ["1111", "2222", "7777"]
    first, second = products[0], products[1]
    assert first.id == "serpapi:1111"
    assert first.provider == "serpapi"
    assert first.retailer == "Wayfair"
    assert first.price == 149.99 and first.priceText == "$149.99" and first.currency == "USD"
    assert first.rating == 4.6 and first.reviewCount == 1320
    assert first.productUrl == "https://www.google.com/shopping/product/1111"
    assert first.imageUrl.startswith("https://")
    assert second.productUrl == "https://www.walmart.com/ip/2222"  # direct merchant link preferred
    assert products[2].imageUrl is None and products[2].rating is None
    for product in products:
        assert product.dimensions.status == "unavailable"
        assert product.dimensions.widthMeters is None
    params = seen[0].url.params
    assert params["engine"] == "google_shopping"
    assert params["q"] == "beige modern fabric accent chair"

    assert len(run(provider.search(query, 2))) == 2


def test_serpapi_no_results_is_an_empty_list() -> None:
    provider = provider_with(json_handler({"search_metadata": {}, "error": "Google Shopping hasn't returned any results for this query."}))
    assert run(provider.search(ProductQueryBuilder().build(chair_analysis()), 5)) == []


@pytest.mark.parametrize("payload", [["not", "an", "object"], {"shopping_results": "nope"}, {"unexpected": True}])
def test_serpapi_malformed_response(payload: Any) -> None:
    provider = provider_with(json_handler(payload))
    with pytest.raises(ProductProviderMalformedResponseError):
        run(provider.search(ProductQueryBuilder().build(chair_analysis()), 5))


def test_serpapi_http_error_and_timeout_do_not_leak_the_key() -> None:
    query = ProductQueryBuilder().build(chair_analysis())
    with pytest.raises(ProductProviderUnavailableError) as http_error:
        run(provider_with(json_handler({"error": "Invalid API key."}, status_code=401)).search(query, 5))
    assert SECRET not in str(http_error.value)

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(ProductProviderTimeoutError) as timeout_error:
        run(provider_with(timeout).search(query, 5))
    assert SECRET not in str(timeout_error.value)
    assert timeout_error.value.__cause__ is None


def test_serpapi_missing_key() -> None:
    provider = SerpApiProductSearchProvider(None)
    assert provider.configured is False
    with pytest.raises(ProductProviderNotConfiguredError):
        run(provider.search(ProductQueryBuilder().build(chair_analysis()), 5))


# ---- Service: live, cache fallback, failure -----------------------------

class SequenceProvider:
    """Returns the same outcome for every query of one request, then the next outcome.

    Milestone 5.5 runs several queries per request; each request consumes one outcome.
    """

    name = "serpapi"

    def __init__(self, *outcomes: Any) -> None:
        self._outcomes = list(outcomes)
        self._current: Any = None

    @property
    def configured(self) -> bool:
        return True

    def next_request(self) -> None:
        self._current = self._outcomes.pop(0)

    async def search(self, query, limit):
        if self._current is None:
            self.next_request()
        outcome = self._current
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def live_products():
    provider = provider_with(json_handler(serpapi_payload()))
    return run(provider.search(ProductQueryBuilder().build(chair_analysis()), 5))


def test_provider_failure_with_cache_hit_is_labeled_cached(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    products = live_products()
    service = ProductSearchService(
        SequenceProvider(products, ProductProviderUnavailableError("down")),
        ProductSearchCache(cache_path),
    )

    live = run(service.search(chair_analysis(), 5))
    assert live.resultSource == "live" and live.cachedAt is None
    assert len(live.queries) == 3 and all(q.status == "ok" for q in live.queries)

    # A fresh cache instance proves the result was persisted to disk.
    service = ProductSearchService(
        SequenceProvider(ProductProviderTimeoutError("slow")), ProductSearchCache(cache_path)
    )
    cached = run(service.search(chair_analysis(), 5))
    assert cached.resultSource == "cache"
    assert cached.cachedAt is not None
    assert cached.query == live.query
    assert [p.model_dump() for p in cached.products] == [p.model_dump() for p in live.products]
    assert cached.message


def test_provider_failure_without_cache_raises(tmp_path: Path) -> None:
    service = ProductSearchService(
        SequenceProvider(ProductProviderMalformedResponseError("bad")),
        ProductSearchCache(tmp_path / "cache.json"),
    )
    with pytest.raises(ProductSearchFailedError) as error:
        run(service.search(chair_analysis(), 5))
    assert isinstance(error.value.cause, ProductProviderMalformedResponseError)


def test_empty_live_results_are_not_cached(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    service = ProductSearchService(SequenceProvider([]), ProductSearchCache(cache_path))
    response = run(service.search(chair_analysis(), 5))
    assert response.products == [] and response.resultSource == "live" and response.message
    assert not cache_path.exists()


def test_corrupt_cache_file_is_ignored(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("{not json", encoding="utf-8")
    service = ProductSearchService(SequenceProvider(ProductProviderUnavailableError("down")), ProductSearchCache(cache_path))
    with pytest.raises(ProductSearchFailedError):
        run(service.search(chair_analysis(), 5))


# ---- HTTP endpoint ------------------------------------------------------

def post_search(service: ProductSearchService, body: dict[str, Any]):
    app.dependency_overrides[get_product_search_service] = lambda: service
    try:
        return TestClient(app).post("/api/v1/products/search", json=body)
    finally:
        app.dependency_overrides.clear()


def test_endpoint_returns_live_products(tmp_path: Path) -> None:
    service = ProductSearchService(
        provider_with(json_handler(serpapi_payload())), ProductSearchCache(tmp_path / "c.json")
    )
    response = post_search(service, {"analysis": chair_analysis().model_dump()})
    assert response.status_code == 200
    body = response.json()
    # Milestone 5.5: the model's queries run first, then the deterministic query.
    assert body["query"] == "beige modern lounge chair"
    assert [q["query"] for q in body["queries"]] == [
        "beige modern lounge chair", "rounded fabric accent chair", "beige modern fabric accent chair"]
    assert body["provider"] == "serpapi"
    assert body["resultSource"] == "live"
    assert len(body["products"]) == 3
    assert body["products"][0]["dimensions"] == {
        "widthMeters": None, "depthMeters": None, "heightMeters": None,
        "status": "unavailable", "source": None,
    }


def test_endpoint_missing_api_key_is_503(tmp_path: Path) -> None:
    service = ProductSearchService(SerpApiProductSearchProvider(None), ProductSearchCache(tmp_path / "c.json"))
    response = post_search(service, {"analysis": chair_analysis().model_dump()})
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ProductProviderTimeoutError("slow"), 504),
        (ProductProviderUnavailableError("down"), 502),
        (ProductProviderMalformedResponseError("bad"), 502),
    ],
)
def test_endpoint_provider_failure_without_cache_is_retryable(tmp_path: Path, error, expected_status) -> None:
    service = ProductSearchService(SequenceProvider(error), ProductSearchCache(tmp_path / "c.json"))
    response = post_search(service, {"analysis": chair_analysis().model_dump()})
    assert response.status_code == expected_status
    assert "retry" in response.json()["detail"].lower()


def test_endpoint_rejects_no_object_analysis(tmp_path: Path) -> None:
    service = ProductSearchService(SequenceProvider(), ProductSearchCache(tmp_path / "c.json"))
    response = post_search(service, {"analysis": {"objectDetected": False, "confidence": 0.1}})
    assert response.status_code == 422
    assert "retry" in response.json()["detail"].lower()


def test_endpoint_rejects_invalid_request() -> None:
    response = TestClient(app).post("/api/v1/products/search", json={"analysis": {"confidence": 2}})
    assert response.status_code == 422

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

from app.image_processing import prepare_lens_upload
from app.main import app, get_product_search_service
from app.models import VisualProductAnalysis
from app.product_cache import ProductSearchCache
from app.product_models import ProductCandidate
from app.product_providers import ProductProviderUnavailableError
from app.product_search_service import ProductSearchService
from app.vision_errors import VisionMalformedResponseError
from app.visual_reranker import (
    CandidateVisualScore,
    OllamaCandidateVisualReranker,
    VisualRerankResult,
    parse_visual_scores,
)
from app.visual_search import (
    SerpApiGoogleLensProvider,
    VisualProductCandidate,
    VisualSearchResult,
    canonical_product_url,
)

SECRET = "lens-test-secret"


def run(coro):
    return asyncio.run(coro)


def jpeg_bytes(size: tuple[int, int] = (800, 600)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, (80, 90, 100)).save(output, "JPEG", quality=95)
    return output.getvalue()


def analysis() -> VisualProductAnalysis:
    return VisualProductAnalysis(
        objectDetected=True,
        category="chair",
        subcategory="mesh office chair",
        color="gray",
        materials=["mesh", "plastic"],
        distinctiveFeatures=["five wheel base", "adjustable arms"],
        searchQueries=["gray mesh office chair with arms", "ergonomic office chair"],
        confidence=0.9,
    )


def product(
    pid: str,
    title: str,
    position: int,
    *,
    url: str | None = None,
    retailer: str = "Example Store",
    price: float | None = 99.0,
) -> ProductCandidate:
    return ProductCandidate(
        id=f"serpapi:{pid}",
        provider="serpapi",
        providerProductId=pid,
        position=position,
        title=title,
        price=price,
        priceText=f"${price:.2f}" if price is not None else None,
        currency="USD" if price is not None else None,
        retailer=retailer,
        imageUrl="https://images.example/item.jpg",
        productUrl=url or f"https://shop.example/items/{pid}",
    )


def lens_candidate(
    pid: str,
    title: str,
    mode: str,
    rank: int,
    *,
    url: str | None = None,
    price: float | None = None,
) -> VisualProductCandidate:
    source = {
        "products": "lens_products",
        "visual_matches": "lens_visual_match",
        "exact_matches": "lens_exact_match",
    }[mode]
    candidate = ProductCandidate(
        id=f"serpapi-lens:{pid}",
        provider="serpapi_google_lens",
        providerProductId=pid,
        position=rank,
        title=title,
        price=price,
        priceText=f"${price:.2f}" if price is not None else None,
        retailer="Example Store",
        imageUrl="https://images.example/lens.jpg",
        productUrl=url or f"https://shop.example/items/{pid}",
        retrievalSources=[source],
        visualRank=rank,
    )
    return VisualProductCandidate(candidate, mode, rank)


def test_lens_upload_then_queries_three_modes_with_image_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/image":
            assert request.method == "POST"
            assert request.headers["content-type"].startswith("multipart/form-data")
            assert SECRET not in str(request.url)
            return httpx.Response(200, json={"image_id": "temporary-image-id"})
        params = request.url.params
        assert params["engine"] == "google_lens"
        assert params["image_id"] == "temporary-image-id"
        assert params["auto_crop"] == "true"
        mode = params["type"]
        return httpx.Response(200, json={
            "search_metadata": {"status": "Success"},
            "visual_matches": [{
                "position": 1,
                "title": f"{mode} gray mesh chair",
                "link": f"https://shop.example/{mode}",
                "source": "Example Store",
                "thumbnail": f"https://images.example/{mode}.jpg",
                "price": {"value": "$129.00", "extracted_value": 129.0, "currency": "$"},
            }],
        })

    provider = SerpApiGoogleLensProvider(SECRET, transport=httpx.MockTransport(handler))
    result = run(provider.search(jpeg_bytes(), "mesh office chair", "gray mesh office chair"))

    assert len(requests) == 4
    assert {item.mode for item in result.candidates} == {"products", "visual_matches", "exact_matches"}
    assert {item.product.retrievalSources[0] for item in result.candidates} == {
        "lens_products", "lens_visual_match", "lens_exact_match"
    }
    assert result.upload_ms >= 0 and result.search_ms >= 0


def test_lens_successful_empty_tabs_are_not_reported_as_provider_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/image":
            return httpx.Response(200, json={"image_id": "temporary-image-id"})
        return httpx.Response(200, json={
            "search_metadata": {"status": "Success"},
            "error": "Google Lens hasn't returned any results for this query.",
        })

    provider = SerpApiGoogleLensProvider(SECRET, transport=httpx.MockTransport(handler))
    result = run(provider.search(jpeg_bytes(), "mesh office chair", "gray mesh office chair"))

    assert result.candidates == []
    assert result.mode_errors == {}


@pytest.mark.parametrize(
    ("mode", "expected_source"),
    [
        ("products", "lens_products"),
        ("visual_matches", "lens_visual_match"),
        ("exact_matches", "lens_exact_match"),
    ],
)
def test_lens_mode_parsing(mode: str, expected_source: str) -> None:
    provider = SerpApiGoogleLensProvider(SECRET)
    rows = [{
        "position": 2,
        "title": "Gray ergonomic mesh chair",
        "link": "https://retailer.example/chair?utm_source=lens",
        "source": "Retailer",
        "thumbnail": "https://images.example/chair.jpg",
        "price": {"value": "$88", "extracted_value": 88.0, "currency": "$"},
        "sku": "CHAIR-88",
        "in_stock": True,
    }]
    parsed = provider.normalize(rows, mode, 5)
    assert len(parsed) == 1
    candidate = parsed[0].product
    assert candidate.retrievalSources == [expected_source]
    assert candidate.visualRank == 2 and candidate.price == 88.0 and candidate.currency == "USD"
    assert candidate.inStock is True
    assert candidate.identifiers["sku"] == "CHAIR-88"


def test_lens_malformed_rows_and_missing_link_are_skipped_but_missing_image_is_retained() -> None:
    provider = SerpApiGoogleLensProvider(SECRET)
    parsed = provider.normalize([
        "bad row",
        {"title": "No link"},
        {"link": "https://shop.example/no-title"},
        {"title": "Usable without image", "link": "https://shop.example/usable", "source": "Shop"},
    ], "visual_matches", 10)
    assert len(parsed) == 1
    assert parsed[0].product.imageUrl is None
    assert parsed[0].product.price is None


def test_lens_rejected_upload_does_not_leak_key() -> None:
    provider = SerpApiGoogleLensProvider(
        SECRET,
        transport=httpx.MockTransport(lambda request: httpx.Response(400, json={"error": "bad image"})),
    )
    with pytest.raises(ProductProviderUnavailableError) as error:
        run(provider.search(jpeg_bytes(), None, None))
    assert SECRET not in str(error.value)


def test_lens_upload_copy_is_below_current_500kb_limit() -> None:
    noisy = Image.effect_noise((2400, 1800), 80).convert("RGB")
    output = BytesIO()
    noisy.save(output, "JPEG", quality=100)
    prepared = prepare_lens_upload(output.getvalue())
    assert len(prepared) <= 500_000
    with Image.open(BytesIO(prepared)) as image:
        assert image.format == "JPEG"


def test_canonical_url_removes_tracking_but_preserves_product_parameters() -> None:
    url = "https://SHOP.example/item/42/?variant=black&utm_source=lens#photo"
    assert canonical_product_url(url) == "https://shop.example/item/42?variant=black"


class TextProvider:
    name = "serpapi"

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome

    @property
    def configured(self) -> bool:
        return True

    async def search(self, query, limit):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class VisualProvider:
    name = "lens"

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.images: list[bytes] = []

    @property
    def configured(self) -> bool:
        return True

    async def search(self, image, category_hint, text_hint):
        self.images.append(image)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def search_service(tmp_path: Path, text: Any, visual: Any, reranker=None) -> ProductSearchService:
    return ProductSearchService(
        TextProvider(text),
        ProductSearchCache(tmp_path / "cache.json"),
        visual_provider=VisualProvider(visual),
        visual_reranker=reranker,
    )


def test_same_product_from_text_and_multiple_lens_modes_is_deduped(tmp_path: Path) -> None:
    shared_url = "https://shop.example/items/chair-1"
    text = product("text-1", "Gray Mesh Office Chair", 2, url=shared_url + "?utm_source=shopping")
    lens = VisualSearchResult([
        lens_candidate("lens-a", "Gray Mesh Office Chair", "products", 3, url=shared_url, price=95),
        lens_candidate("lens-b", "Gray Mesh Office Chair", "visual_matches", 1, url=shared_url),
        lens_candidate("lens-c", "Gray Mesh Office Chair", "exact_matches", 2, url=shared_url),
    ], 10, 20)
    response = run(search_service(tmp_path, [text], lens).search(analysis(), 5, jpeg_bytes()))
    assert len(response.products) == 1
    merged = response.products[0]
    assert merged.retrievalSources == [
        "text_search", "lens_products", "lens_visual_match", "lens_exact_match"
    ]
    assert merged.textRank == 2 and merged.visualRank == 1
    assert merged.price == 99.0
    assert response.retrievalMode == "text_visual"


def test_lens_failure_falls_back_to_text(tmp_path: Path) -> None:
    response = run(search_service(
        tmp_path,
        [product("1", "Gray Mesh Office Chair", 1)],
        ProductProviderUnavailableError("lens down"),
    ).search(analysis(), 5, jpeg_bytes()))
    assert len(response.products) == 1
    assert response.visualSearchStatus == "failed"
    assert response.retrievalMode == "text_only"


def test_text_failure_falls_back_to_lens(tmp_path: Path) -> None:
    lens = VisualSearchResult([
        lens_candidate("1", "Gray Mesh Office Chair", "products", 1, price=89),
    ], 5, 15)
    response = run(search_service(
        tmp_path,
        ProductProviderUnavailableError("shopping down"),
        lens,
    ).search(analysis(), 5, jpeg_bytes()))
    assert len(response.products) == 1
    assert response.retrievalMode == "visual_only"
    assert all(query.status == "failed" for query in response.queries)


def test_cross_generator_agreement_beats_unrelated_high_text_rank(tmp_path: Path) -> None:
    shared_url = "https://shop.example/items/similar"
    unrelated = product("unrelated", "Stainless Steel Water Bottle", 1)
    similar = product("similar", "Gray Mesh Office Chair Adjustable Arms", 6, url=shared_url)
    lens = VisualSearchResult([
        lens_candidate("visual", "Gray Mesh Office Chair Adjustable Arms", "visual_matches", 2, url=shared_url),
    ], 5, 10)
    response = run(search_service(tmp_path, [unrelated, similar], lens).search(analysis(), 5, jpeg_bytes()))
    assert response.products[0].title.startswith("Gray Mesh")
    assert response.products[0].combinedScore > response.products[1].combinedScore


class StubVisualReranker:
    async def rerank(self, original_image, candidates):
        target = next(candidate for candidate in candidates if "Mesh" in candidate.title)
        score = CandidateVisualScore(
            candidateId=target.id,
            sameProductProbability=0.9,
            visualSimilarity=0.95,
            categoryMatch=1.0,
            reason="same mesh back and five-wheel base",
        )
        return VisualRerankResult({target.id: score}, 17, 2)


def test_local_visual_scores_are_exposed_and_affect_combined_rank(tmp_path: Path) -> None:
    weak_text = product("mesh", "Gray Mesh Office Chair", 8)
    high_text = product("generic", "Ergonomic Office Chair", 1)
    empty_lens = VisualSearchResult([], 4, 6)
    response = run(search_service(
        tmp_path, [high_text, weak_text], empty_lens, StubVisualReranker()
    ).search(analysis(), 5, jpeg_bytes()))
    assert response.products[0].providerProductId == "mesh"
    assert response.products[0].visualSimilarityScore == 0.95
    assert response.timings.localVisualRerankMs == 17


def test_malformed_local_visual_rerank_output_is_rejected() -> None:
    with pytest.raises(VisionMalformedResponseError):
        parse_visual_scores('{"scores":[{"candidateId":"x","visualSimilarity":"very"}]}', {"x"})


def test_local_visual_reranker_clamps_only_floating_point_boundary_noise() -> None:
    content = ('{"scores":[{"candidateId":"x","sameProductProbability":0.5,'
               '"visualSimilarity":0.8,"categoryMatch":1.0000000000000002,"reason":"same category"}]}')
    assert parse_visual_scores(content, {"x"})["x"].categoryMatch == 1.0
    with pytest.raises(VisionMalformedResponseError):
        parse_visual_scores(content.replace("1.0000000000000002", "1.01"), {"x"})


def test_local_visual_reranker_drops_hallucinated_candidate_ids() -> None:
    content = ('{"scores":['
               '{"candidateId":"known","sameProductProbability":0.2,"visualSimilarity":0.8,'
               '"categoryMatch":1.0,"reason":"similar shape"},'
               '{"candidateId":"invented","sameProductProbability":1.0,"visualSimilarity":1.0,'
               '"categoryMatch":1.0,"reason":"not in shortlist"}'
               ']}')
    assert set(parse_visual_scores(content, {"known"})) == {"known"}


def test_local_visual_reranker_compares_shortlist_one_candidate_at_a_time() -> None:
    candidates = [product("first", "First chair", 1), product("second", "Second chair", 2)]
    requests: list[dict[str, Any]] = []

    def ollama(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        candidate = candidates[len(requests) - 1]
        return httpx.Response(200, json={"message": {"content": json.dumps({"scores": [{
            "candidateId": candidate.id,
            "sameProductProbability": 0.5,
            "visualSimilarity": 0.7,
            "categoryMatch": 1.0,
            "reason": "same product category and similar form",
        }]})}})

    image = jpeg_bytes((256, 256))
    reranker = OllamaCandidateVisualReranker(
        "http://ollama.test",
        "qwen3-vl:8b",
        max_candidates=2,
        num_ctx=16384,
        transport=httpx.MockTransport(ollama),
        image_transport=httpx.MockTransport(lambda request: httpx.Response(
            200,
            content=image,
            headers={"content-type": "image/jpeg"},
        )),
    )
    result = run(reranker.rerank(image, candidates))

    assert set(result.scores) == {candidate.id for candidate in candidates}
    assert result.compared == 2 and len(requests) == 2
    assert all(len(body["messages"][1]["images"]) == 2 for body in requests)
    assert all(body["options"]["num_ctx"] == 16384 for body in requests)


def test_search_endpoint_passes_same_uploaded_frame_to_visual_provider(tmp_path: Path) -> None:
    visual = VisualProvider(VisualSearchResult([
        lens_candidate("1", "Gray Mesh Office Chair", "products", 1, price=99),
    ], 3, 7))
    service = ProductSearchService(
        TextProvider([]), ProductSearchCache(tmp_path / "cache.json"), visual_provider=visual
    )
    app.dependency_overrides[get_product_search_service] = lambda: service
    try:
        response = TestClient(app).post("/api/v1/products/search", json={
            "analysis": analysis().model_dump(mode="json"),
            "imageBase64": base64.b64encode(jpeg_bytes()).decode("ascii"),
            "mimeType": "image/jpeg",
            "rotationDegrees": 0,
        })
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["retrievalMode"] == "visual_only"
    assert len(visual.images) == 1 and len(visual.images[0]) <= 500_000

from __future__ import annotations

import asyncio
import base64
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app.conversation_models import (
    AGENT_ACTION_ADAPTER,
    CheckFitAction,
    ConversationContextRequest,
    FindSimilarAction,
    FindSimilarArgs,
    MeasuredSpace,
    ProductReferenceArgs,
    RefineSearchAction,
    RequestArPreviewAction,
    SearchConstraints,
)
from app.conversation_planner import (
    DeterministicFallbackPlanner,
    PlannerMalformedActionError,
)
from app.conversation_ar import M6ArPreviewGateway, M6CachedDimensionGateway
from app.conversation_service import ConversationService
from app.conversation_sponsor import ConversationSponsorBridge
from app.conversation_store import ConversationStore
from app.conversation_tools import (
    ArPreviewGatewayResult,
    ShoppingToolExecutor,
    UnavailableArPreviewGateway,
)
from app.dimension_models import ResolvedDimensions
from app.dimension_resolver import ResolutionCache
from app.models import VisualProductAnalysis
from app.main import app, get_conversation_service
from app.integrations import build_sponsor_integration_service
from app.product_models import ProductCandidate, ProductDimensions, ProductSearchResponse


def analysis() -> VisualProductAnalysis:
    return VisualProductAnalysis(
        objectDetected=True,
        category="chair",
        subcategory="accent chair",
        color="beige",
        materials=["fabric"],
        style=["modern"],
        shape="rounded",
        searchQueries=["modern accent chair"],
        confidence=0.9,
    )


def product(
    number: int,
    title: str,
    price: float,
    *,
    width: float | None = None,
    depth: float | None = None,
) -> ProductCandidate:
    return ProductCandidate(
        id=f"serpapi:p{number}",
        provider="serpapi",
        providerProductId=f"p{number}",
        position=number,
        title=title,
        price=price,
        priceText=f"${price:.2f}",
        currency="USD",
        retailer="Walmart" if number != 3 else "Target",
        imageUrl=f"https://example.test/{number}.jpg",
        productUrl=f"https://example.test/product/{number}",
        dimensions=ProductDimensions(
            widthMeters=width,
            depthMeters=depth,
            heightMeters=None,
            source="fixture" if width and depth else None,
        ),
    )


PRODUCTS = [
    product(1, "Black Fabric Lounge Chair", 150, width=0.9, depth=0.9),
    product(2, "Black Modern Accent Chair", 80, width=0.8, depth=0.6),
    product(3, "Wooden Accent Chair", 60, width=0.7, depth=0.7),
]


class FakeSearch:
    def __init__(self) -> None:
        self.calls: list[bytes | None] = []

    async def search(
        self,
        value: VisualProductAnalysis,
        max_results: int,
        image_bytes: bytes | None = None,
    ) -> ProductSearchResponse:
        self.calls.append(image_bytes)
        return ProductSearchResponse(
            query=value.searchQueries[0],
            provider="fake",
            resultSource="live",
            products=PRODUCTS[:max_results],
        )


class FakeDimensions:
    async def resolve(self, candidate: ProductCandidate) -> ResolvedDimensions:
        return ResolvedDimensions(productId=candidate.id, message="No dimensions in fixture.")


class ReadyAr:
    async def request(self, candidate: ProductCandidate) -> ArPreviewGatewayResult:
        return ArPreviewGatewayResult(True, f"{candidate.title} is ready. Tap a surface to place it.")


class QueuePlanner:
    def __init__(self, *actions) -> None:
        self.actions = list(actions)

    async def plan(self, message, state):
        return self.actions.pop(0)


class StateCapturingPlanner(QueuePlanner):
    def __init__(self, *actions) -> None:
        super().__init__(*actions)
        self.remembered: list[list[str]] = []

    async def plan(self, message, state):
        self.remembered.append(list(state.rememberedPreferences))
        return await super().plan(message, state)


class MalformedPlanner:
    async def plan(self, message, state):
        raise PlannerMalformedActionError("bad action")


class ResponseWriter:
    def __init__(self, response: str | None) -> None:
        self.response = response
        self.outcome = None

    async def respond(self, message, action, outcome, state):
        self.outcome = outcome
        if self.response is None:
            raise PlannerMalformedActionError("bad response")
        return self.response


def make_service(planner, *, ar=None, response_writer=None) -> ConversationService:
    store = ConversationStore()
    tools = ShoppingToolExecutor(FakeSearch(), FakeDimensions(), ar or UnavailableArPreviewGateway())
    return ConversationService(store, planner, tools, response_writer=response_writer)


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("Find something like this", "find_similar_products"),
        ("Show me a cheaper one", "refine_search"),
        ("Only show black ones under $100", "refine_search"),
        ("Would the second one fit here?", "check_fit"),
        ("Show that one in my room", "request_ar_preview"),
        ("Go back to the first one", "select_product"),
        ("Which of these is smaller?", "compare_products"),
    ],
)
def test_demo_phrases_have_deterministic_typed_fallbacks(utterance: str, expected: str) -> None:
    action = asyncio.run(DeterministicFallbackPlanner().plan(utterance, ConversationStore().create()))
    assert action.action == expected
    if "second" in utterance.lower():
        assert action.arguments.resultNumber == 2


def test_find_refine_and_reference_preserve_current_result_order() -> None:
    planner = QueuePlanner(
        FindSimilarAction(action="find_similar_products", arguments=FindSimilarArgs()),
        RefineSearchAction(
            action="refine_search",
            arguments={"constraints": {"maxPrice": 100}},
        ),
        CheckFitAction(action="check_fit", arguments=ProductReferenceArgs(resultNumber=2)),
    )
    service = make_service(planner)
    created = service.create()
    service.sync_context(created.sessionId, ConversationContextRequest(analysis=analysis()))

    found = asyncio.run(service.turn(created.sessionId, "Find something like this"))
    assert [p.id for p in found.products] == ["serpapi:p1", "serpapi:p2", "serpapi:p3"]

    refined = asyncio.run(service.turn(created.sessionId, "Only under $100"))
    assert [p.id for p in refined.products] == ["serpapi:p2", "serpapi:p3"]

    # The second reference resolves against the refined order, not a stale provider index.
    fit = asyncio.run(service.turn(created.sessionId, "Would the second one fit?"))
    assert fit.selectedProduct.id == "serpapi:p3"
    assert fit.status == "needs_input"
    assert fit.uiDirective == "measure_space"


def test_find_uses_analyzed_image_once_and_reuses_current_results() -> None:
    search = FakeSearch()
    store = ConversationStore()
    planner = QueuePlanner(
        FindSimilarAction(action="find_similar_products", arguments=FindSimilarArgs()),
        FindSimilarAction(action="find_similar_products", arguments=FindSimilarArgs()),
    )
    service = ConversationService(store, planner, ShoppingToolExecutor(search, FakeDimensions()))
    session_id = service.create().sessionId
    service.sync_context(
        session_id,
        ConversationContextRequest(analysis=analysis(), imageBase64="prepared-by-route"),
        image_bytes=b"camera-jpeg",
    )

    first = asyncio.run(service.turn(session_id, "Find something like this"))
    second = asyncio.run(service.turn(session_id, "Show those matches again"))

    assert search.calls == [b"camera-jpeg"]
    assert first.searchResult is not None
    assert second.products == first.products


def test_conversation_context_route_prepares_visual_search_image() -> None:
    search = FakeSearch()
    service = ConversationService(
        ConversationStore(),
        QueuePlanner(
            FindSimilarAction(action="find_similar_products", arguments=FindSimilarArgs()),
            FindSimilarAction(action="find_similar_products", arguments=FindSimilarArgs()),
        ),
        ShoppingToolExecutor(search, FakeDimensions()),
    )
    image = BytesIO()
    Image.new("RGB", (12, 8), (30, 60, 90)).save(image, format="JPEG")
    app.dependency_overrides[get_conversation_service] = lambda: service
    try:
        client = TestClient(app)
        session_id = client.post("/api/v1/conversations").json()["sessionId"]
        context = client.put(
            f"/api/v1/conversations/{session_id}/context",
            json={
                "analysis": analysis().model_dump(mode="json"),
                "imageBase64": base64.b64encode(image.getvalue()).decode("ascii"),
                "mimeType": "image/jpeg",
                "rotationDegrees": 0,
            },
        )
        assert context.status_code == 200
        assert context.json()["state"]["analyzedImageAvailable"] is True

        first = client.post(
            f"/api/v1/conversations/{session_id}/messages",
            json={"message": "Find something like this"},
        )
        second = client.post(
            f"/api/v1/conversations/{session_id}/messages",
            json={"message": "Show those matches again"},
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert len(search.calls) == 1
        assert search.calls[0]
    finally:
        app.dependency_overrides.clear()


def test_conversation_rejects_products_not_returned_by_backend() -> None:
    store = ConversationStore()
    service = ConversationService(
        store,
        QueuePlanner(),
        ShoppingToolExecutor(FakeSearch(), FakeDimensions()),
        product_lookup=lambda _product_id: None,
    )
    session_id = service.create().sessionId

    with pytest.raises(ValueError, match="no longer known"):
        service.sync_context(session_id, ConversationContextRequest(products=[PRODUCTS[0]]))


def test_grounded_qwen_response_is_normal_path_with_safe_tool_text_fallback() -> None:
    action = FindSimilarAction(action="find_similar_products", arguments=FindSimilarArgs())
    writer = ResponseWriter("I found three matches; choose one or refine the results.")
    service = make_service(QueuePlanner(action), response_writer=writer)
    session_id = service.create().sessionId
    service.sync_context(session_id, ConversationContextRequest(analysis=analysis()))

    response = asyncio.run(service.turn(session_id, "Find something like this"))

    assert response.responseWriter == "local_qwen"
    assert response.message == "I found three matches; choose one or refine the results."
    assert writer.outcome.status == "success"

    fallback = make_service(QueuePlanner(action), response_writer=ResponseWriter(None))
    fallback_id = fallback.create().sessionId
    fallback.sync_context(fallback_id, ConversationContextRequest(analysis=analysis()))
    safe = asyncio.run(fallback.turn(fallback_id, "Find something like this"))
    assert safe.responseWriter == "deterministic_fallback"
    assert safe.message.startswith("I found 3 matches")


def test_fit_uses_width_and_depth_without_requiring_height() -> None:
    planner = QueuePlanner(
        CheckFitAction(action="check_fit", arguments=ProductReferenceArgs(resultNumber=2)),
        CheckFitAction(action="check_fit", arguments=ProductReferenceArgs()),
    )
    service = make_service(planner)
    session_id = service.create().sessionId
    service.sync_context(
        session_id,
        ConversationContextRequest(analysis=analysis(), products=PRODUCTS),
    )

    missing = asyncio.run(service.turn(session_id, "Would the second one fit?"))
    assert missing.fit.verdict == "needs_measurement"

    service.sync_context(
        session_id,
        ConversationContextRequest(measuredSpace=MeasuredSpace(widthMeters=0.7, depthMeters=0.9)),
    )
    result = asyncio.run(service.turn(session_id, "Would it fit now?"))
    assert result.fit.verdict == "fits"
    assert result.fit.rotated is True
    assert "0.80 × 0.60" in result.message


def test_ar_reports_real_gateway_status_and_never_fabricates_success() -> None:
    action = RequestArPreviewAction(action="request_ar_preview", arguments=ProductReferenceArgs(resultNumber=1))
    unavailable = make_service(QueuePlanner(action))
    session_id = unavailable.create().sessionId
    unavailable.sync_context(session_id, ConversationContextRequest(products=PRODUCTS))
    blocked = asyncio.run(unavailable.turn(session_id, "Show it in my room"))
    assert blocked.status == "unavailable"
    assert blocked.uiDirective == "none"
    assert blocked.state.latestArPreviewProductId is None

    ready = make_service(QueuePlanner(action), ar=ReadyAr())
    ready_id = ready.create().sessionId
    ready.sync_context(ready_id, ConversationContextRequest(products=PRODUCTS))
    success = asyncio.run(ready.turn(ready_id, "Show it in my room"))
    assert success.status == "success"
    assert success.uiDirective == "enter_ar_preview"
    assert success.state.latestArPreviewProductId == "serpapi:p1"


def test_m6_ar_gateway_accepts_started_job_and_preserves_unavailable_result() -> None:
    class Response:
        def __init__(self, status: str, message: str | None = None) -> None:
            self.status = status
            self.sourceProductId = PRODUCTS[0].id
            self.message = message

    async def generating(_product: ProductCandidate) -> Response:
        return Response("generating", "Generating 3D preview…")

    async def unavailable(_product: ProductCandidate) -> Response:
        return Response("unavailable", "Verified height is unavailable.")

    started = asyncio.run(M6ArPreviewGateway(generating).request(PRODUCTS[0]))
    blocked = asyncio.run(M6ArPreviewGateway(unavailable).request(PRODUCTS[0]))

    assert started.ready is True
    assert "Generating" in started.message
    assert blocked.ready is False
    assert blocked.message == "Verified height is unavailable."


def test_conversation_fit_gateway_reuses_m6_dimension_cache() -> None:
    class Resolver:
        calls = 0

        async def resolve(self, candidate: ProductCandidate) -> ResolvedDimensions:
            self.calls += 1
            return ResolvedDimensions(
                productId=candidate.id,
                widthMeters=0.8,
                depthMeters=0.6,
                heightMeters=1.0,
                sourceType="spec_table",
            )

    resolver = Resolver()
    gateway = M6CachedDimensionGateway(resolver, ResolutionCache())

    first = asyncio.run(gateway.resolve(PRODUCTS[0]))
    second = asyncio.run(gateway.resolve(PRODUCTS[0]))

    assert first is second
    assert resolver.calls == 1


def test_malformed_model_action_falls_back_and_unknown_actions_are_rejected() -> None:
    service = make_service(MalformedPlanner())
    session_id = service.create().sessionId
    response = asyncio.run(service.turn(session_id, "Find something like this"))
    assert response.planner == "deterministic_fallback"
    assert response.action.action == "find_similar_products"
    assert response.status == "needs_input"

    with pytest.raises(ValidationError):
        AGENT_ACTION_ADAPTER.validate_python({"action": "delete_everything", "arguments": {}})


def test_session_reset_clears_references_and_constraints() -> None:
    service = make_service(QueuePlanner())
    session_id = service.create().sessionId
    service.sync_context(
        session_id,
        ConversationContextRequest(
            analysis=analysis(),
            products=PRODUCTS,
            selectedProductId=PRODUCTS[0].id,
            measuredSpace=MeasuredSpace(widthMeters=1, depthMeters=1),
        ),
    )
    reset = service.reset(session_id)
    assert reset.state.latestResultIds == []
    assert reset.state.selectedProductId is None
    assert reset.state.measuredSpace is None
    assert reset.state.analyzedObject is None


def test_sponsor_bridge_recalls_hints_and_persists_typed_successes() -> None:
    async def scenario() -> None:
        shopper_id = "android-shopper-123"
        sponsor = build_sponsor_integration_service(object())
        await sponsor.remember(shopper_id, "Prefers compact modern furniture", "preference")
        planner = StateCapturingPlanner(
            RefineSearchAction(
                action="refine_search",
                arguments={"constraints": {"color": "black", "maxPrice": 100}},
            ),
            CheckFitAction(action="check_fit", arguments=ProductReferenceArgs(resultNumber=1)),
        )
        bridge = ConversationSponsorBridge(sponsor, recall_timeout_seconds=0.1)
        service = ConversationService(
            ConversationStore(),
            planner,
            ShoppingToolExecutor(FakeSearch(), FakeDimensions()),
            sponsor_bridge=bridge,
        )
        session_id = service.create().sessionId
        synced = service.sync_context(
            session_id,
            ConversationContextRequest(
                shopperId=shopper_id,
                analysis=analysis(),
                products=PRODUCTS,
                measuredSpace=MeasuredSpace(widthMeters=1, depthMeters=1),
            ),
        )
        assert synced.state.shopperId == shopper_id

        refined = await service.turn(session_id, "Only black options under $100")
        assert refined.status == "success"
        await bridge.wait_pending()
        fit = await service.turn(session_id, "Will that one fit?")
        assert fit.status == "success"
        await bridge.wait_pending()

        assert planner.remembered[0] == ["Prefers compact modern furniture"]
        assert any("explicitRefinement" in value for value in planner.remembered[1])
        saved_session = await sponsor.get_session(shopper_id)
        saved_products = await sponsor.recent_products(shopper_id, 5)
        saved_fits = await sponsor.fit_history(shopper_id, 5)
        memories = await sponsor.search_memory(shopper_id, "black", 5)
        assert saved_session is not None and saved_session.conversationId == session_id
        assert [item.productId for item in saved_products] == [PRODUCTS[1].id]
        assert saved_fits[0].result == "fits"
        assert any('"color":"black"' in item.content for item in memories.memories)

        reset = service.reset(session_id)
        assert reset.state.shopperId == shopper_id

    asyncio.run(scenario())


def test_sponsor_bridge_times_out_and_fails_open() -> None:
    class FailingSponsor:
        async def search_memory(self, session_id, query, limit):
            await asyncio.sleep(1)

        async def save_session(self, record):
            raise RuntimeError("offline")

        async def save_product(self, record):
            raise RuntimeError("offline")

        async def save_fit(self, record):
            raise RuntimeError("offline")

        async def remember(self, session_id, content, kind):
            raise RuntimeError("offline")

    async def scenario() -> None:
        action = RefineSearchAction(
            action="refine_search",
            arguments={"constraints": {"maxPrice": 200}},
        )
        planner = StateCapturingPlanner(action)
        bridge = ConversationSponsorBridge(FailingSponsor(), recall_timeout_seconds=0.001)
        service = ConversationService(
            ConversationStore(),
            planner,
            ShoppingToolExecutor(FakeSearch(), FakeDimensions()),
            sponsor_bridge=bridge,
        )
        session_id = service.create().sessionId
        service.sync_context(session_id, ConversationContextRequest(products=PRODUCTS))

        result = await service.turn(session_id, "Under $200")
        await bridge.wait_pending()

        assert result.status == "success"
        assert planner.remembered == [[]]

    asyncio.run(scenario())

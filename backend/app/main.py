from __future__ import annotations

import logging
import time
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Response, status

from .ollama_vision import OllamaVisionService, ollama_health
from .vision_errors import (
    VisionMalformedResponseError,
    VisionNotConfiguredError,
    VisionUnavailableError,
)
from .image_processing import InvalidImageError, prepare_image, prepare_search_image
from .models import AnalyzeProductRequest, VisualProductAnalysis
from .dimension_models import DimensionRequest, ResolvedDimensions
from fastapi.responses import FileResponse

from .ar_models import ArAssetResponse, ArPreviewRequest
from .ar_preview import ArAssetStore, ArPreviewService, DimensionLookupResult
from .dimension_semantic import OllamaSemanticDimensionExtractor
from .dimension_content import (
    HelperBrowserUseProvider,
    HelperCrawl4AIProvider,
    PragmaticDimensionResolver,
    SerpApiDimensionSearchProvider,
)
from .reconstruction import SidecarReconstructionProvider
from .dimension_resolver import DimensionResolver, ResolutionCache
from .page_fetcher import HttpPageFetcher
from .product_cache import ProductSearchCache
from .product_details import SerpApiImmersiveProductDetailSource
from .product_models import ProductSearchRequest, ProductSearchResponse
from .product_providers import (
    ProductProviderMalformedResponseError,
    ProductProviderNotConfiguredError,
    ProductProviderTimeoutError,
)
from .product_search_service import ProductSearchFailedError, ProductSearchService
from .query_builder import QueryBuildError, QueryPlanner
from .serpapi_provider import SerpApiProductSearchProvider
from .settings import Settings, get_settings
from .visual_reranker import OllamaCandidateVisualReranker
from .visual_search import SerpApiGoogleLensProvider
from .conversation_ar import M6ArPreviewGateway, M6CachedDimensionGateway
from .conversation_models import (
    ConversationContextRequest,
    ConversationHistoryResponse,
    ConversationTurnRequest,
    ConversationTurnResponse,
    CreateConversationResponse,
)
from .conversation_planner import OllamaConversationPlanner
from .conversation_service import ConversationService
from .conversation_store import ConversationNotFoundError, ConversationStore
from .conversation_sponsor import ConversationSponsorBridge
from .conversation_tools import ShoppingToolExecutor
from .commerce_models import MerchantVerificationRequest, TrustVerification
from .commerce_service import CommerceService
from .integrations import SponsorIntegrationService
from .sponsor_api import get_sponsor_integration_service, router as sponsor_router

logger = logging.getLogger(__name__)
_app_logger = logging.getLogger("app")
if not _app_logger.handlers:
    # Surface local-vision timing logs next to uvicorn's output without a telemetry service.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
    _app_logger.addHandler(_handler)
    _app_logger.setLevel(logging.INFO)
    _app_logger.propagate = False


def get_analysis_service(
    settings: Settings = Depends(get_settings),
) -> OllamaVisionService:
    return OllamaVisionService(
        settings.ollama_base_url,
        settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        keep_alive=settings.ollama_keep_alive,
        min_confidence=settings.vision_min_confidence,
    )


@lru_cache
def _product_cache(path: str) -> ProductSearchCache:
    return ProductSearchCache(path)


@lru_cache
def _resolution_cache() -> ResolutionCache:
    return ResolutionCache()


def get_product_cache(settings: Settings = Depends(get_settings)) -> ProductSearchCache:
    return _product_cache(settings.product_cache_path)


def get_dimension_resolver(settings: Settings = Depends(get_settings)) -> PragmaticDimensionResolver:
    page_fetcher = HttpPageFetcher(timeout_seconds=settings.retailer_fetch_timeout_seconds)
    detail_source = SerpApiImmersiveProductDetailSource(
        settings.serpapi_api_key, timeout_seconds=settings.serpapi_timeout_seconds
    )
    extractor = OllamaSemanticDimensionExtractor(
        settings.ollama_base_url,
        settings.ollama_dimension_model,
        timeout_seconds=settings.ollama_dimension_timeout_seconds,
        num_ctx=settings.ollama_dimension_num_ctx,
        keep_alive=settings.ollama_keep_alive,
    )
    # FAST: structured/provider fields only. Do not spend model time here.
    fast = DimensionResolver(
        detail_source,
        page_fetcher,
        extractor=None,
        page_max_chars=settings.dimension_page_max_chars,
        max_pages=1,
    )
    medium = []
    if settings.dimension_crawl_enabled:
        medium.append(HelperCrawl4AIProvider(
            settings.dimension_helper_url,
            timeout_seconds=settings.dimension_crawl_timeout_seconds,
        ))
    if settings.dimension_web_search_enabled:
        medium.append(SerpApiDimensionSearchProvider(
            settings.serpapi_api_key,
            page_fetcher,
            timeout_seconds=settings.serpapi_timeout_seconds,
            country=settings.serpapi_country,
            language=settings.serpapi_language,
        ))
    browser = HelperBrowserUseProvider(
        settings.dimension_helper_url,
        timeout_seconds=settings.dimension_browser_timeout_seconds,
    ) if settings.dimension_browser_use_enabled else None
    return PragmaticDimensionResolver(
        fast,
        extractor,
        medium,
        browser,
        page_max_chars=settings.dimension_page_max_chars,
    )


def get_legacy_dimension_resolver(settings: Settings = Depends(get_settings)) -> DimensionResolver:
    """Old full resolver retained for scripts/regression tests, not app wiring."""
    return DimensionResolver(
        SerpApiImmersiveProductDetailSource(
            settings.serpapi_api_key, timeout_seconds=settings.serpapi_timeout_seconds
        ),
        HttpPageFetcher(timeout_seconds=settings.retailer_fetch_timeout_seconds),
        extractor=OllamaSemanticDimensionExtractor(
            settings.ollama_base_url,
            settings.ollama_dimension_model,
            timeout_seconds=settings.ollama_dimension_timeout_seconds,
            num_ctx=settings.ollama_dimension_num_ctx,
            keep_alive=settings.ollama_keep_alive,
        ) if settings.dimension_llm_enabled else None,
        page_max_chars=settings.dimension_page_max_chars,
    )


def get_resolution_cache() -> ResolutionCache:
    return _resolution_cache()


_ar_service: ArPreviewService | None = None


def get_ar_preview_service(settings: Settings = Depends(get_settings)) -> ArPreviewService:
    # One long-lived instance: it owns the in-memory job registry and GPU semaphore.
    global _ar_service
    if _ar_service is None:
        resolver = get_dimension_resolver(settings)
        resolutions = _resolution_cache()

        async def dimensions(product):
            cache_started = time.perf_counter()
            cached = resolutions.get(product.id)
            cache_seconds = round(time.perf_counter() - cache_started, 3)
            if cached is not None:
                return DimensionLookupResult(
                    cached,
                    cache_hit=True,
                    path="check_fit_cache",
                    timings={"cacheLookupSeconds": cache_seconds},
                )
            result = await resolver.resolve(product)
            resolutions.put(result)
            trace = getattr(resolver, "last_trace", {})
            resolver_timings = trace.get("timings") if isinstance(trace, dict) else None
            timings = {"cacheLookupSeconds": cache_seconds}
            if isinstance(resolver_timings, dict):
                timings.update({
                    str(key): float(value)
                    for key, value in resolver_timings.items()
                    if isinstance(value, (int, float))
                })
            winner = trace.get("winner") if isinstance(trace, dict) else None
            return DimensionLookupResult(
                result,
                cache_hit=False,
                path=str(winner or result.extractionMethod or result.sourceType or "resolver"),
                timings=timings,
            )

        _ar_service = ArPreviewService(
            ArAssetStore(settings.ar_asset_cache_dir),
            SidecarReconstructionProvider(settings.reconstruction_service_url,
                                          timeout_seconds=settings.reconstruction_timeout_seconds),
            dimensions,
        )
    return _ar_service


@lru_cache
def get_conversation_store() -> ConversationStore:
    return ConversationStore()


@lru_cache
def get_commerce_service() -> CommerceService:
    return CommerceService()


def get_product_search_service(
    settings: Settings = Depends(get_settings),
) -> ProductSearchService:
    provider = SerpApiProductSearchProvider(
        settings.serpapi_api_key,
        timeout_seconds=settings.serpapi_timeout_seconds,
        country=settings.serpapi_country,
        language=settings.serpapi_language,
    )
    valid_lens_modes = {"products", "visual_matches", "exact_matches"}
    lens_modes = tuple(
        mode.strip() for mode in settings.lens_modes.split(",")
        if mode.strip() in valid_lens_modes
    ) or ("products", "visual_matches", "exact_matches")
    visual_provider = SerpApiGoogleLensProvider(
        settings.serpapi_api_key,
        timeout_seconds=settings.serpapi_timeout_seconds,
        country=settings.serpapi_country,
        language=settings.serpapi_language,
        modes=lens_modes,
        limit_per_mode=settings.product_search_results_per_query,
    ) if settings.lens_search_enabled else None
    visual_reranker = OllamaCandidateVisualReranker(
        settings.ollama_base_url,
        settings.visual_rerank_model,
        timeout_seconds=settings.visual_rerank_timeout_seconds,
        max_candidates=settings.visual_rerank_max_candidates,
        num_ctx=settings.visual_rerank_num_ctx,
        keep_alive=settings.ollama_keep_alive,
    ) if settings.visual_rerank_enabled else None
    return ProductSearchService(
        provider,
        _product_cache(settings.product_cache_path),
        planner=QueryPlanner(max_queries=settings.product_search_max_queries),
        results_per_query=settings.product_search_results_per_query,
        visual_provider=visual_provider,
        visual_reranker=visual_reranker,
    )


def get_conversation_service(
    settings: Settings = Depends(get_settings),
    search: ProductSearchService = Depends(get_product_search_service),
    dimensions: PragmaticDimensionResolver = Depends(get_dimension_resolver),
    resolutions: ResolutionCache = Depends(get_resolution_cache),
    ar_preview: ArPreviewService = Depends(get_ar_preview_service),
    cache: ProductSearchCache = Depends(get_product_cache),
    store: ConversationStore = Depends(get_conversation_store),
    sponsor: SponsorIntegrationService = Depends(get_sponsor_integration_service),
    commerce: CommerceService = Depends(get_commerce_service),
) -> ConversationService:
    planner = OllamaConversationPlanner(
        settings.ollama_base_url,
        settings.ollama_agent_model,
        timeout_seconds=settings.ollama_agent_timeout_seconds,
        keep_alive=settings.ollama_keep_alive,
    )
    tools = ShoppingToolExecutor(
        search,
        M6CachedDimensionGateway(dimensions, resolutions),
        ar_preview=M6ArPreviewGateway(ar_preview.request),
        product_lookup=cache.find_product,
        commerce=commerce,
    )
    return ConversationService(
        store,
        planner,
        tools,
        response_writer=planner,
        product_lookup=cache.find_product,
        sponsor_bridge=ConversationSponsorBridge(sponsor),
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Spatial Commerce Visual Analysis",
        version="0.8.0",
        docs_url="/docs",
        redoc_url=None,
    )
    app.include_router(sponsor_router)

    @app.post("/api/v1/commerce/merchant/verify", response_model=TrustVerification)
    async def verify_trusted_agent(
        request: MerchantVerificationRequest,
        commerce: CommerceService = Depends(get_commerce_service),
    ) -> TrustVerification:
        return commerce.verifier.verify(request.envelope, request.intent)

    @app.get("/health")
    async def health(settings: Settings = Depends(get_settings)) -> dict[str, object]:
        vision = await ollama_health(settings.ollama_base_url, settings.ollama_model)
        dimension = await ollama_health(settings.ollama_base_url, settings.ollama_dimension_model)
        agent = dimension if settings.ollama_agent_model == settings.ollama_dimension_model else await ollama_health(
            settings.ollama_base_url, settings.ollama_agent_model
        )
        return {
            "status": "ok",
            "vision": {
                "provider": "ollama",
                "baseUrl": settings.ollama_base_url,
                "model": settings.ollama_model,
                **vision,
            },
            "dimensionModel": {
                "model": settings.ollama_dimension_model,
                "enabled": settings.dimension_llm_enabled,
                "installed": dimension.get("modelInstalled", False),
            },
            "conversationAgent": {
                "provider": "ollama",
                "model": settings.ollama_agent_model,
                "installed": agent.get("modelInstalled", False),
            },
            "productSearchConfigured": bool(settings.serpapi_api_key),
            "visualSearch": {
                "provider": "serpapi_google_lens",
                "configured": bool(settings.serpapi_api_key) and settings.lens_search_enabled,
                "uploadMode": "image_id",
                "modes": list(lens_modes_from_settings(settings.lens_modes)),
                "localReranker": settings.visual_rerank_enabled,
                "localRerankerModel": settings.visual_rerank_model,
                "localRerankerNumCtx": settings.visual_rerank_num_ctx,
            },
            "reconstruction": await SidecarReconstructionProvider(settings.reconstruction_service_url).health(),
        }

    @app.post("/api/v1/conversations", response_model=CreateConversationResponse)
    async def create_conversation(
        service: ConversationService = Depends(get_conversation_service),
    ) -> CreateConversationResponse:
        return service.create()

    @app.get("/api/v1/conversations/{session_id}", response_model=ConversationHistoryResponse)
    async def get_conversation(
        session_id: str,
        service: ConversationService = Depends(get_conversation_service),
    ) -> ConversationHistoryResponse:
        try:
            return service.history(session_id)
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Conversation session not found.") from exc

    @app.put("/api/v1/conversations/{session_id}/context", response_model=ConversationHistoryResponse)
    async def sync_conversation_context(
        session_id: str,
        request: ConversationContextRequest,
        settings: Settings = Depends(get_settings),
        service: ConversationService = Depends(get_conversation_service),
    ) -> ConversationHistoryResponse:
        try:
            image_bytes = None
            if request.imageBase64 is not None:
                if request.analysis is None:
                    raise InvalidImageError("An analyzed object is required with the conversation image.")
                image_bytes = prepare_search_image(
                    ProductSearchRequest(
                        analysis=request.analysis,
                        imageBase64=request.imageBase64,
                        mimeType=request.mimeType,
                        rotationDegrees=request.rotationDegrees,
                    ),
                    settings.max_image_bytes,
                )
            return service.sync_context(session_id, request, image_bytes=image_bytes)
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Conversation session not found.") from exc
        except InvalidImageError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/api/v1/conversations/{session_id}/messages",
        response_model=ConversationTurnResponse,
    )
    async def conversation_turn(
        session_id: str,
        request: ConversationTurnRequest,
        service: ConversationService = Depends(get_conversation_service),
    ) -> ConversationTurnResponse:
        try:
            return await service.turn(session_id, request.message)
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Conversation session not found.") from exc

    @app.post("/api/v1/conversations/{session_id}/reset", response_model=CreateConversationResponse)
    async def reset_conversation(
        session_id: str,
        service: ConversationService = Depends(get_conversation_service),
    ) -> CreateConversationResponse:
        try:
            return service.reset(session_id)
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Conversation session not found.") from exc

    @app.post("/api/v1/analyze", response_model=VisualProductAnalysis)
    async def analyze_product(
        request: AnalyzeProductRequest,
        response: Response,
        settings: Settings = Depends(get_settings),
        service: OllamaVisionService = Depends(get_analysis_service),
    ) -> VisualProductAnalysis:
        started = time.monotonic()
        try:
            image_bytes = prepare_image(request, settings.max_image_bytes)
            result = await service.analyze(image_bytes, request.userRequest)
            logger.info("Visual analysis result=%s", result.model_dump_json())
            response.headers["X-Vision-Model"] = settings.ollama_model
            response.headers["X-Analysis-Duration-Ms"] = str(int((time.monotonic() - started) * 1000))
            return result
        except InvalidImageError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except VisionNotConfiguredError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Local vision is not configured on the backend.",
            ) from exc
        except TimeoutError as exc:  # includes VisionTimeoutError and asyncio timeouts
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Local vision analysis timed out. Please retry.",
            ) from exc
        except VisionMalformedResponseError as exc:
            logger.warning("Vision model returned unusable output: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"{exc} Please retry.",
            ) from exc
        except VisionUnavailableError as exc:
            logger.warning("Vision request failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"{exc} Please retry.",
            ) from exc

    @app.post("/api/v1/products/search", response_model=ProductSearchResponse)
    async def search_products(
        request: ProductSearchRequest,
        settings: Settings = Depends(get_settings),
        service: ProductSearchService = Depends(get_product_search_service),
    ) -> ProductSearchResponse:
        try:
            image_bytes = prepare_search_image(request, settings.max_image_bytes)
            return await service.search(request.analysis, request.maxResults, image_bytes)
        except InvalidImageError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except QueryBuildError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"{exc} Analyze a clearly visible product and retry.",
            ) from exc
        except ProductProviderNotConfiguredError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Product search is not configured on the backend.",
            ) from exc
        except ProductSearchFailedError as exc:
            if isinstance(exc.cause, ProductProviderTimeoutError):
                code, detail = status.HTTP_504_GATEWAY_TIMEOUT, "Product search timed out. Please retry."
            elif isinstance(exc.cause, ProductProviderMalformedResponseError):
                code, detail = status.HTTP_502_BAD_GATEWAY, "Product search returned an invalid response. Please retry."
            else:
                code, detail = status.HTTP_502_BAD_GATEWAY, "Product search is temporarily unavailable. Please retry."
            logger.warning("Product search failed without cache: %s", type(exc.cause).__name__)
            raise HTTPException(status_code=code, detail=detail) from exc

    @app.post("/api/v1/products/dimensions", response_model=ResolvedDimensions)
    async def resolve_product_dimensions(
        request: DimensionRequest,
        cache: ProductSearchCache = Depends(get_product_cache),
        resolver: DimensionResolver = Depends(get_dimension_resolver),
        resolutions: ResolutionCache = Depends(get_resolution_cache),
    ) -> ResolvedDimensions:
        # Only products this backend itself returned from the provider are resolved;
        # client-supplied URLs or dimensions are never trusted.
        product = cache.find_product(request.productId)
        if product is None or product.productUrl != request.productUrl:
            raise HTTPException(
                status_code=404,
                detail="This product is no longer known to the backend. Search again and reselect it.",
            )
        cached = resolutions.get(product.id)
        if cached is not None:
            return cached
        result = await resolver.resolve(product)
        resolutions.put(result)
        return result

    @app.post("/api/v1/products/ar-preview", response_model=ArAssetResponse)
    async def request_ar_preview(
        request: ArPreviewRequest,
        cache: ProductSearchCache = Depends(get_product_cache),
        service: ArPreviewService = Depends(get_ar_preview_service),
    ) -> ArAssetResponse:
        # Same trust rule as dimensions: only products this backend returned; dimensions
        # are resolved server-side (M5), never taken from the client.
        product = cache.find_product(request.productId)
        if product is None or product.productUrl != request.productUrl:
            raise HTTPException(
                status_code=404,
                detail="This product is no longer known to the backend. Search again and reselect it.",
            )
        return await service.request(product)

    @app.get("/api/v1/ar-assets/{asset_id}", response_model=ArAssetResponse)
    async def ar_asset_status(asset_id: str, service: ArPreviewService = Depends(get_ar_preview_service)) -> ArAssetResponse:
        response = service.status(asset_id)
        if response is None:
            raise HTTPException(status_code=404, detail="Unknown AR asset. Request the preview again.")
        return response

    @app.get("/api/v1/ar-assets/{asset_id}/model.glb")
    async def ar_asset_model(asset_id: str, service: ArPreviewService = Depends(get_ar_preview_service)) -> FileResponse:
        path = service.glb_path(asset_id)
        if path is None:
            raise HTTPException(status_code=404, detail="AR asset not found.")
        return FileResponse(path, media_type="model/gltf-binary", filename="model.glb")

    return app


def lens_modes_from_settings(value: str) -> tuple[str, ...]:
    valid = {"products", "visual_matches", "exact_matches"}
    modes = tuple(mode.strip() for mode in value.split(",") if mode.strip() in valid)
    return modes or ("products", "visual_matches", "exact_matches")


app = create_app()

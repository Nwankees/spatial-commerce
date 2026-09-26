from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, status

from .gemini_service import (
    GeminiAnalysisService,
    GeminiMalformedResponseError,
    GeminiNotConfiguredError,
    GeminiUpstreamError,
)
from .image_processing import InvalidImageError, prepare_image
from .models import AnalyzeProductRequest, VisualProductAnalysis
from .product_cache import ProductSearchCache
from .product_models import ProductSearchRequest, ProductSearchResponse
from .product_providers import (
    ProductProviderMalformedResponseError,
    ProductProviderNotConfiguredError,
    ProductProviderTimeoutError,
)
from .product_search_service import ProductSearchFailedError, ProductSearchService
from .query_builder import QueryBuildError
from .serpapi_provider import SerpApiProductSearchProvider
from .settings import Settings, get_settings

logger = logging.getLogger(__name__)


def get_analysis_service(
    settings: Settings = Depends(get_settings),
) -> GeminiAnalysisService:
    return GeminiAnalysisService(settings)


@lru_cache
def _product_cache(path: str) -> ProductSearchCache:
    return ProductSearchCache(path)


def get_product_search_service(
    settings: Settings = Depends(get_settings),
) -> ProductSearchService:
    provider = SerpApiProductSearchProvider(
        settings.serpapi_api_key,
        timeout_seconds=settings.serpapi_timeout_seconds,
        country=settings.serpapi_country,
        language=settings.serpapi_language,
    )
    return ProductSearchService(provider, _product_cache(settings.product_cache_path))


def create_app() -> FastAPI:
    app = FastAPI(
        title="Spatial Commerce Visual Analysis",
        version="0.4.0",
        docs_url="/docs",
        redoc_url=None,
    )

    @app.get("/health")
    async def health(settings: Settings = Depends(get_settings)) -> dict[str, object]:
        return {
            "status": "ok",
            "geminiConfigured": bool(settings.gemini_api_key),
            "model": settings.gemini_model,
            "productSearchConfigured": bool(settings.serpapi_api_key),
        }

    @app.post("/api/v1/analyze", response_model=VisualProductAnalysis)
    async def analyze_product(
        request: AnalyzeProductRequest,
        settings: Settings = Depends(get_settings),
        service: GeminiAnalysisService = Depends(get_analysis_service),
    ) -> VisualProductAnalysis:
        try:
            image_bytes = prepare_image(request, settings.max_image_bytes)
            return await service.analyze(image_bytes, request.userRequest)
        except InvalidImageError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except GeminiNotConfiguredError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Gemini is not configured on the backend.",
            ) from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Gemini analysis timed out. Please retry.",
            ) from exc
        except GeminiMalformedResponseError as exc:
            logger.warning("Gemini returned malformed structured output", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned an invalid structured response. Please retry.",
            ) from exc
        except GeminiUpstreamError as exc:
            logger.warning("Gemini request failed", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini analysis failed. Please retry.",
            ) from exc

    @app.post("/api/v1/products/search", response_model=ProductSearchResponse)
    async def search_products(
        request: ProductSearchRequest,
        service: ProductSearchService = Depends(get_product_search_service),
    ) -> ProductSearchResponse:
        try:
            return await service.search(request.analysis, request.maxResults)
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

    return app


app = create_app()

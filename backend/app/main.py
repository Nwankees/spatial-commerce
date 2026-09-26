from __future__ import annotations

import asyncio
import logging

from fastapi import Depends, FastAPI, HTTPException, status

from .gemini_service import (
    GeminiAnalysisService,
    GeminiMalformedResponseError,
    GeminiNotConfiguredError,
    GeminiUpstreamError,
)
from .image_processing import InvalidImageError, prepare_image
from .models import AnalyzeProductRequest, VisualProductAnalysis
from .settings import Settings, get_settings

logger = logging.getLogger(__name__)


def get_analysis_service(
    settings: Settings = Depends(get_settings),
) -> GeminiAnalysisService:
    return GeminiAnalysisService(settings)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Spatial Commerce Visual Analysis",
        version="0.3.0",
        docs_url="/docs",
        redoc_url=None,
    )

    @app.get("/health")
    async def health(settings: Settings = Depends(get_settings)) -> dict[str, object]:
        return {
            "status": "ok",
            "geminiConfigured": bool(settings.gemini_api_key),
            "model": settings.gemini_model,
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

    return app


app = create_app()

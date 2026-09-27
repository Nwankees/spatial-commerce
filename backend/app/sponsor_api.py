from __future__ import annotations

import hmac
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from .integrations import SponsorIntegrationService, build_sponsor_integration_service
from .integrations.models import (
    FitHistoryRecord,
    IntegrationStatus,
    IntegrationWriteResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryWriteRequest,
    SavedProductRecord,
    UserSessionRecord,
    VoiceRequest,
    VoiceResponse,
)
from .settings import Settings, get_settings

router = APIRouter(prefix="/api/v1/sponsor", tags=["sponsor integrations"])


def require_sponsor_access(
    supplied_token: Annotated[
        str | None,
        Header(alias="X-Will-It-Fit-Token"),
    ] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    """Protect hosted demo operations without burdening the local demo."""

    expected_token = settings.sponsor_demo_token
    if not expected_token:
        return
    if supplied_token is None or not hmac.compare_digest(
        supplied_token.encode("utf-8"),
        expected_token.encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="Missing or invalid sponsor demo token.")


@lru_cache
def _service_for_configuration(
    mongodb_uri: str | None,
    mongodb_database: str,
    backboard_api_key: str | None,
    backboard_assistant_id: str | None,
    backboard_base_url: str,
    elevenlabs_api_key: str | None,
    elevenlabs_voice_id: str | None,
    elevenlabs_model_id: str,
    elevenlabs_output_format: str,
    elevenlabs_timeout_seconds: float,
    sponsor_timeout_seconds: float,
    public_base_url: str | None,
) -> SponsorIntegrationService:
    class Configuration:
        pass

    config = Configuration()
    config.mongodb_uri = mongodb_uri
    config.mongodb_database = mongodb_database
    config.backboard_api_key = backboard_api_key
    config.backboard_assistant_id = backboard_assistant_id
    config.backboard_base_url = backboard_base_url
    config.elevenlabs_api_key = elevenlabs_api_key
    config.elevenlabs_voice_id = elevenlabs_voice_id
    config.elevenlabs_model_id = elevenlabs_model_id
    config.elevenlabs_output_format = elevenlabs_output_format
    config.elevenlabs_timeout_seconds = elevenlabs_timeout_seconds
    config.sponsor_timeout_seconds = sponsor_timeout_seconds
    config.public_base_url = public_base_url
    return build_sponsor_integration_service(config)


def get_sponsor_integration_service(
    settings: Settings = Depends(get_settings),
) -> SponsorIntegrationService:
    return _service_for_configuration(
        settings.mongodb_uri,
        settings.mongodb_database,
        settings.backboard_api_key,
        settings.backboard_assistant_id,
        settings.backboard_base_url,
        settings.elevenlabs_api_key,
        settings.elevenlabs_voice_id,
        settings.elevenlabs_model_id,
        settings.elevenlabs_output_format,
        settings.elevenlabs_timeout_seconds,
        settings.sponsor_timeout_seconds,
        settings.public_base_url,
    )


@router.get("/status", response_model=IntegrationStatus)
async def integration_status(
    service: SponsorIntegrationService = Depends(get_sponsor_integration_service),
) -> IntegrationStatus:
    return service.status()


@router.post(
    "/sessions",
    response_model=IntegrationWriteResponse,
    dependencies=[Depends(require_sponsor_access)],
)
async def save_session(
    record: UserSessionRecord,
    service: SponsorIntegrationService = Depends(get_sponsor_integration_service),
) -> IntegrationWriteResponse:
    return await service.save_session(record)


@router.get(
    "/sessions/{session_id}",
    response_model=UserSessionRecord,
    dependencies=[Depends(require_sponsor_access)],
)
async def get_session(
    session_id: str,
    service: SponsorIntegrationService = Depends(get_sponsor_integration_service),
) -> UserSessionRecord:
    record = await service.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return record


@router.post(
    "/products",
    response_model=IntegrationWriteResponse,
    dependencies=[Depends(require_sponsor_access)],
)
async def save_product(
    record: SavedProductRecord,
    service: SponsorIntegrationService = Depends(get_sponsor_integration_service),
) -> IntegrationWriteResponse:
    return await service.save_product(record)


@router.get(
    "/sessions/{session_id}/products",
    response_model=list[SavedProductRecord],
    dependencies=[Depends(require_sponsor_access)],
)
async def recent_products(
    session_id: str,
    limit: int = Query(default=10, ge=1, le=25),
    service: SponsorIntegrationService = Depends(get_sponsor_integration_service),
) -> list[SavedProductRecord]:
    return await service.recent_products(session_id, limit)


@router.post(
    "/fit-history",
    response_model=IntegrationWriteResponse,
    dependencies=[Depends(require_sponsor_access)],
)
async def save_fit(
    record: FitHistoryRecord,
    service: SponsorIntegrationService = Depends(get_sponsor_integration_service),
) -> IntegrationWriteResponse:
    return await service.save_fit(record)


@router.get(
    "/sessions/{session_id}/fit-history",
    response_model=list[FitHistoryRecord],
    dependencies=[Depends(require_sponsor_access)],
)
async def fit_history(
    session_id: str,
    limit: int = Query(default=10, ge=1, le=25),
    service: SponsorIntegrationService = Depends(get_sponsor_integration_service),
) -> list[FitHistoryRecord]:
    return await service.fit_history(session_id, limit)


@router.post(
    "/memory",
    response_model=IntegrationWriteResponse,
    dependencies=[Depends(require_sponsor_access)],
)
async def remember(
    request: MemoryWriteRequest,
    service: SponsorIntegrationService = Depends(get_sponsor_integration_service),
) -> IntegrationWriteResponse:
    return await service.remember(request.sessionId, request.content, request.kind)


@router.post(
    "/memory/search",
    response_model=MemorySearchResponse,
    dependencies=[Depends(require_sponsor_access)],
)
async def search_memory(
    request: MemorySearchRequest,
    service: SponsorIntegrationService = Depends(get_sponsor_integration_service),
) -> MemorySearchResponse:
    return await service.search_memory(request.sessionId, request.query, request.limit)


@router.post(
    "/voice",
    response_model=VoiceResponse,
    dependencies=[Depends(require_sponsor_access)],
)
async def synthesize_voice(
    request: VoiceRequest,
    service: SponsorIntegrationService = Depends(get_sponsor_integration_service),
) -> VoiceResponse:
    return await service.synthesize(request.text, request.enabled)

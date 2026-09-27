from __future__ import annotations

import base64

from .memory import BackboardMemoryProvider, FailOpenMemoryProvider
from .models import (
    FitHistoryRecord,
    IntegrationStatus,
    IntegrationWriteResponse,
    MemoryItem,
    MemorySearchResponse,
    PurchaseIntentRecord,
    SavedProductRecord,
    UserSessionRecord,
    VoiceResponse,
)
from .persistence import FailOpenPersistenceProvider, MongoPersistenceProvider
from .voice import ElevenLabsVoiceProvider, FailOpenVoiceProvider


class SponsorIntegrationService:
    def __init__(
        self,
        persistence: FailOpenPersistenceProvider,
        memory: FailOpenMemoryProvider,
        voice: FailOpenVoiceProvider,
        *,
        public_base_url: str | None = None,
    ) -> None:
        self.persistence = persistence
        self.memory = memory
        self.voice = voice
        self.public_base_url = public_base_url

    def status(self) -> IntegrationStatus:
        return IntegrationStatus(
            persistence={"provider": self.persistence.name, "configured": self.persistence.remote},
            memory={"provider": self.memory.name, "configured": self.memory.remote},
            voice={"provider": self.voice.name, "configured": self.voice.remote},
            publicBaseUrl=self.public_base_url,
        )

    async def save_session(self, record: UserSessionRecord) -> IntegrationWriteResponse:
        await self.persistence.save_session(record)
        return IntegrationWriteResponse(
            accepted=True,
            provider=self.persistence.name,
            persistedRemotely=self.persistence.last_write_remote,
        )

    async def get_session(self, session_id: str) -> UserSessionRecord | None:
        return await self.persistence.get_session(session_id)

    async def save_product(self, record: SavedProductRecord) -> IntegrationWriteResponse:
        await self.persistence.save_product(record)
        return IntegrationWriteResponse(
            accepted=True,
            provider=self.persistence.name,
            persistedRemotely=self.persistence.last_write_remote,
        )

    async def recent_products(self, session_id: str, limit: int) -> list[SavedProductRecord]:
        return await self.persistence.recent_products(session_id, limit)

    async def save_fit(self, record: FitHistoryRecord) -> IntegrationWriteResponse:
        await self.persistence.save_fit(record)
        return IntegrationWriteResponse(
            accepted=True,
            provider=self.persistence.name,
            persistedRemotely=self.persistence.last_write_remote,
        )

    async def fit_history(self, session_id: str, limit: int) -> list[FitHistoryRecord]:
        return await self.persistence.fit_history(session_id, limit)

    async def save_purchase(self, record: PurchaseIntentRecord) -> IntegrationWriteResponse:
        await self.persistence.save_purchase(record)
        return IntegrationWriteResponse(
            accepted=True,
            provider=self.persistence.name,
            persistedRemotely=self.persistence.last_write_remote,
        )

    async def recent_purchases(self, session_id: str, limit: int) -> list[PurchaseIntentRecord]:
        return await self.persistence.recent_purchases(session_id, limit)

    async def remember(self, session_id: str, content: str, kind: str) -> IntegrationWriteResponse:
        await self.memory.remember(session_id, content, kind)
        return IntegrationWriteResponse(
            accepted=True,
            provider=self.memory.name,
            persistedRemotely=self.memory.last_operation_remote,
        )

    async def search_memory(self, session_id: str, query: str, limit: int) -> MemorySearchResponse:
        memories: list[MemoryItem] = await self.memory.search(session_id, query, limit)
        return MemorySearchResponse(
            provider=self.memory.name,
            persistedRemotely=self.memory.last_operation_remote,
            memories=memories,
        )

    async def synthesize(self, text: str, enabled: bool = True) -> VoiceResponse:
        if not enabled:
            return VoiceResponse(
                voiceAvailable=False,
                provider="text_only",
                fallbackText=text,
                reason="Voice is turned off.",
            )
        audio = await self.voice.synthesize(text)
        if audio is None:
            return VoiceResponse(
                voiceAvailable=False,
                provider="text_only",
                fallbackText=text,
                reason=self.voice.last_error,
            )
        return VoiceResponse(
            voiceAvailable=True,
            provider=self.voice.name,
            fallbackText=text,
            audioBase64=base64.b64encode(audio.data).decode("ascii"),
            contentType=audio.content_type,
            audioByteCount=len(audio.data),
        )


def build_sponsor_integration_service(settings: object) -> SponsorIntegrationService:
    mongodb = None
    mongodb_uri = getattr(settings, "mongodb_uri", None)
    if mongodb_uri:
        try:
            mongodb = MongoPersistenceProvider(
                mongodb_uri,
                getattr(settings, "mongodb_database", "will_it_fit"),
            )
        except Exception:
            # Missing optional client or invalid setup must not stop the app.
            mongodb = None

    backboard = None
    backboard_key = getattr(settings, "backboard_api_key", None)
    backboard_assistant = getattr(settings, "backboard_assistant_id", None)
    if backboard_key and backboard_assistant:
        backboard = BackboardMemoryProvider(
            backboard_key,
            backboard_assistant,
            base_url=getattr(settings, "backboard_base_url", "https://app.backboard.io/api"),
            timeout_seconds=getattr(settings, "sponsor_timeout_seconds", 8.0),
        )

    elevenlabs = None
    elevenlabs_key = getattr(settings, "elevenlabs_api_key", None)
    elevenlabs_voice = getattr(settings, "elevenlabs_voice_id", None)
    if elevenlabs_key and elevenlabs_voice:
        elevenlabs = ElevenLabsVoiceProvider(
            elevenlabs_key,
            elevenlabs_voice,
            model_id=getattr(settings, "elevenlabs_model_id", "eleven_flash_v2_5"),
            output_format=getattr(settings, "elevenlabs_output_format", "mp3_44100_128"),
            timeout_seconds=getattr(settings, "elevenlabs_timeout_seconds", 20.0),
        )

    return SponsorIntegrationService(
        persistence=FailOpenPersistenceProvider(mongodb),
        memory=FailOpenMemoryProvider(backboard),
        voice=FailOpenVoiceProvider(elevenlabs),
        public_base_url=getattr(settings, "public_base_url", None),
    )

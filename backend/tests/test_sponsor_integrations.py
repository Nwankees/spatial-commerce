from __future__ import annotations

import asyncio

import httpx
from fastapi.testclient import TestClient

from app.integrations.contracts import SpeechAudio
from app.integrations.memory import BackboardMemoryProvider, FailOpenMemoryProvider
from app.integrations.models import FitHistoryRecord, SavedProductRecord, UserSessionRecord
from app.integrations.persistence import (
    FailOpenPersistenceProvider,
    InMemoryPersistenceProvider,
    MongoPersistenceProvider,
)
from app.integrations.service import SponsorIntegrationService
from app.integrations.voice import ElevenLabsVoiceProvider, FailOpenVoiceProvider
from app.main import app
from app.settings import Settings, get_settings
from app.sponsor_api import get_sponsor_integration_service


class FakeCursor:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def sort(self, key: str, direction: int) -> FakeCursor:
        self.records.sort(key=lambda item: str(item.get(key, "")), reverse=direction < 0)
        return self

    def limit(self, value: int) -> FakeCursor:
        self.records = self.records[:value]
        return self

    def __iter__(self):
        return iter(self.records)


class FakeCollection:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def replace_one(self, query, document, upsert=False):
        for index, item in enumerate(self.records):
            if all(item.get(key) == value for key, value in query.items()):
                self.records[index] = dict(document)
                return
        self.records.append(dict(document))

    def insert_one(self, document):
        self.records.append(dict(document))

    def find_one(self, query, projection=None):
        return next(
            (dict(item) for item in self.records if all(item.get(key) == value for key, value in query.items())),
            None,
        )

    def find(self, query, projection=None):
        return FakeCursor(
            [dict(item) for item in self.records if all(item.get(key) == value for key, value in query.items())]
        )


class FakeDatabase:
    def __init__(self) -> None:
        self.user_sessions = FakeCollection()
        self.saved_products = FakeCollection()
        self.fit_history = FakeCollection()


def test_mongodb_repository_contract_saves_and_reads_core_history() -> None:
    async def scenario() -> None:
        provider = MongoPersistenceProvider(database=FakeDatabase())
        session = UserSessionRecord(sessionId="demo", conversationId="conversation-1")
        product = SavedProductRecord(
            sessionId="demo",
            productId="chair-1",
            title="Black lounge chair",
            productUrl="https://shop.example/chair-1",
        )
        fit = FitHistoryRecord(
            sessionId="demo",
            productId="chair-1",
            measuredWidthMeters=0.9,
            measuredDepthMeters=0.8,
            result="fits",
        )

        await provider.save_session(session)
        await provider.save_product(product)
        await provider.save_fit(fit)

        assert (await provider.get_session("demo")).conversationId == "conversation-1"
        assert [item.productId for item in await provider.recent_products("demo")] == ["chair-1"]
        assert [item.result for item in await provider.fit_history("demo")] == ["fits"]

    asyncio.run(scenario())


def test_unavailable_persistence_fails_open_to_in_memory() -> None:
    class BrokenPersistence(InMemoryPersistenceProvider):
        name = "mongodb_atlas"
        remote = True

        async def save_product(self, record: SavedProductRecord) -> None:
            raise TimeoutError("Atlas unavailable")

        async def recent_products(self, session_id: str, limit: int = 10):
            raise TimeoutError("Atlas unavailable")

    async def scenario() -> None:
        provider = FailOpenPersistenceProvider(BrokenPersistence())
        record = SavedProductRecord(sessionId="demo", productId="lamp-1", title="Desk lamp")
        await provider.save_product(record)
        assert provider.last_write_remote is False
        assert [item.productId for item in await provider.recent_products("demo")] == ["lamp-1"]

    asyncio.run(scenario())


def test_backboard_uses_documented_memory_endpoints_and_scopes_sessions() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/memories/search"):
            return httpx.Response(
                200,
                json={
                    "memories": [
                        {
                            "content": "Prefers walnut furniture",
                            "metadata": {"sessionId": "demo", "kind": "preference"},
                            "score": 0.94,
                        },
                        {
                            "content": "Other user's private preference",
                            "metadata": {"sessionId": "someone-else"},
                        },
                    ]
                },
            )
        return httpx.Response(201, json={"memory_id": "memory-1"})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = BackboardMemoryProvider(
                "secret-not-logged",
                "assistant-123",
                base_url="https://app.backboard.io/api",
                client=client,
            )
            await provider.remember("demo", "Prefers walnut furniture", "preference")
            memories = await provider.search("demo", "wood style", 5)

        assert [item.content for item in memories] == ["Prefers walnut furniture"]
        assert calls[0].url.path == "/api/assistants/assistant-123/memories"
        assert calls[1].url.path == "/api/assistants/assistant-123/memories/search"
        assert calls[0].headers["x-api-key"] == "secret-not-logged"

    asyncio.run(scenario())


def test_elevenlabs_tts_and_failure_both_preserve_text_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/text-to-speech/voice-123"
        assert request.url.params["output_format"] == "mp3_44100_128"
        return httpx.Response(200, content=b"fake-mp3", headers={"content-type": "audio/mpeg"})

    class BrokenVoice:
        name = "elevenlabs"
        remote = True

        async def synthesize(self, text: str) -> SpeechAudio:
            raise TimeoutError("quota or provider failure")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = ElevenLabsVoiceProvider("secret", "voice-123", client=client)
            service = SponsorIntegrationService(
                FailOpenPersistenceProvider(None),
                FailOpenMemoryProvider(None),
                FailOpenVoiceProvider(provider),
            )
            success = await service.synthesize("This chair fits.")
            assert success.voiceAvailable is True
            assert success.audioBase64 == "ZmFrZS1tcDM="
            assert success.contentType == "audio/mpeg"
            assert success.audioByteCount == 8

        failed_service = SponsorIntegrationService(
            FailOpenPersistenceProvider(None),
            FailOpenMemoryProvider(None),
            FailOpenVoiceProvider(BrokenVoice()),
        )
        failure = await failed_service.synthesize("Still show this text.")
        assert failure.voiceAvailable is False
        assert failure.fallbackText == "Still show this text."

    asyncio.run(scenario())


def test_sponsor_api_runs_with_every_credential_absent() -> None:
    service = SponsorIntegrationService(
        FailOpenPersistenceProvider(None),
        FailOpenMemoryProvider(None),
        FailOpenVoiceProvider(None),
    )
    app.dependency_overrides[get_sponsor_integration_service] = lambda: service
    try:
        client = TestClient(app)
        status = client.get("/api/v1/sponsor/status")
        saved = client.post(
            "/api/v1/sponsor/products",
            json={"sessionId": "demo", "productId": "chair-1", "title": "Chair"},
        )
        voice = client.post(
            "/api/v1/sponsor/voice",
            json={"text": "The chair fits.", "enabled": True},
        )
        memory_write = client.post(
            "/api/v1/sponsor/memory",
            json={"sessionId": "demo", "content": "I prefer black furniture", "kind": "preference"},
        )
        memory_search = client.post(
            "/api/v1/sponsor/memory/search",
            json={"sessionId": "demo", "query": "black furniture"},
        )
    finally:
        app.dependency_overrides.clear()

    assert status.status_code == 200
    assert status.json()["persistence"]["configured"] is False
    assert saved.json() == {"accepted": True, "provider": "in_memory", "persistedRemotely": False}
    assert voice.json()["fallbackText"] == "The chair fits."
    assert voice.json()["voiceAvailable"] is False
    assert memory_write.json()["persistedRemotely"] is False
    assert memory_search.json()["memories"][0]["content"] == "I prefer black furniture"


def test_sponsor_demo_token_keeps_status_public_and_protects_operations() -> None:
    service = SponsorIntegrationService(
        FailOpenPersistenceProvider(None),
        FailOpenMemoryProvider(None),
        FailOpenVoiceProvider(None),
    )
    app.dependency_overrides[get_settings] = lambda: Settings(sponsor_demo_token="demo-secret")
    app.dependency_overrides[get_sponsor_integration_service] = lambda: service
    try:
        client = TestClient(app)
        assert client.get("/api/v1/sponsor/status").status_code == 200

        operations = (
            (
                "/api/v1/sponsor/products",
                {"sessionId": "demo", "productId": "chair-1", "title": "Chair"},
            ),
            (
                "/api/v1/sponsor/memory/search",
                {"sessionId": "demo", "query": "black furniture"},
            ),
            ("/api/v1/sponsor/voice", {"text": "The chair fits.", "enabled": True}),
        )
        for path, payload in operations:
            assert client.post(path, json=payload).status_code == 401

        authorized = client.post(
            "/api/v1/sponsor/products",
            headers={"X-Will-It-Fit-Token": "demo-secret"},
            json={"sessionId": "demo", "productId": "chair-1", "title": "Chair"},
        )
        assert authorized.status_code == 200
    finally:
        app.dependency_overrides.clear()

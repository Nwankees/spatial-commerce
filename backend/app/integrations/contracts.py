from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import FitHistoryRecord, MemoryItem, PurchaseIntentRecord, SavedProductRecord, UserSessionRecord


class PersistenceProvider(Protocol):
    name: str
    remote: bool

    async def save_session(self, record: UserSessionRecord) -> None: ...

    async def get_session(self, session_id: str) -> UserSessionRecord | None: ...

    async def save_product(self, record: SavedProductRecord) -> None: ...

    async def recent_products(self, session_id: str, limit: int = 10) -> list[SavedProductRecord]: ...

    async def save_fit(self, record: FitHistoryRecord) -> None: ...

    async def fit_history(self, session_id: str, limit: int = 10) -> list[FitHistoryRecord]: ...

    async def save_purchase(self, record: PurchaseIntentRecord) -> None: ...

    async def recent_purchases(self, session_id: str, limit: int = 10) -> list[PurchaseIntentRecord]: ...


class MemoryProvider(Protocol):
    name: str
    remote: bool

    async def remember(self, session_id: str, content: str, kind: str) -> None: ...

    async def search(self, session_id: str, query: str, limit: int = 5) -> list[MemoryItem]: ...


@dataclass(frozen=True)
class SpeechAudio:
    data: bytes
    content_type: str


class VoiceProvider(Protocol):
    name: str
    remote: bool

    async def synthesize(self, text: str) -> SpeechAudio: ...

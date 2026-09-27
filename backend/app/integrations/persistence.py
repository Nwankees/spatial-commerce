from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from .contracts import PersistenceProvider
from .models import FitHistoryRecord, PurchaseIntentRecord, SavedProductRecord, UserSessionRecord

logger = logging.getLogger("app.integrations.persistence")


class InMemoryPersistenceProvider:
    """Development fallback that preserves behavior when Atlas is unavailable."""

    name = "in_memory"
    remote = False

    def __init__(self) -> None:
        self._sessions: dict[str, UserSessionRecord] = {}
        self._products: dict[str, dict[str, SavedProductRecord]] = defaultdict(dict)
        self._fits: dict[str, list[FitHistoryRecord]] = defaultdict(list)
        self._purchases: dict[str, dict[str, PurchaseIntentRecord]] = defaultdict(dict)

    async def save_session(self, record: UserSessionRecord) -> None:
        self._sessions[record.sessionId] = record.model_copy(deep=True)

    async def get_session(self, session_id: str) -> UserSessionRecord | None:
        record = self._sessions.get(session_id)
        return record.model_copy(deep=True) if record else None

    async def save_product(self, record: SavedProductRecord) -> None:
        self._products[record.sessionId][record.productId] = record.model_copy(deep=True)

    async def recent_products(self, session_id: str, limit: int = 10) -> list[SavedProductRecord]:
        records = sorted(
            self._products.get(session_id, {}).values(),
            key=lambda item: item.savedAt,
            reverse=True,
        )
        return [item.model_copy(deep=True) for item in records[:limit]]

    async def save_fit(self, record: FitHistoryRecord) -> None:
        self._fits[record.sessionId].append(record.model_copy(deep=True))

    async def fit_history(self, session_id: str, limit: int = 10) -> list[FitHistoryRecord]:
        return [item.model_copy(deep=True) for item in reversed(self._fits.get(session_id, []))][:limit]

    async def save_purchase(self, record: PurchaseIntentRecord) -> None:
        self._purchases[record.sessionId][record.intentId] = record.model_copy(deep=True)

    async def recent_purchases(self, session_id: str, limit: int = 10) -> list[PurchaseIntentRecord]:
        values = sorted(self._purchases.get(session_id, {}).values(), key=lambda item: item.updatedAt, reverse=True)
        return [item.model_copy(deep=True) for item in values[:limit]]


class MongoPersistenceProvider:
    """MongoDB Atlas repository. Model files are deliberately never stored here."""

    name = "mongodb_atlas"
    remote = True

    def __init__(
        self,
        uri: str | None = None,
        database_name: str = "will_it_fit",
        *,
        database: Any | None = None,
    ) -> None:
        self._client: Any | None = None
        if database is not None:
            self._db = database
            return
        if not uri:
            raise ValueError("MongoDB URI is required")
        # Imported only when configured so the normal local demo remains usable
        # even when the optional Atlas dependency is not installed.
        from pymongo import MongoClient

        self._client = MongoClient(uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000)
        self._db = self._client[database_name]

    async def save_session(self, record: UserSessionRecord) -> None:
        document = record.model_dump(mode="python")
        await asyncio.to_thread(
            self._db.user_sessions.replace_one,
            {"sessionId": record.sessionId},
            document,
            upsert=True,
        )

    async def get_session(self, session_id: str) -> UserSessionRecord | None:
        document = await asyncio.to_thread(
            self._db.user_sessions.find_one,
            {"sessionId": session_id},
            {"_id": 0},
        )
        return UserSessionRecord.model_validate(document) if document else None

    async def save_product(self, record: SavedProductRecord) -> None:
        document = record.model_dump(mode="python")
        await asyncio.to_thread(
            self._db.saved_products.replace_one,
            {"sessionId": record.sessionId, "productId": record.productId},
            document,
            upsert=True,
        )

    async def recent_products(self, session_id: str, limit: int = 10) -> list[SavedProductRecord]:
        def load() -> list[dict[str, Any]]:
            cursor = (
                self._db.saved_products.find({"sessionId": session_id}, {"_id": 0})
                .sort("savedAt", -1)
                .limit(limit)
            )
            return list(cursor)

        return [SavedProductRecord.model_validate(item) for item in await asyncio.to_thread(load)]

    async def save_fit(self, record: FitHistoryRecord) -> None:
        await asyncio.to_thread(
            self._db.fit_history.insert_one,
            record.model_dump(mode="python"),
        )

    async def fit_history(self, session_id: str, limit: int = 10) -> list[FitHistoryRecord]:
        def load() -> list[dict[str, Any]]:
            cursor = (
                self._db.fit_history.find({"sessionId": session_id}, {"_id": 0})
                .sort("checkedAt", -1)
                .limit(limit)
            )
            return list(cursor)

        return [FitHistoryRecord.model_validate(item) for item in await asyncio.to_thread(load)]

    async def save_purchase(self, record: PurchaseIntentRecord) -> None:
        await asyncio.to_thread(
            self._db.purchase_intents.replace_one,
            {"sessionId": record.sessionId, "intentId": record.intentId},
            record.model_dump(mode="python"),
            upsert=True,
        )

    async def recent_purchases(self, session_id: str, limit: int = 10) -> list[PurchaseIntentRecord]:
        def load() -> list[dict[str, Any]]:
            return list(
                self._db.purchase_intents.find({"sessionId": session_id}, {"_id": 0})
                .sort("updatedAt", -1)
                .limit(limit)
            )
        return [PurchaseIntentRecord.model_validate(item) for item in await asyncio.to_thread(load)]


class FailOpenPersistenceProvider:
    """Uses Atlas when healthy and transparently retains data in memory otherwise."""

    def __init__(
        self,
        primary: PersistenceProvider | None,
        fallback: InMemoryPersistenceProvider | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback or InMemoryPersistenceProvider()
        self.name = primary.name if primary else self._fallback.name
        self.remote = bool(primary and primary.remote)
        self.last_write_remote = False

    async def _write(self, method: str, record: object) -> None:
        self.last_write_remote = False
        if self._primary is not None:
            try:
                await getattr(self._primary, method)(record)
                self.last_write_remote = True
            except Exception as exc:
                logger.warning("MongoDB operation failed open: %s", type(exc).__name__)
        await getattr(self._fallback, method)(record)

    async def _read(self, method: str, *args: object) -> object:
        if self._primary is not None:
            try:
                result = await getattr(self._primary, method)(*args)
                if result:
                    return result
            except Exception as exc:
                logger.warning("MongoDB read failed open: %s", type(exc).__name__)
        return await getattr(self._fallback, method)(*args)

    async def save_session(self, record: UserSessionRecord) -> None:
        await self._write("save_session", record)

    async def get_session(self, session_id: str) -> UserSessionRecord | None:
        result = await self._read("get_session", session_id)
        return result if isinstance(result, UserSessionRecord) else None

    async def save_product(self, record: SavedProductRecord) -> None:
        await self._write("save_product", record)

    async def recent_products(self, session_id: str, limit: int = 10) -> list[SavedProductRecord]:
        result = await self._read("recent_products", session_id, limit)
        return list(result) if isinstance(result, list) else []

    async def save_fit(self, record: FitHistoryRecord) -> None:
        await self._write("save_fit", record)

    async def fit_history(self, session_id: str, limit: int = 10) -> list[FitHistoryRecord]:
        result = await self._read("fit_history", session_id, limit)
        return list(result) if isinstance(result, list) else []

    async def save_purchase(self, record: PurchaseIntentRecord) -> None:
        await self._write("save_purchase", record)

    async def recent_purchases(self, session_id: str, limit: int = 10) -> list[PurchaseIntentRecord]:
        result = await self._read("recent_purchases", session_id, limit)
        return list(result) if isinstance(result, list) else []

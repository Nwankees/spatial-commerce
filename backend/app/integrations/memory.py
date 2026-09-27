from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import httpx

from .contracts import MemoryProvider
from .models import MemoryItem

logger = logging.getLogger("app.integrations.memory")


class InMemoryMemoryProvider:
    name = "session_memory"
    remote = False

    def __init__(self) -> None:
        self._items: dict[str, list[MemoryItem]] = defaultdict(list)

    async def remember(self, session_id: str, content: str, kind: str) -> None:
        self._items[session_id].append(MemoryItem(content=content, kind=kind))

    async def search(self, session_id: str, query: str, limit: int = 5) -> list[MemoryItem]:
        terms = {word.casefold() for word in query.split() if len(word) > 2}
        ranked: list[tuple[int, MemoryItem]] = []
        for item in self._items.get(session_id, []):
            haystack = item.content.casefold()
            ranked.append((sum(term in haystack for term in terms), item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item.model_copy(deep=True) for _, item in ranked[:limit]]


class BackboardMemoryProvider:
    """Durable shopping memory using Backboard's documented memory API."""

    name = "backboard"
    remote = True

    def __init__(
        self,
        api_key: str,
        assistant_id: str,
        *,
        base_url: str = "https://app.backboard.io/api",
        timeout_seconds: float = 8.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key or not assistant_id:
            raise ValueError("Backboard API key and assistant ID are required")
        self._assistant_id = assistant_id
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self._timeout = timeout_seconds

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if self._client is not None:
            response = await self._client.request(method, url, **kwargs)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    async def remember(self, session_id: str, content: str, kind: str) -> None:
        await self._request(
            "POST",
            f"{self._base_url}/assistants/{self._assistant_id}/memories",
            headers=self._headers,
            json={
                "content": content,
                "metadata": {
                    "sessionId": session_id,
                    "kind": kind,
                    "product": "will-it-fit",
                },
            },
        )

    async def search(self, session_id: str, query: str, limit: int = 5) -> list[MemoryItem]:
        response = await self._request(
            "POST",
            f"{self._base_url}/assistants/{self._assistant_id}/memories/search",
            headers=self._headers,
            json={"query": query, "limit": min(limit * 3, 25)},
        )
        payload = response.json()
        raw_items = payload.get("memories", []) if isinstance(payload, dict) else []
        results: list[MemoryItem] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            metadata = raw.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            # An assistant can serve multiple sessions. Never return one demo
            # session's preference to another session.
            if metadata.get("sessionId") != session_id:
                continue
            content = raw.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            score = raw.get("score")
            results.append(
                MemoryItem(
                    content=content.strip(),
                    kind=str(metadata.get("kind") or "context"),
                    score=float(score) if isinstance(score, (int, float)) else None,
                )
            )
            if len(results) >= limit:
                break
        return results


class FailOpenMemoryProvider:
    def __init__(
        self,
        primary: MemoryProvider | None,
        fallback: InMemoryMemoryProvider | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback or InMemoryMemoryProvider()
        self.name = primary.name if primary else self._fallback.name
        self.remote = bool(primary and primary.remote)
        self.last_operation_remote = False

    async def remember(self, session_id: str, content: str, kind: str) -> None:
        self.last_operation_remote = False
        if self._primary is not None:
            try:
                await self._primary.remember(session_id, content, kind)
                self.last_operation_remote = True
            except Exception as exc:
                logger.warning("Backboard write failed open: %s", type(exc).__name__)
        await self._fallback.remember(session_id, content, kind)

    async def search(self, session_id: str, query: str, limit: int = 5) -> list[MemoryItem]:
        self.last_operation_remote = False
        if self._primary is not None:
            try:
                results = await self._primary.search(session_id, query, limit)
                self.last_operation_remote = True
                if results:
                    return results
            except Exception as exc:
                logger.warning("Backboard search failed open: %s", type(exc).__name__)
        return await self._fallback.search(session_id, query, limit)

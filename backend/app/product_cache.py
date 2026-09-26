from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .product_models import ProductCandidate

logger = logging.getLogger(__name__)
_PRODUCTS = TypeAdapter(list[ProductCandidate])
_CACHE_VERSION = 1


@dataclass(frozen=True)
class CachedSearch:
    query: str
    provider: str
    products: list[ProductCandidate]
    retrieved_at: datetime


class ProductSearchCache:
    """Small persisted cache of successful live searches, keyed by normalized query.

    It only ever stores results that a provider actually returned, and it is
    best-effort: read/write failures are logged and never fail a request.
    """

    def __init__(self, path: str | Path, max_entries: int = 200) -> None:
        self._path = Path(path)
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, object]] = self._load()

    def get(self, provider: str, cache_key: str) -> CachedSearch | None:
        with self._lock:
            entry = self._entries.get(_key(provider, cache_key))
        if entry is None:
            return None
        try:
            return CachedSearch(
                query=str(entry["query"]),
                provider=provider,
                products=_PRODUCTS.validate_python(entry["products"]),
                retrieved_at=datetime.fromisoformat(str(entry["retrievedAt"])),
            )
        except (KeyError, ValueError, ValidationError):
            logger.warning("Ignoring an unreadable product cache entry.")
            return None

    def put(self, provider: str, cache_key: str, query: str, products: list[ProductCandidate]) -> None:
        if not products:
            return
        entry = {
            "query": query,
            "retrievedAt": datetime.now(timezone.utc).isoformat(),
            "products": _PRODUCTS.dump_python(products, mode="json"),
        }
        with self._lock:
            key = _key(provider, cache_key)
            self._entries.pop(key, None)
            self._entries[key] = entry
            while len(self._entries) > self._max_entries:
                self._entries.pop(next(iter(self._entries)))
            snapshot = dict(self._entries)
            self._persist(snapshot)

    def _load(self) -> dict[str, dict[str, object]]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            logger.warning("Product cache file is unreadable; starting with an empty cache.")
            return {}
        if not isinstance(raw, dict) or raw.get("version") != _CACHE_VERSION:
            return {}
        entries = raw.get("entries")
        return dict(entries) if isinstance(entries, dict) else {}

    def _persist(self, entries: dict[str, dict[str, object]]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                prefix=".product-cache-", suffix=".tmp", dir=self._path.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump({"version": _CACHE_VERSION, "entries": entries}, handle)
                os.replace(temp_name, self._path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
        except OSError:
            logger.warning("Could not persist the product cache; continuing without it.")


def _key(provider: str, cache_key: str) -> str:
    return f"{provider}|{cache_key}"

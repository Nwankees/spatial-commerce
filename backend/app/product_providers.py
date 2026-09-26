from __future__ import annotations

from typing import Protocol

from .product_models import ProductCandidate
from .query_builder import ProductQuery


class ProductProviderError(RuntimeError):
    """Base class for provider failures. Messages must never contain secrets."""


class ProductProviderNotConfiguredError(ProductProviderError):
    pass


class ProductProviderTimeoutError(ProductProviderError):
    pass


class ProductProviderUnavailableError(ProductProviderError):
    """Provider HTTP/network failure, rate limit, or rejected request."""


class ProductProviderMalformedResponseError(ProductProviderError):
    pass


class ProductSearchProvider(Protocol):
    """Replaceable product source. Returns only real, normalized provider results."""

    name: str

    @property
    def configured(self) -> bool: ...

    async def search(self, query: ProductQuery, limit: int) -> list[ProductCandidate]: ...

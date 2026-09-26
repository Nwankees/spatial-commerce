from __future__ import annotations

import logging

from .models import VisualProductAnalysis
from .product_cache import ProductSearchCache
from .product_models import ProductSearchResponse
from .product_providers import (
    ProductProviderError,
    ProductProviderNotConfiguredError,
    ProductSearchProvider,
)
from .query_builder import ProductQueryBuilder

logger = logging.getLogger(__name__)


class ProductSearchFailedError(RuntimeError):
    """Live search failed and no cached result exists. Wraps the provider error."""

    def __init__(self, query: str, cause: ProductProviderError) -> None:
        super().__init__(str(cause))
        self.query = query
        self.cause = cause


class ProductSearchService:
    def __init__(
        self,
        provider: ProductSearchProvider,
        cache: ProductSearchCache | None,
        query_builder: ProductQueryBuilder | None = None,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._query_builder = query_builder or ProductQueryBuilder()

    async def search(self, analysis: VisualProductAnalysis, max_results: int) -> ProductSearchResponse:
        # Raises QueryBuildError for analyses that cannot produce a query.
        query = self._query_builder.build(analysis)
        try:
            products = await self._provider.search(query, max_results)
        except ProductProviderNotConfiguredError:
            # A configuration problem should be visible, not masked by old cache data.
            raise
        except ProductProviderError as exc:
            cached = self._cache.get(self._provider.name, query.cache_key) if self._cache else None
            if cached is None:
                raise ProductSearchFailedError(query.text, exc) from exc
            logger.warning(
                "Live product search failed (%s); serving cached results.", type(exc).__name__
            )
            return ProductSearchResponse(
                query=query.text,
                provider=self._provider.name,
                resultSource="cache",
                cachedAt=cached.retrieved_at,
                products=cached.products[:max_results],
                message="Live search is unavailable; showing results saved from an earlier live search.",
            )

        if products and self._cache:
            self._cache.put(self._provider.name, query.cache_key, query.text, products)
        return ProductSearchResponse(
            query=query.text,
            provider=self._provider.name,
            resultSource="live",
            products=products,
            message=None if products else "No purchasable matches were found for this product.",
        )

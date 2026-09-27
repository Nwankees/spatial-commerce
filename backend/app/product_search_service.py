from __future__ import annotations

import asyncio
import logging

from .models import VisualProductAnalysis
from .product_cache import ProductSearchCache
from .product_models import ProductCandidate, ProductSearchResponse, QueryOutcome
from .product_providers import (
    ProductProviderError,
    ProductProviderNotConfiguredError,
    ProductSearchProvider,
)
from .product_reranker import ProductReranker, ScoredCandidate
from .query_builder import ProductQuery, QueryPlanner, normalize_query
from .variant_context import parse_google_ids

logger = logging.getLogger(__name__)


class ProductSearchFailedError(RuntimeError):
    """Every live search failed and no cached result exists. Wraps a provider error."""

    def __init__(self, query: str, cause: ProductProviderError) -> None:
        super().__init__(str(cause))
        self.query = query
        self.cause = cause


class ProductSearchService:
    """analysis -> capped queries -> concurrent provider searches -> merge/dedupe -> rerank."""

    def __init__(
        self,
        provider: ProductSearchProvider,
        cache: ProductSearchCache | None,
        planner: QueryPlanner | None = None,
        reranker: ProductReranker | None = None,
        results_per_query: int = 10,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._planner = planner or QueryPlanner()
        self._results_per_query = results_per_query
        self._reranker = reranker or ProductReranker(results_per_query=results_per_query)

    async def search(self, analysis: VisualProductAnalysis, max_results: int) -> ProductSearchResponse:
        queries = self._planner.plan(analysis)  # Raises QueryBuildError for unusable analyses.
        outcomes = await asyncio.gather(
            *(self._provider.search(query, self._results_per_query) for query in queries),
            return_exceptions=True,
        )

        report: list[QueryOutcome] = []
        successes: list[tuple[ProductQuery, list[ProductCandidate]]] = []
        errors: list[ProductProviderError] = []
        for query, outcome in zip(queries, outcomes):
            if isinstance(outcome, ProductProviderNotConfiguredError):
                raise outcome  # Configuration problems are visible, never masked by cache.
            if isinstance(outcome, ProductProviderError):
                errors.append(outcome)
                report.append(QueryOutcome(query=query.text, status="failed", error=str(outcome)[:200]))
            elif isinstance(outcome, BaseException):
                logger.warning("Product search query failed unexpectedly: %s", type(outcome).__name__)
                errors.append(ProductProviderError("The product search failed unexpectedly."))
                report.append(QueryOutcome(query=query.text, status="failed", error="Unexpected error."))
            else:
                successes.append((query, outcome))
                report.append(QueryOutcome(query=query.text, status="ok" if outcome else "empty", resultCount=len(outcome)))
                if outcome and self._cache:
                    self._cache.put(self._provider.name, query.cache_key, query.text, outcome)

        primary = queries[0].text
        if not successes:
            return self._cached_fallback(analysis, queries, report, errors, max_results)

        ranked = self._merge_and_rank(analysis, successes, primary)[:max_results]
        if ranked and self._cache:
            self._cache.put(self._provider.name, _plan_key(queries), primary, ranked)
        failed = sum(1 for r in report if r.status == "failed")
        message = None
        if not ranked:
            message = "No purchasable matches were found for this product."
        elif failed:
            message = f"{failed} of {len(queries)} searches failed; showing results from the others."
        return ProductSearchResponse(
            query=primary, queries=report, provider=self._provider.name,
            resultSource="live", products=ranked, message=message,
        )

    def _merge_and_rank(
        self,
        analysis: VisualProductAnalysis,
        results: list[tuple[ProductQuery, list[ProductCandidate]]],
        primary: str,
    ) -> list[ProductCandidate]:
        merged: dict[str, ScoredCandidate] = {}
        aliases: dict[str, str] = {}
        for query, products in results:
            for index, product in enumerate(products):
                position = product.position or index + 1
                keys = _identity_keys(product)
                existing_key = next((aliases[k] for k in keys if k in aliases), None)
                if existing_key is None:
                    merged[product.id] = ScoredCandidate(product, position, [query.text])
                    for key in keys:
                        aliases[key] = product.id
                    continue
                item = merged[existing_key]
                if query.text not in item.queries:
                    item.queries.append(query.text)
                if position < item.best_position:
                    # Keep the best-ranked occurrence of the product.
                    merged[existing_key] = ScoredCandidate(product, position, item.queries)
                for key in keys:
                    aliases.setdefault(key, existing_key)
        ranked = self._reranker.rerank(analysis, list(merged.values()), primary)
        return [
            item.candidate.model_copy(update={"matchedQueries": list(item.queries), "retrievalScore": item.score})
            for item in ranked
        ]

    def _cached_fallback(
        self,
        analysis: VisualProductAnalysis,
        queries: list[ProductQuery],
        report: list[QueryOutcome],
        errors: list[ProductProviderError],
        max_results: int,
    ) -> ProductSearchResponse:
        primary = queries[0].text
        if self._cache:
            cached = self._cache.get(self._provider.name, _plan_key(queries))
            per_query = [] if cached else [
                (q, entry) for q in queries if (entry := self._cache.get(self._provider.name, q.cache_key))
            ]
            if cached or per_query:
                if cached:
                    products, retrieved_at = cached.products[:max_results], cached.retrieved_at
                else:
                    products = self._merge_and_rank(analysis, [(q, e.products) for q, e in per_query], primary)[:max_results]
                    retrieved_at = min(e.retrieved_at for _, e in per_query)
                logger.warning("All live product searches failed; serving cached results.")
                return ProductSearchResponse(
                    query=primary, queries=report, provider=self._provider.name, resultSource="cache",
                    cachedAt=retrieved_at, products=products,
                    message="Live search is unavailable; showing results saved from an earlier live search.",
                )
        raise ProductSearchFailedError(primary, errors[0])


def _identity_keys(product: ProductCandidate) -> list[str]:
    """Keys that prove two results are the same concrete offer.

    Same provider product ID or same URL always means the same offer. The weaker
    same-title-and-retailer key only merges results whose Google offer identity
    (headlineOfferDocid) is equal or absent on both, so two variants/offers that
    share a title are never merged.
    """
    keys = [f"id:{product.providerProductId}", f"url:{product.productUrl}"]
    title = normalize_query(product.title)
    if title:
        offer = parse_google_ids(product.productUrl).get("headlineOfferDocid", "")
        keys.append(f"listing:{title}|{normalize_query(product.retailer or '')}|{offer}")
    return keys


def _plan_key(queries: list[ProductQuery]) -> str:
    return "plan:" + "|".join(sorted(q.cache_key for q in queries))

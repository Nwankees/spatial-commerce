from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from .image_processing import InvalidImageError, prepare_lens_upload
from .models import VisualProductAnalysis
from .product_cache import ProductSearchCache
from .product_models import ProductCandidate, ProductSearchResponse, QueryOutcome, RetrievalTimings
from .product_providers import ProductProviderError, ProductProviderNotConfiguredError, ProductSearchProvider
from .product_reranker import ProductReranker, ScoredCandidate
from .query_builder import ProductQuery, QueryPlanner, normalize_query
from .variant_context import parse_google_ids
from .visual_reranker import VisualCandidateReranker
from .visual_search import VisualSearchProvider, VisualSearchResult, canonical_product_url

logger = logging.getLogger(__name__)


class ProductSearchFailedError(RuntimeError):
    """Every live retrieval path failed and no cached result exists."""

    def __init__(self, query: str, cause: ProductProviderError) -> None:
        super().__init__(str(cause))
        self.query = query
        self.cause = cause


@dataclass(frozen=True)
class _TextRun:
    report: list[QueryOutcome]
    successes: list[tuple[ProductQuery, list[ProductCandidate]]]
    errors: list[ProductProviderError]
    elapsed_ms: int


class ProductSearchService:
    """Concurrent text + visual candidate generation, merge, and rerank."""

    def __init__(
        self,
        provider: ProductSearchProvider,
        cache: ProductSearchCache | None,
        planner: QueryPlanner | None = None,
        reranker: ProductReranker | None = None,
        results_per_query: int = 10,
        visual_provider: VisualSearchProvider | None = None,
        visual_reranker: VisualCandidateReranker | None = None,
    ) -> None:
        self._provider = provider
        self._visual_provider = visual_provider
        self._visual_reranker = visual_reranker
        self._cache = cache
        self._planner = planner or QueryPlanner()
        self._results_per_query = results_per_query
        self._reranker = reranker or ProductReranker(results_per_query=results_per_query)

    async def search(
        self,
        analysis: VisualProductAnalysis,
        max_results: int,
        image_bytes: bytes | None = None,
    ) -> ProductSearchResponse:
        total_started = time.perf_counter()
        queries = self._planner.plan(analysis)
        text, visual_outcome = await asyncio.gather(
            self._run_text(queries),
            self._run_visual(analysis, image_bytes),
        )

        visual: VisualSearchResult | None = None
        visual_error: ProductProviderError | None = None
        visual_status = "skipped"
        if isinstance(visual_outcome, ProductProviderError):
            visual_error = visual_outcome
            visual_status = "failed"
            logger.warning(
                "Visual product search failed; text results remain usable: type=%s reason=%s",
                type(visual_outcome).__name__,
                str(visual_outcome)[:300],
            )
        elif visual_outcome is not None:
            visual = visual_outcome
            visual_status = "ok" if visual.candidates else "empty"

        missing_configuration = next(
            (error for error in text.errors if isinstance(error, ProductProviderNotConfiguredError)),
            None,
        )
        if missing_configuration and not (visual and visual.candidates):
            raise missing_configuration

        primary = queries[0].text
        text_worked = bool(text.successes)
        visual_worked = visual is not None
        if not text_worked and not visual_worked:
            errors = text.errors + ([visual_error] if visual_error else [])
            if errors:
                if all(isinstance(error, ProductProviderNotConfiguredError) for error in errors):
                    raise errors[0]
                return self._cached_fallback(
                    analysis, queries, text.report, errors, max_results, image_bytes, total_started
                )

        merge_started = time.perf_counter()
        merged = self._merge(text.successes, visual)
        initial_ranked = self._reranker.rerank(analysis, list(merged.values()), primary)
        merge_ms = _elapsed_ms(merge_started)
        logger.info(
            "Merged candidates=%s",
            [{"id": item.candidate.id, "title": item.candidate.title,
              "sources": item.retrieval_sources, "score": item.score}
             for item in initial_ranked],
        )

        local_ms: int | None = None
        if image_bytes and self._visual_reranker and initial_ranked:
            try:
                local = await self._visual_reranker.rerank(
                    image_bytes,
                    [item.candidate for item in initial_ranked[:12]],
                )
                local_ms = local.duration_ms
                for item in initial_ranked:
                    score = local.scores.get(item.candidate.id)
                    if score:
                        item.same_product_probability = score.sameProductProbability
                        item.visual_similarity = score.visualSimilarity
                        item.visual_category_match = score.categoryMatch
                initial_ranked = self._reranker.rerank(analysis, initial_ranked, primary)
                logger.info(
                    "Local visual scores=%s",
                    {candidate_id: score.model_dump(mode="json")
                     for candidate_id, score in local.scores.items()},
                )
                logger.info("Local visual rerank compared=%s duration=%sms", local.compared, local.duration_ms)
            except Exception as exc:
                logger.warning(
                    "Local visual rerank failed open: type=%s reason=%s",
                    type(exc).__name__,
                    str(exc)[:300],
                )

        ranked = [self._finalize(item) for item in initial_ranked[:max_results]]
        logger.info(
            "Final products=%s",
            [{"rank": index + 1, "id": product.id, "title": product.title,
              "sources": product.retrievalSources, "score": product.combinedScore}
             for index, product in enumerate(ranked)],
        )
        request_key = _request_key(queries, image_bytes)
        if ranked and self._cache:
            self._cache.put(self._provider.name, request_key, primary, ranked)

        text_has_products = any(products for _, products in text.successes)
        visual_has_products = bool(visual and visual.candidates)
        retrieval_mode = (
            "text_visual" if text_has_products and visual_has_products
            else "visual_only" if visual_has_products
            else "text_only"
        )
        failed_text = sum(1 for outcome in text.report if outcome.status == "failed")
        message = None
        if not ranked:
            message = "No purchasable or visually similar matches were found for this product."
        elif visual_status == "failed":
            message = "Visual matching was unavailable; showing product-search matches."
        elif failed_text and visual_has_products:
            message = "Some text searches failed; visual matches are included."
        elif failed_text:
            message = f"{failed_text} of {len(queries)} searches failed; showing results from the others."

        timings = RetrievalTimings(
            lensUploadMs=visual.upload_ms if visual else None,
            lensSearchMs=visual.search_ms if visual else None,
            textSearchMs=text.elapsed_ms,
            mergeRerankMs=merge_ms,
            localVisualRerankMs=local_ms,
            totalMs=_elapsed_ms(total_started),
        )
        logger.info(
            "Product retrieval mode=%s text=%sms lens_upload=%s lens_search=%s merge=%sms local=%s total=%sms",
            retrieval_mode, text.elapsed_ms, timings.lensUploadMs, timings.lensSearchMs,
            merge_ms, local_ms, timings.totalMs,
        )
        return ProductSearchResponse(
            query=primary,
            queries=text.report,
            provider=self._provider.name,
            resultSource="live",
            products=ranked,
            message=message,
            retrievalMode=retrieval_mode,
            visualSearchStatus=visual_status,
            timings=timings,
        )

    async def _run_text(self, queries: list[ProductQuery]) -> _TextRun:
        started = time.perf_counter()
        logger.info("Text product queries=%s", [query.text for query in queries])
        outcomes = await asyncio.gather(
            *(self._provider.search(query, self._results_per_query) for query in queries),
            return_exceptions=True,
        )
        report: list[QueryOutcome] = []
        successes: list[tuple[ProductQuery, list[ProductCandidate]]] = []
        errors: list[ProductProviderError] = []
        for query, outcome in zip(queries, outcomes):
            if isinstance(outcome, ProductProviderError):
                errors.append(outcome)
                report.append(QueryOutcome(query=query.text, status="failed", error=str(outcome)[:200]))
            elif isinstance(outcome, BaseException):
                logger.warning("Product search query failed unexpectedly: %s", type(outcome).__name__)
                errors.append(ProductProviderError("The product search failed unexpectedly."))
                report.append(QueryOutcome(query=query.text, status="failed", error="Unexpected error."))
            else:
                normalized = [
                    product.model_copy(update={
                        "retrievalSources": list(dict.fromkeys([*product.retrievalSources, "text_search"])),
                        "textRank": product.position or index + 1,
                    })
                    for index, product in enumerate(outcome)
                ]
                successes.append((query, normalized))
                report.append(QueryOutcome(
                    query=query.text,
                    status="ok" if normalized else "empty",
                    resultCount=len(normalized),
                ))
                if normalized and self._cache:
                    self._cache.put(self._provider.name, query.cache_key, query.text, normalized)
        return _TextRun(report, successes, errors, _elapsed_ms(started))

    async def _run_visual(
        self,
        analysis: VisualProductAnalysis,
        image_bytes: bytes | None,
    ) -> VisualSearchResult | ProductProviderError | None:
        if not image_bytes or self._visual_provider is None:
            return None
        try:
            upload = prepare_lens_upload(image_bytes)
            text_hint = analysis.searchQueries[0] if analysis.searchQueries else analysis.subcategory
            return await self._visual_provider.search(upload, analysis.subcategory or analysis.category, text_hint)
        except InvalidImageError as exc:
            return ProductProviderError(str(exc))
        except ProductProviderError as exc:
            return exc
        except Exception as exc:
            logger.warning("Visual product search failed unexpectedly: %s", type(exc).__name__)
            return ProductProviderError("Visual product search failed unexpectedly.")

    def _merge(
        self,
        text_results: list[tuple[ProductQuery, list[ProductCandidate]]],
        visual: VisualSearchResult | None,
    ) -> dict[str, ScoredCandidate]:
        merged: dict[str, ScoredCandidate] = {}
        aliases: dict[str, str] = {}
        for query, products in text_results:
            for index, product in enumerate(products):
                self._add(
                    merged, aliases, product, product.position or index + 1,
                    query.text, list(product.retrievalSources or ["text_search"]),
                    text_rank=product.textRank or product.position or index + 1,
                )
        if visual:
            for visual_candidate in visual.candidates:
                product = visual_candidate.product
                self._add(
                    merged, aliases, product, visual_candidate.rank, None,
                    list(product.retrievalSources), visual_rank=visual_candidate.rank,
                )
        return merged

    def _add(
        self,
        merged: dict[str, ScoredCandidate],
        aliases: dict[str, str],
        product: ProductCandidate,
        position: int,
        query: str | None,
        sources: list[str],
        *,
        text_rank: int | None = None,
        visual_rank: int | None = None,
    ) -> None:
        keys = _identity_keys(product)
        existing_key = next((aliases[key] for key in keys if key in aliases), None)
        if existing_key is None:
            merged[product.id] = ScoredCandidate(
                product,
                position,
                [query] if query else [],
                retrieval_sources=list(dict.fromkeys(sources)),
                text_rank=text_rank,
                visual_rank=visual_rank,
            )
            for key in keys:
                aliases[key] = product.id
            return

        item = merged[existing_key]
        better_text_occurrence = (
            query is not None
            and text_rank is not None
            and (item.text_rank is None or text_rank < item.text_rank)
        )
        if query and query not in item.queries:
            item.queries.append(query)
        item.best_position = min(item.best_position, position)
        item.text_rank = _min_optional(item.text_rank, text_rank)
        item.visual_rank = _min_optional(item.visual_rank, visual_rank)
        item.retrieval_sources = list(dict.fromkeys([*item.retrieval_sources, *sources]))
        item.candidate = _combine_products(item.candidate, product, prefer_incoming=better_text_occurrence)
        for key in [*keys, *_identity_keys(item.candidate)]:
            aliases[key] = existing_key

    def _finalize(self, item: ScoredCandidate) -> ProductCandidate:
        return item.candidate.model_copy(update={
            "position": item.best_position,
            "matchedQueries": list(item.queries),
            "retrievalSources": list(item.retrieval_sources),
            "textRank": item.text_rank,
            "visualRank": item.visual_rank,
            "visualSimilarityScore": item.visual_similarity,
            "sameProductProbability": item.same_product_probability,
            "visualCategoryMatch": item.visual_category_match,
            "retrievalScore": item.score,
            "combinedScore": item.score,
        })

    def _cached_fallback(
        self,
        analysis: VisualProductAnalysis,
        queries: list[ProductQuery],
        report: list[QueryOutcome],
        errors: list[ProductProviderError],
        max_results: int,
        image_bytes: bytes | None,
        total_started: float,
    ) -> ProductSearchResponse:
        primary = queries[0].text
        if self._cache:
            cached = self._cache.get(self._provider.name, _request_key(queries, image_bytes))
            per_query = [] if cached else [
                (query, entry)
                for query in queries
                if (entry := self._cache.get(self._provider.name, query.cache_key))
            ]
            if cached or per_query:
                if cached:
                    products, retrieved_at = cached.products[:max_results], cached.retrieved_at
                else:
                    merged = self._merge(
                        [(query, entry.products) for query, entry in per_query],
                        None,
                    )
                    products = [self._finalize(item) for item in self._reranker.rerank(
                        analysis, list(merged.values()), primary
                    )[:max_results]]
                    retrieved_at = min(entry.retrieved_at for _, entry in per_query)
                logger.warning("All live product retrieval paths failed; serving cached results.")
                return ProductSearchResponse(
                    query=primary,
                    queries=report,
                    provider=self._provider.name,
                    resultSource="cache",
                    cachedAt=retrieved_at,
                    products=products,
                    message="Live visual and product search are unavailable; showing saved results.",
                    retrievalMode="text_only",
                    visualSearchStatus="failed" if image_bytes else "skipped",
                    timings=RetrievalTimings(totalMs=_elapsed_ms(total_started)),
                )
        cause = next((error for error in errors if not isinstance(error, ProductProviderNotConfiguredError)), errors[0])
        raise ProductSearchFailedError(primary, cause)


def _combine_products(
    current: ProductCandidate,
    incoming: ProductCandidate,
    *,
    prefer_incoming: bool = False,
) -> ProductCandidate:
    if prefer_incoming:
        # Preserve the best-ranked occurrence's offer token/identity exactly,
        # while filling fields it lacks from the earlier occurrence.
        return incoming.model_copy(update={
            "price": incoming.price if incoming.price is not None else current.price,
            "priceText": incoming.priceText or current.priceText,
            "currency": incoming.currency or current.currency,
            "retailer": incoming.retailer or current.retailer,
            "imageUrl": incoming.imageUrl or current.imageUrl,
            "rating": incoming.rating if incoming.rating is not None else current.rating,
            "reviewCount": incoming.reviewCount if incoming.reviewCount is not None else current.reviewCount,
            "inStock": incoming.inStock if incoming.inStock is not None else current.inStock,
            "detailPageToken": incoming.detailPageToken or current.detailPageToken,
            "identifiers": {**current.identifiers, **incoming.identifiers},
        })
    current_direct = _is_direct_retailer_url(current.productUrl)
    incoming_direct = _is_direct_retailer_url(incoming.productUrl)
    product_url = incoming.productUrl if incoming_direct and not current_direct else current.productUrl
    richer_price = current if current.price is not None else incoming
    return current.model_copy(update={
        "title": current.title if len(current.title) >= len(incoming.title) else incoming.title,
        "price": richer_price.price,
        "priceText": richer_price.priceText,
        "currency": richer_price.currency,
        "retailer": current.retailer or incoming.retailer,
        "imageUrl": current.imageUrl or incoming.imageUrl,
        "productUrl": product_url,
        "rating": current.rating if current.rating is not None else incoming.rating,
        "reviewCount": current.reviewCount if current.reviewCount is not None else incoming.reviewCount,
        "inStock": current.inStock if current.inStock is not None else incoming.inStock,
        "detailPageToken": current.detailPageToken or incoming.detailPageToken,
        "identifiers": {**incoming.identifiers, **current.identifiers},
    })


def _identity_keys(product: ProductCandidate) -> list[str]:
    keys = [
        f"id:{normalize_query(product.provider)}:{normalize_query(product.providerProductId)}",
        f"url:{canonical_product_url(product.productUrl)}",
    ]
    google_ids = parse_google_ids(product.productUrl)
    for name, value in google_ids.items():
        if value:
            keys.append(f"google:{name}:{value}")
    for name in ("gtin", "upc"):
        value = normalize_query(product.identifiers.get(name, ""))
        if value:
            keys.append(f"global:{name}:{value}")
    for name in ("sku", "model", "model_number", "merchant_product_id", "product_id"):
        value = normalize_query(product.identifiers.get(name, ""))
        if value and product.retailer:
            keys.append(f"merchant:{normalize_query(product.retailer)}:{name}:{value}")
    title = normalize_query(product.title)
    retailer = normalize_query(product.retailer or _host(product.productUrl))
    if title and retailer:
        offer = google_ids.get("headlineOfferDocid", "")
        keys.append(f"listing:{title}|{retailer}|{offer}")
    return list(dict.fromkeys(keys))


def _request_key(queries: list[ProductQuery], image_bytes: bytes | None) -> str:
    base = "plan:" + "|".join(sorted(query.cache_key for query in queries))
    if image_bytes:
        base += "|image:" + hashlib.sha256(image_bytes).hexdigest()[:16]
    return base


def _is_direct_retailer_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return not (host.endswith("google.com") or host.endswith("googleusercontent.com"))


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").removeprefix("www.")


def _min_optional(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))

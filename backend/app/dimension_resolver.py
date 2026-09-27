from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from .dimension_extraction import SOURCE_PRIORITY, SourceFinding, extract_from_features, extract_from_html
from .dimension_models import ResolvedDimensions
from .page_fetcher import PageFetcher, PageFetchError, is_public_http_url
from .product_details import ProductDetailSource, StoreLink
from .product_models import ProductCandidate
from .product_providers import ProductProviderError, ProductProviderNotConfiguredError

logger = logging.getLogger(__name__)
MAX_RETAILER_PAGES = 2


@dataclass(frozen=True)
class _Failure:
    message: str
    retryable: bool


@dataclass(frozen=True)
class _NamedFinding:
    finding: SourceFinding
    source_name: str | None


class DimensionResolver:
    """Resolves trustworthy dimensions for one selected product.

    Sources, highest priority first: schema.org JSON-LD on the retailer page;
    structured metadata (provider product-detail spec fields, page microdata);
    retailer spec tables; explicitly labeled dimensions in retailer page text.
    A source with all three axes wins by priority; otherwise the highest
    priority source with both width and depth; otherwise any partial result.
    Values from different sources are never mixed. Nothing is estimated.
    """

    def __init__(
        self,
        detail_source: ProductDetailSource | None,
        page_fetcher: PageFetcher,
        *,
        max_pages: int = MAX_RETAILER_PAGES,
    ) -> None:
        self._detail_source = detail_source
        self._page_fetcher = page_fetcher
        self._max_pages = max_pages

    async def resolve(self, candidate: ProductCandidate) -> ResolvedDimensions:
        try:
            return await self._resolve(candidate)
        except Exception as exc:  # Never let resolution failures escape as 500s.
            logger.warning("Dimension resolution failed unexpectedly: %s", type(exc).__name__)
            return ResolvedDimensions(
                productId=candidate.id,
                retryable=True,
                message="Dimension lookup failed unexpectedly. Please retry.",
            )

    async def _resolve(self, candidate: ProductCandidate) -> ResolvedDimensions:
        findings: list[_NamedFinding] = []
        failures: list[_Failure] = []
        stores: list[StoreLink] = []

        if _is_retailer_url(candidate.productUrl):
            stores.append(StoreLink(candidate.retailer, candidate.productUrl))

        if self._detail_source is not None and candidate.detailPageToken:
            try:
                detail = await self._detail_source.fetch(candidate)
            except ProductProviderNotConfiguredError:
                failures.append(_Failure("Product details provider is not configured.", False))
                detail = None
            except ProductProviderError as exc:
                failures.append(_Failure(str(exc), True))
                detail = None
            if detail is not None:
                for finding in extract_from_features(detail.features, candidate.productUrl):
                    findings.append(_NamedFinding(finding, detail.source_name))
                stores.extend(_order_stores(detail.stores, candidate.retailer))

        pages = _unique_pages(stores)[: self._max_pages]
        results = await asyncio.gather(
            *(self._page_fetcher.fetch(store.url) for store in pages), return_exceptions=True
        )
        for store, result in zip(pages, results):
            if isinstance(result, PageFetchError):
                failures.append(_Failure(f"{store.name or 'Retailer'}: {result}", result.retryable))
            elif isinstance(result, BaseException):
                failures.append(_Failure(f"{store.name or 'Retailer'}: page could not be read.", True))
            else:
                for finding in extract_from_html(result.html, result.url):
                    findings.append(_NamedFinding(finding, store.name))

        chosen = _choose(findings)
        if chosen is not None:
            c = chosen.finding.candidate
            return ResolvedDimensions(
                productId=candidate.id,
                widthMeters=c.width,
                depthMeters=c.depth,
                heightMeters=c.height,
                sourceType=chosen.finding.source_type,
                sourceUrl=chosen.finding.source_url,
                sourceName=chosen.source_name,
                rawDimensions=c.raw_text(),
            )

        if not pages and not candidate.detailPageToken:
            message = "No retailer page or product details are available for this product."
        elif failures and all(not f.retryable for f in failures) and not findings:
            message = "No explicit dimensions found. " + "; ".join(f.message for f in failures[:3])
        else:
            message = "No explicit product dimensions were found in retailer or provider data."
            if failures:
                message += " Some sources failed: " + "; ".join(f.message for f in failures[:3])
        return ResolvedDimensions(
            productId=candidate.id,
            retryable=any(f.retryable for f in failures),
            message=message[:400],
        )


def _choose(findings: list[_NamedFinding]) -> _NamedFinding | None:
    ranked = sorted(findings, key=lambda f: SOURCE_PRIORITY.index(f.finding.source_type))
    for predicate in (
        lambda c: c.known_axes == 3,
        lambda c: c.width is not None and c.depth is not None,
        lambda c: c.known_axes > 0,
    ):
        for named in ranked:
            if predicate(named.finding.candidate):
                return named
    return None


def _is_retailer_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return is_public_http_url(url) and not (host == "google.com" or host.endswith(".google.com"))


def _order_stores(stores: list[StoreLink], retailer: str | None) -> list[StoreLink]:
    wanted = (retailer or "").lower()

    def matches(store: StoreLink) -> bool:
        name = (store.name or "").lower()
        return bool(wanted and name and (wanted.startswith(name) or name.startswith(wanted)))

    return sorted(stores, key=lambda store: not matches(store))


def _unique_pages(stores: list[StoreLink]) -> list[StoreLink]:
    seen: set[str] = set()
    unique = []
    for store in stores:
        if store.url not in seen and _is_retailer_url(store.url):
            seen.add(store.url)
            unique.append(store)
    return unique


class ResolutionCache:
    """In-memory cache of definitive (non-retryable) results to spare provider calls."""

    def __init__(self, ttl_seconds: float = 3600.0, max_entries: int = 500) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._entries: dict[str, tuple[float, ResolvedDimensions]] = {}

    def get(self, product_id: str) -> ResolvedDimensions | None:
        entry = self._entries.get(product_id)
        if entry is None or time.monotonic() - entry[0] > self._ttl:
            self._entries.pop(product_id, None)
            return None
        return entry[1]

    def put(self, result: ResolvedDimensions) -> None:
        if result.retryable:
            return
        self._entries.pop(result.productId, None)
        self._entries[result.productId] = (time.monotonic(), result)
        while len(self._entries) > self._max:
            self._entries.pop(next(iter(self._entries)))

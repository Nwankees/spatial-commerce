"""Hackathon-grade dimension fallback providers.

The primary backend stays small and stable. Rendering/browser automation lives in
an optional local helper service, while SerpApi web search remains in this process.
All providers return source-bound content; the existing Qwen extractor and
deterministic validator decide whether the content contains trustworthy dimensions.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from .dimension_models import AxisMapping, ResolvedDimensions
from .dimension_parsing import parse_labeled_dimensions
from .dimension_resolver import DimensionExtractor, DimensionResolver
from .dimension_semantic import DimensionExtractionError, validate_semantic
from .page_fetcher import PageFetcher, PageFetchError, is_public_http_url
from .page_representation import (
    PageRepresentation,
    build_page_representation,
    build_text_representation,
)
from .product_models import ProductCandidate
from .serpapi_provider import SERPAPI_ENDPOINT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DimensionContent:
    provider: str
    source_url: str
    source_label: str
    text: str | None = None
    html: str | None = None


class DimensionContentProvider(Protocol):
    name: str

    async def collect(self, candidate: ProductCandidate) -> list[DimensionContent]: ...


class HelperCrawl4AIProvider:
    name = "crawl4ai"

    def __init__(self, helper_url: str, *, timeout_seconds: float = 75.0) -> None:
        self._url = helper_url.rstrip("/")
        self._timeout = timeout_seconds

    async def collect(self, candidate: ProductCandidate) -> list[DimensionContent]:
        if not is_public_http_url(candidate.productUrl):
            return []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._url}/crawl", json={"url": candidate.productUrl})
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            logger.info("Crawl4AI helper unavailable for %s", candidate.id)
            return []
        text = payload.get("content") if isinstance(payload, dict) else None
        final_url = payload.get("url") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            return []
        source_url = final_url if isinstance(final_url, str) and is_public_http_url(final_url) else candidate.productUrl
        return [DimensionContent(self.name, source_url, candidate.retailer or _host(source_url), text=text)]


class SerpApiDimensionSearchProvider:
    name = "web_search"

    def __init__(
        self,
        api_key: str | None,
        page_fetcher: PageFetcher,
        *,
        timeout_seconds: float = 15.0,
        country: str = "us",
        language: str = "en",
        max_queries: int = 2,
        max_sources: int = 3,
    ) -> None:
        self._key = api_key
        self._fetcher = page_fetcher
        self._timeout = timeout_seconds
        self._country = country
        self._language = language
        self._max_queries = max_queries
        self._max_sources = max_sources

    async def collect(self, candidate: ProductCandidate) -> list[DimensionContent]:
        if not self._key:
            return []
        queries = _dimension_queries(candidate)[: self._max_queries]
        responses = await asyncio.gather(*(self._search(query) for query in queries), return_exceptions=True)
        organic: list[dict[str, Any]] = []
        seen: set[str] = set()
        for response in responses:
            if not isinstance(response, list):
                continue
            for result in response:
                link = result.get("link")
                if not isinstance(link, str) or link in seen or not is_public_http_url(link):
                    continue
                seen.add(link)
                organic.append(result)
                if len(organic) >= self._max_sources:
                    break
            if len(organic) >= self._max_sources:
                break
        if not organic:
            return []

        # Search snippets are immediately useful on blocked retailer pages. Fetch
        # the same small source set in parallel to provide full page data when open.
        fetches = await asyncio.gather(
            *(self._fetcher.fetch(item["link"]) for item in organic),
            return_exceptions=True,
        )
        contents: list[DimensionContent] = []
        for item, fetched in zip(organic, fetches):
            link = item["link"]
            label = item.get("source") or item.get("displayed_link") or _host(link)
            if not isinstance(label, str):
                label = _host(link)
            if not isinstance(fetched, (BaseException, PageFetchError)):
                contents.append(DimensionContent(self.name, fetched.url, label, html=fetched.html))
            snippet = "\n".join(
                value for value in (item.get("title"), item.get("snippet")) if isinstance(value, str)
            )
            if snippet.strip():
                contents.append(DimensionContent(self.name, link, label, text=snippet))
        return contents

    async def _search(self, query: str) -> list[dict[str, Any]]:
        params = {
            "engine": "google",
            "q": query,
            "gl": self._country,
            "hl": self._language,
            "num": self._max_sources,
            "api_key": self._key,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return []
        results = payload.get("organic_results") if isinstance(payload, dict) else None
        return [item for item in results if isinstance(item, dict)] if isinstance(results, list) else []


class HelperBrowserUseProvider:
    name = "browser_use"

    def __init__(self, helper_url: str, *, timeout_seconds: float = 240.0) -> None:
        self._url = helper_url.rstrip("/")
        self._timeout = timeout_seconds

    async def collect(self, candidate: ProductCandidate) -> list[DimensionContent]:
        if not is_public_http_url(candidate.productUrl):
            return []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._url}/browse",
                    json={
                        "url": candidate.productUrl,
                        "title": candidate.title,
                        "retailer": candidate.retailer,
                        "identifiers": candidate.identifiers,
                    },
                )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            logger.info("Browser Use helper unavailable for %s", candidate.id)
            return []
        text = payload.get("content") if isinstance(payload, dict) else None
        final_url = payload.get("url") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            return []
        source_url = final_url if isinstance(final_url, str) and is_public_http_url(final_url) else candidate.productUrl
        return [DimensionContent(self.name, source_url, candidate.retailer or _host(source_url), text=text)]


class PragmaticDimensionResolver:
    """FAST structured lookup, parallel rendered/search lookup, Browser Use last."""

    def __init__(
        self,
        fast_resolver: DimensionResolver,
        extractor: DimensionExtractor,
        medium_providers: list[DimensionContentProvider],
        browser_provider: DimensionContentProvider | None,
        *,
        page_max_chars: int,
    ) -> None:
        self._fast = fast_resolver
        self._extractor = extractor
        self._medium = medium_providers
        self._browser = browser_provider
        self._page_max_chars = page_max_chars
        self.last_trace: dict[str, Any] = {}

    async def resolve(self, candidate: ProductCandidate) -> ResolvedDimensions:
        started = time.monotonic()
        timings: dict[str, float] = {}
        self.last_trace = {
            "product": candidate.model_dump(exclude={"detailPageToken"}),
            "timings": timings,
            "semantic_attempts": [],
        }

        provider_result = _provider_dimensions(candidate)
        if provider_result is not None and _enough(provider_result):
            timings["structured"] = 0.0
            return self._finish(provider_result, "structured", started)

        mark = time.monotonic()
        fast = await self._fast.resolve(candidate)
        timings["structured"] = round(time.monotonic() - mark, 2)
        fast_trace = getattr(self._fast, "last_trace", {})
        self.last_trace["structured_trace"] = fast_trace
        if _enough(fast):
            return self._finish(fast, "structured", started)
        best = provider_result or (fast if fast.status != "unavailable" else None)

        # The fast resolver has already produced compact, variant-scoped provider/page
        # representations. Give those directly to Qwen before launching a browser.
        # This preserves the existing successful M6 path for fields such as a plain
        # "Product Dimensions: A x B x C" triplet.
        for attempt in fast_trace.get("attempts", []):
            content = DimensionContent(
                provider="structured_semantic",
                source_url=attempt.url or candidate.productUrl,
                source_label=attempt.name or candidate.retailer or "Product source",
                text="placeholder",
            )
            result = await self._extract_rep(candidate, attempt.rep, content, timings)
            if result is not None and (best is None or _rank(result) > _rank(best)):
                best = result
            if result is not None and _enough(result):
                return self._finish(result, "structured_semantic", started)

        retrieval_candidate = _candidate_with_direct_url(candidate, getattr(self._fast, "last_trace", {}))

        async def timed_collect(provider: DimensionContentProvider):
            mark = time.monotonic()
            contents = await provider.collect(retrieval_candidate)
            return provider.name, contents, round(time.monotonic() - mark, 2)

        tasks = [asyncio.create_task(timed_collect(provider)) for provider in self._medium]
        try:
            for task in asyncio.as_completed(tasks):
                provider_name, contents, elapsed = await task
                timings[f"{provider_name}_collect"] = elapsed
                for content in contents:
                    result = await self._extract(candidate, content, timings)
                    if result is None:
                        continue
                    if best is None or _rank(result) > _rank(best):
                        best = result
                    if _enough(result):
                        for pending in tasks:
                            if not pending.done():
                                pending.cancel()
                        return self._finish(result, content.provider, started)
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)

        if self._browser is not None:
            mark = time.monotonic()
            contents = await self._browser.collect(retrieval_candidate)
            timings["browser_use_collect"] = round(time.monotonic() - mark, 2)
            for content in contents:
                result = await self._extract(candidate, content, timings)
                if result is not None and (best is None or _rank(result) > _rank(best)):
                    best = result
                if result is not None and _enough(result):
                    return self._finish(result, "browser_use", started)

        if best is not None:
            return self._finish(best, "partial", started)
        unavailable = ResolvedDimensions(
            productId=candidate.id,
            retryable=True,
            message=(fast.message or "No trustworthy footprint dimensions were found.")
            + " Rendered page, exact web search, and browser fallback returned no verified dimensions.",
        )
        return self._finish(unavailable, "unavailable", started)

    async def _extract(
        self,
        candidate: ProductCandidate,
        content: DimensionContent,
        timings: dict[str, float],
    ) -> ResolvedDimensions | None:
        rep = _representation(content, candidate)
        return await self._extract_rep(candidate, rep, content, timings)

    async def _extract_rep(
        self,
        candidate: ProductCandidate,
        rep: PageRepresentation,
        content: DimensionContent,
        timings: dict[str, float],
    ) -> ResolvedDimensions | None:
        if rep.is_empty:
            return None
        page_json = rep.to_llm_json(self._page_max_chars)
        debug_attempt: dict[str, Any] = {
            "provider": content.provider,
            "source_url": content.source_url,
            "source_label": content.source_label,
            "rep": rep,
            "page_json": page_json,
        }
        self.last_trace.setdefault("semantic_attempts", []).append(debug_attempt)
        mark = time.monotonic()
        try:
            extraction = await self._extractor.extract(rep, page_json)
            debug_attempt["raw_llm"] = getattr(self._extractor, "last_raw", None)
            debug_attempt["extraction"] = extraction
        except DimensionExtractionError as exc:
            debug_attempt["raw_llm"] = getattr(self._extractor, "last_raw", None)
            debug_attempt["error"] = str(exc)
            return None
        finally:
            timings[f"{content.provider}_qwen"] = round(time.monotonic() - mark, 2)
        validated = validate_semantic(extraction, rep)
        debug_attempt["validation"] = validated
        if not validated.accepted:
            logger.info(
                "Dimension fallback rejected provider=%s source=%s model=%s errors=%s",
                content.provider,
                content.source_url,
                extraction.status,
                "; ".join(validated.errors)[:500],
            )
            return None
        axis = validated.axis
        values = validated.values_m
        scope = validated.scope or "page"
        context = getattr(self._fast, "last_trace", {}).get("variant_context")
        variant_identity = (
            context.describe()
            if scope.startswith("exact") and callable(getattr(context, "describe", None))
            else None
        )

        def value(index: int | None) -> float | None:
            return values[index] if index is not None and index < len(values) else None

        return ResolvedDimensions(
            productId=candidate.id,
            widthMeters=value(axis.widthIndex),
            depthMeters=value(axis.depthIndex),
            heightMeters=value(axis.heightIndex),
            dimensionsMeters=values,
            axisMapping=AxisMapping(**axis.as_dict()),
            sourceType=validated.source_type or "page_text",
            sourceUrl=content.source_url,
            sourceName=content.source_label,
            rawDimensions=validated.raw_text,
            sourcePath=validated.source_path,
            extractionMethod="llm",
            variantScope=_VARIANT_SCOPE.get(scope, "retailer_page"),
            variantIdentity=variant_identity,
        )

    def _finish(self, result: ResolvedDimensions, winner: str, started: float) -> ResolvedDimensions:
        total = round(time.monotonic() - started, 2)
        self.last_trace["winner"] = winner
        self.last_trace["result"] = result.model_dump()
        self.last_trace["timings"]["total"] = total
        logger.info(
            "Dimension lookup product=%s winner=%s status=%s dimensions=%s source=%s timings=%s",
            result.productId,
            winner,
            result.status,
            result.dimensionsMeters,
            result.sourceUrl,
            self.last_trace["timings"],
        )
        return result


def _representation(content: DimensionContent, candidate: ProductCandidate) -> PageRepresentation:
    if content.html:
        return build_page_representation(
            content.html,
            content.source_url,
            product_title=candidate.title,
            retailer=content.source_label,
        )
    return build_text_representation(
        content.text or "",
        content.source_url,
        product_title=candidate.title,
        retailer=content.source_label,
        heading=f"{content.provider} evidence",
    )


def _enough(result: ResolvedDimensions) -> bool:
    return (
        result.widthMeters is not None and result.depthMeters is not None
    ) or len(result.dimensionsMeters or []) == 3


def _rank(result: ResolvedDimensions) -> tuple[int, int]:
    return (1 if _enough(result) else 0, len(result.dimensionsMeters or []))


def _dimension_queries(candidate: ProductCandidate) -> list[str]:
    identifiers = [value.strip() for value in candidate.identifiers.values() if value.strip()]
    queries = [f'"{identifier}" dimensions width depth height' for identifier in identifiers]
    title = " ".join(candidate.title.split(" | ", 1)[0].split())[:220]
    retailer = f" {candidate.retailer}" if candidate.retailer else ""
    queries.append(f'"{title}"{retailer} dimensions')
    # Keep order while removing duplicates; exact identifiers are deliberately first.
    return list(dict.fromkeys(queries))


def _host(url: str) -> str:
    return (urlparse(url).hostname or "Web source").removeprefix("www.")


def _candidate_with_direct_url(candidate: ProductCandidate, trace: dict[str, Any]) -> ProductCandidate:
    for attempt in trace.get("attempts", []):
        url = getattr(attempt, "url", None)
        host = (urlparse(url).hostname or "").lower() if isinstance(url, str) else ""
        if url and host and host != "google.com" and not host.endswith(".google.com"):
            return candidate.model_copy(update={"productUrl": url})
    return candidate


def _provider_dimensions(candidate: ProductCandidate) -> ResolvedDimensions | None:
    existing = candidate.dimensions
    if existing.widthMeters is not None or existing.depthMeters is not None or existing.heightMeters is not None:
        return ResolvedDimensions(
            productId=candidate.id,
            widthMeters=existing.widthMeters,
            depthMeters=existing.depthMeters,
            heightMeters=existing.heightMeters,
            sourceType="structured_metadata",
            sourceUrl=candidate.productUrl,
            sourceName=existing.source or candidate.retailer or "Shopping provider",
            rawDimensions="Provider-supplied explicit dimensions",
            extractionMethod="structured",
            variantScope="product_family",
        )
    # Google Shopping often includes explicit W/D/H measurements directly in the
    # selected offer title. Accept this only for an actual SerpApi Google Shopping
    # candidate; arbitrary caller/debug titles are not measurement provenance.
    host = (urlparse(candidate.productUrl).hostname or "").lower()
    if candidate.provider != "serpapi" or (host != "google.com" and not host.endswith(".google.com")):
        return None
    parsed = parse_labeled_dimensions(candidate.title, require_keyword=False)
    if parsed.width is None or parsed.depth is None or parsed.height is None:
        return None
    return ResolvedDimensions(
        productId=candidate.id,
        widthMeters=parsed.width,
        depthMeters=parsed.depth,
        heightMeters=parsed.height,
        sourceType="structured_metadata",
        sourceUrl=candidate.productUrl,
        sourceName=candidate.retailer or "Shopping provider",
        rawDimensions=parsed.raw_text(),
        extractionMethod="structured",
        variantScope="product_family",
    )


_VARIANT_SCOPE = {
    "exact_record": "exact_variant",
    "exact_page": "exact_variant_page",
    "family": "product_family",
    "page": "retailer_page",
}

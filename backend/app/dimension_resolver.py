from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from typing import Protocol

from .dimension_evidence import DEFAULT_MAX_CHARS, build_dimension_evidence
from .dimension_extraction import SOURCE_PRIORITY, SourceFinding, extract_from_features, extract_from_html
from .dimension_llm import DimensionExtraction, DimensionExtractionError, extract_and_validate
from .dimension_models import ResolvedDimensions
from .dimension_parsing import DimensionCandidate, parse_field
from .page_fetcher import FetchedPage, PageFetcher, PageFetchError, is_public_http_url
from .product_details import ProductDetailSource, StoreLink
from .product_models import ProductCandidate
from .product_providers import ProductProviderError, ProductProviderNotConfiguredError
from .variant_context import build_variant_context
from .variant_scope import find_variant_records, ids_for_selected_options, record_evidence

logger = logging.getLogger(__name__)
MAX_RETAILER_PAGES = 2


class DimensionExtractor(Protocol):
    async def extract(self, evidence: str) -> DimensionExtraction: ...


@dataclass(frozen=True)
class _Failure:
    message: str
    retryable: bool


@dataclass(frozen=True)
class _NamedFinding:
    finding: SourceFinding
    source_name: str | None
    scope: str = "page"  # exact_record | exact_page | family | page


class DimensionResolver:
    """Resolves trustworthy dimensions for one selected product.

    Hierarchy (highest first):
      1. provider product-detail structured specs (SerpApi 'about_the_product' fields)
      2. retailer JSON-LD
      3. retailer structured metadata (microdata) and specification tables
      4. cleaned retailer spec/page text -> local LLM *extraction* -> deterministic validation
      5. deterministic labeled-text parsing of the page (legacy fallback)
      6. unavailable
    A source stating all three axes wins by priority; otherwise the highest priority
    source with width and depth; otherwise any partial result. Sources are never
    mixed and nothing is estimated. Cheaper sources short-circuit: retailer pages are
    only fetched when the provider lacks width+depth, and the LLM only runs when no
    structured page source has them.
    """

    def __init__(
        self,
        detail_source: ProductDetailSource | None,
        page_fetcher: PageFetcher,
        *,
        extractor: DimensionExtractor | None = None,
        evidence_max_chars: int = DEFAULT_MAX_CHARS,
        max_pages: int = MAX_RETAILER_PAGES,
    ) -> None:
        self._detail_source = detail_source
        self._page_fetcher = page_fetcher
        self._extractor = extractor
        self._evidence_max_chars = evidence_max_chars
        self._max_pages = max_pages
        self.last_trace: dict[str, object] = {}

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
        trace: dict[str, object] = {}
        self.last_trace = trace
        direct_url = candidate.productUrl if _is_retailer_url(candidate.productUrl) else None
        if direct_url:
            stores.append(StoreLink(candidate.retailer, direct_url, None, candidate.price))

        detail = None
        if self._detail_source is not None and candidate.detailPageToken:
            try:
                detail = await self._detail_source.fetch(candidate)
            except ProductProviderNotConfiguredError:
                failures.append(_Failure("Product details provider is not configured.", False))
            except ProductProviderError as exc:
                failures.append(_Failure(str(exc), True))
            if detail is not None:
                for finding in extract_from_features(detail.features, candidate.productUrl):
                    findings.append(_NamedFinding(finding, detail.source_name, scope="family"))
                stores.extend(_order_stores(detail.stores, candidate.retailer))

        context = build_variant_context(
            candidate.productUrl, candidate.retailer, candidate.price,
            detail.stores if detail else [], detail.selected_options if detail else {}, direct_url,
        )
        trace["variant_context"] = context
        fetched: list[tuple[StoreLink, FetchedPage]] = []
        attempted: set[str] = set()

        # 1. Exact variant identity: the selected offer's own retailer record/page outranks
        #    SerpApi's product-family specs. Explicit selected options can establish the
        #    identity only when they match exactly one item ID in the retailer's own data.
        if context.identity == "options_only" and context.offer is not None:
            offer_store = StoreLink(context.offer.storeName, context.offer.storeUrl, context.offer.storeTitle, context.offer.price)
            attempted.add(offer_store.url)
            page = await self._fetch(offer_store, failures)
            if page is not None:
                fetched.append((offer_store, page))
                matches = ids_for_selected_options(page.html, context.selectedOptions)
                trace["option_matches"] = sorted(matches)
                if len(matches) == 1:
                    context.retailerIds.variantId = next(iter(matches))
                    context.identity = "exact_item"
                    context.reason = "identified by explicitly selected options"
                    await self._exact_findings(context, offer_store, page, findings, failures, trace, page_level=False)
                else:
                    context.reason = ("selected options match no single retailer item"
                                      if not matches else "selected options match several retailer items")
        elif context.exact and context.offer is not None:
            offer_store = StoreLink(context.offer.storeName, context.offer.storeUrl, context.offer.storeTitle, context.offer.price)
            attempted.add(offer_store.url)
            page = await self._fetch(offer_store, failures)
            if page is not None:
                fetched.append((offer_store, page))
                await self._exact_findings(context, offer_store, page, findings, failures, trace)

        # 2. Otherwise (or if the exact offer gave no width+depth): SerpApi family specs,
        #    then other retailer pages, as before.
        pages: list[StoreLink] = []
        if not _has_footprint(findings):
            pages = [s for s in _unique_pages(stores) if s.url not in attempted][: self._max_pages]
            results = await asyncio.gather(*(self._page_fetcher.fetch(s.url) for s in pages), return_exceptions=True)
            for store, result in zip(pages, results):
                if isinstance(result, PageFetchError):
                    failures.append(_Failure(f"{store.name or 'Retailer'}: {result}", result.retryable))
                elif isinstance(result, BaseException):
                    failures.append(_Failure(f"{store.name or 'Retailer'}: page could not be read.", True))
                else:
                    fetched.append((store, result))
                    for finding in extract_from_html(result.html, result.url):
                        findings.append(_NamedFinding(finding, store.name, scope="page"))

            if self._extractor is not None and not _has_footprint(findings, exclude=("page_text",)):
                for store, page in fetched:
                    if store.url in attempted:
                        continue  # the exact offer page was already handled in step 1
                    if await self._llm_finding(page.html and build_dimension_evidence(page.html, self._evidence_max_chars),
                                               page.url, store.name, "page", findings, failures):
                        break

        chosen = _choose(findings)
        trace["chosen"] = chosen
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
                variantScope=_SCOPE_LABEL[chosen.scope],
                variantIdentity=context.describe() if chosen.scope.startswith("exact") else None,
            )

        if not stores and not candidate.detailPageToken:
            message = "No retailer page or product details are available for this product."
        elif failures and all(not f.retryable for f in failures) and not findings:
            message = "No explicit dimensions found. " + "; ".join(f.message for f in failures[:3])
        else:
            message = "No explicit product dimensions were found in retailer or provider data."
            if failures:
                message += " Some sources failed: " + "; ".join(f.message for f in failures[:3])
        if not context.exact and context.reason:
            message += f" Variant not identified exactly: {context.reason}."
        return ResolvedDimensions(
            productId=candidate.id,
            retryable=any(f.retryable for f in failures),
            message=message[:400],
        )

    async def _fetch(self, store: StoreLink, failures: list[_Failure]) -> FetchedPage | None:
        try:
            return await self._page_fetcher.fetch(store.url)
        except PageFetchError as exc:
            failures.append(_Failure(f"{store.name or 'Retailer'}: {exc}", exc.retryable))
        except Exception:
            failures.append(_Failure(f"{store.name or 'Retailer'}: page could not be read.", True))
        return None

    async def _exact_findings(self, context, store: StoreLink, page: FetchedPage,
                              findings: list[_NamedFinding], failures: list[_Failure], trace: dict,
                              page_level: bool = True) -> None:
        name = f"{store.name or 'Retailer'} (selected variant)"
        records = find_variant_records(page.html, context.retailerIds.all())
        evidence = record_evidence(records, self._evidence_max_chars)
        trace["variant_records"] = [r.path for r in records if r.has_content]
        trace["record_evidence"] = evidence
        if evidence:
            structured = DimensionCandidate()
            for record in records:
                for label, value in record.pairs:
                    structured.merge(parse_field(label, value))
            if structured.known_axes:
                findings.append(_NamedFinding(SourceFinding("structured_metadata", structured, page.url), name, scope="exact_record"))
            if not _has_footprint([f for f in findings if f.scope == "exact_record"]):
                await self._llm_finding(evidence, page.url, name, "exact_record", findings, failures)
        if _has_footprint([f for f in findings if f.scope.startswith("exact")]) or not page_level:
            return
        # The exact item's own page, when no scoped record carried dimensions (e.g. pages
        # rendered for one item). Sibling variants on the page still trigger conflict drops.
        for finding in extract_from_html(page.html, page.url):
            findings.append(_NamedFinding(finding, store.name, scope="exact_page"))
        if not _has_footprint([f for f in findings if f.scope.startswith("exact")], exclude=("page_text",)):
            page_evidence = build_dimension_evidence(page.html, self._evidence_max_chars)
            trace["page_evidence"] = page_evidence
            await self._llm_finding(page_evidence, page.url, store.name, "exact_page", findings, failures)

    async def _llm_finding(self, evidence: str, url: str, name: str | None, scope: str,
                           findings: list[_NamedFinding], failures: list[_Failure]) -> bool:
        """Runs the extractor on already-scoped evidence. True when width+depth were found."""
        if self._extractor is None or not evidence:
            return False
        try:
            validated = await extract_and_validate(self._extractor, evidence)
        except DimensionExtractionError as exc:
            failures.append(_Failure(f"{name or 'Retailer'}: {exc}", exc.retryable))
            return False
        if validated.rejected:
            logger.info("Dimension extraction rejections: %s", "; ".join(validated.rejected)[:300])
        if not validated.known_axes:
            return False
        dims = DimensionCandidate(width=validated.width, depth=validated.depth, height=validated.height,
                                  raw=[validated.evidence] if validated.evidence else [])
        findings.append(_NamedFinding(SourceFinding("page_text_llm", dims, url), name, scope=scope))
        return validated.width is not None and validated.depth is not None


# Exact-variant data outranks SerpApi's product-family specs; family specs outrank
# other (non-exact) retailer pages. A family/exact disagreement is not a conflict:
# the exact-variant value simply wins.
_SCOPE_RANK = {"exact_record": 0, "exact_page": 1, "family": 2, "page": 3}
_SCOPE_LABEL = {"exact_record": "exact_variant", "exact_page": "exact_variant_page", "family": "product_family", "page": "retailer_page"}


def _rank(named: _NamedFinding) -> tuple[int, int]:
    source_rank = -1 if named.scope == "family" else SOURCE_PRIORITY.index(named.finding.source_type)
    return _SCOPE_RANK[named.scope], source_rank


def _has_footprint(findings: list[_NamedFinding], exclude: tuple[str, ...] = ()) -> bool:
    return any(
        f.finding.candidate.width is not None and f.finding.candidate.depth is not None
        for f in findings if f.finding.source_type not in exclude
    )


def _choose(findings: list[_NamedFinding]) -> _NamedFinding | None:
    """Width+depth results first (by scope, then completeness, then source); otherwise
    the best partial result. Values from different sources are never mixed."""
    footprint = [f for f in findings if f.finding.candidate.width is not None and f.finding.candidate.depth is not None]
    if footprint:
        return min(footprint, key=lambda f: (_rank(f)[0], f.finding.candidate.known_axes != 3, _rank(f)[1]))
    partial = [f for f in findings if f.finding.candidate.known_axes > 0]
    if partial:
        return min(partial, key=lambda f: (_rank(f)[0], -f.finding.candidate.known_axes, _rank(f)[1]))
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
    """In-memory cache of resolved (verified/partial) results to spare provider and model calls."""

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
        # Only cache successful resolutions: an "unavailable" result may be fixed by
        # installing a model or a transient page problem, so it is re-checked next time.
        if result.retryable or result.status == "unavailable":
            return
        self._entries.pop(result.productId, None)
        self._entries[result.productId] = (time.monotonic(), result)
        while len(self._entries) > self._max:
            self._entries.pop(next(iter(self._entries)))

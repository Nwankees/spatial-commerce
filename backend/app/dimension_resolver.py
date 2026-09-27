from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from .dimension_models import AxisMapping, ResolvedDimensions
from .dimension_parsing import DimensionCandidate, axis_for_field_label, parse_field, parse_labeled_dimensions
from .dimension_semantic import (
    AxisAssignment,
    DimensionExtractionError,
    SemanticExtraction,
    SemanticResult,
    validate_semantic,
)
from .page_fetcher import FetchedPage, PageFetcher, PageFetchError, is_public_http_url
from .page_representation import (
    DEFAULT_MAX_CHARS,
    PageRepresentation,
    build_page_representation,
    build_provider_representation,
)
from .product_details import ProductDetailSource, StoreLink
from .product_models import ProductCandidate
from .product_providers import ProductProviderError, ProductProviderNotConfiguredError
from .variant_context import build_variant_context
from .variant_scope import ids_for_selected_options

logger = logging.getLogger(__name__)
MAX_RETAILER_PAGES = 2
MAX_LLM_CALLS = 3


class DimensionExtractor(Protocol):
    async def extract(self, rep: PageRepresentation, page_json: str) -> SemanticExtraction: ...


@dataclass(frozen=True)
class _Failure:
    message: str
    retryable: bool


@dataclass
class _Outcome:
    values_m: list[float]
    axis: AxisAssignment
    scope: str  # exact_record | exact_page | family | page
    method: str  # structured | llm
    source_type: str
    source_path: str | None
    raw_text: str | None
    source_name: str | None
    source_url: str | None
    labeled: dict[str, float] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.values_m)


@dataclass
class Attempt:
    """Debug trace of one representation (for the integration script / logs)."""
    name: str | None
    url: str | None
    scope: str
    rep: PageRepresentation
    page_json: str | None = None
    fast_path: dict[str, float] | None = None
    raw_llm: str | None = None
    extraction: SemanticExtraction | None = None
    validation: SemanticResult | None = None
    llm_seconds: float | None = None
    error: str | None = None


class DimensionResolver:
    """Resolves trustworthy overall dimensions for one selected product.

    Pipeline per source (exact selected-variant page first, then provider family specs,
    then other retailer pages): page -> generic structured representation (noise removed,
    structure kept, sibling variants excluded when the exact item is known) ->
      * fast path: explicit, separately labeled Width/Depth/Height fields in structured data;
      * otherwise the local model (qwen3:4b-instruct) chooses the field that states the
        overall dimensions -> deterministic provenance validation (source exists, every
        number/unit literally present, one section, selected variant) -> meters in code.
    Values from different sources are never mixed and nothing is estimated. A source with
    three values wins over partial ones; then exact variant > product family > other pages.
    """

    def __init__(
        self,
        detail_source: ProductDetailSource | None,
        page_fetcher: PageFetcher,
        *,
        extractor: DimensionExtractor | None = None,
        page_max_chars: int = DEFAULT_MAX_CHARS,
        max_pages: int = MAX_RETAILER_PAGES,
        max_llm_calls: int = MAX_LLM_CALLS,
        **_: Any,
    ) -> None:
        self._detail_source = detail_source
        self._page_fetcher = page_fetcher
        self._extractor = extractor
        self._page_max_chars = page_max_chars
        self._max_pages = max_pages
        self._max_llm_calls = max_llm_calls
        self.last_trace: dict[str, object] = {}

    async def resolve(self, candidate: ProductCandidate) -> ResolvedDimensions:
        try:
            return await self._resolve(candidate)
        except Exception as exc:  # Never let resolution failures escape as 500s.
            logger.warning("Dimension resolution failed unexpectedly: %s", type(exc).__name__, exc_info=True)
            return ResolvedDimensions(
                productId=candidate.id,
                retryable=True,
                message="Dimension lookup failed unexpectedly. Please retry.",
            )

    async def _resolve(self, candidate: ProductCandidate) -> ResolvedDimensions:
        outcomes: list[_Outcome] = []
        failures: list[_Failure] = []
        stores: list[StoreLink] = []
        attempts: list[Attempt] = []
        trace: dict[str, object] = {"attempts": attempts}
        self.last_trace = trace
        self._llm_calls = 0
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
                stores.extend(_order_stores(detail.stores, candidate.retailer))

        context = build_variant_context(
            candidate.productUrl, candidate.retailer, candidate.price,
            detail.stores if detail else [], detail.selected_options if detail else {}, direct_url,
        )
        trace["variant_context"] = context
        attempted: set[str] = set()
        complete = False

        # 1. The exact selected variant's own retailer page (sibling variants excluded).
        if context.offer is not None and context.identity in ("options_only", "exact_item", "exact_offer"):
            offer_store = StoreLink(context.offer.storeName, context.offer.storeUrl, context.offer.storeTitle, context.offer.price)
            attempted.add(offer_store.url)
            page = await self._fetch(offer_store, failures)
            if page is not None and context.identity == "options_only":
                matches = ids_for_selected_options(page.html, context.selectedOptions)
                trace["option_matches"] = sorted(matches)
                if len(matches) == 1:
                    context.retailerIds.variantId = next(iter(matches))
                    context.identity = "exact_item"
                    context.reason = "identified by explicitly selected options"
                else:
                    context.reason = ("selected options match no single retailer item"
                                      if not matches else "selected options match several retailer items")
            if page is not None and context.exact:
                rep = build_page_representation(
                    page.html, page.url, product_title=candidate.title, retailer=candidate.retailer,
                    context=context, exact_ids=context.retailerIds.all())
                if rep.variant_mismatch:
                    failures.append(_Failure(f"{offer_store.name or 'Retailer'}: the page is for a different item "
                                             f"than the selected variant.", False))
                else:
                    complete = await self._consider(rep, f"{offer_store.name or 'Retailer'} (selected variant)",
                                                    page.url, "exact", outcomes, failures, attempts)

        # 2. Provider (SerpApi) product-family specifications.
        if not complete and detail is not None and detail.features:
            rep = build_provider_representation(detail.features, product_title=candidate.title,
                                                retailer=candidate.retailer, source_name=detail.source_name)
            complete = await self._consider(rep, detail.source_name, candidate.productUrl, "family",
                                            outcomes, failures, attempts)

        # 3. Other retailer pages for the product (not variant-exact).
        if not complete:
            pages = [s for s in _unique_pages(stores) if s.url not in attempted][: self._max_pages]
            results = await asyncio.gather(*(self._page_fetcher.fetch(s.url) for s in pages), return_exceptions=True)
            for store, result in zip(pages, results):
                if isinstance(result, PageFetchError):
                    failures.append(_Failure(f"{store.name or 'Retailer'}: {result}", result.retryable))
                    continue
                if isinstance(result, BaseException):
                    failures.append(_Failure(f"{store.name or 'Retailer'}: page could not be read.", True))
                    continue
                rep = build_page_representation(result.html, result.url, product_title=candidate.title,
                                                retailer=candidate.retailer, context=context)
                if await self._consider(rep, store.name, result.url, "page", outcomes, failures, attempts):
                    break

        chosen = _choose(outcomes)
        trace["chosen"] = chosen
        if chosen is not None:
            axis = chosen.axis
            values = chosen.values_m

            def axis_value(index: int | None) -> float | None:
                return values[index] if index is not None and index < len(values) else None

            return ResolvedDimensions(
                productId=candidate.id,
                widthMeters=axis_value(axis.widthIndex),
                depthMeters=axis_value(axis.depthIndex),
                heightMeters=axis_value(axis.heightIndex),
                dimensionsMeters=values,
                axisMapping=AxisMapping(**axis.as_dict()),
                sourceType=_SOURCE_TYPE.get(chosen.source_type, "page_text"),
                sourceUrl=chosen.source_url,
                sourceName=chosen.source_name,
                rawDimensions=chosen.raw_text,
                sourcePath=chosen.source_path,
                extractionMethod=chosen.method,  # type: ignore[arg-type]
                variantScope=_SCOPE_LABEL[chosen.scope],
                variantIdentity=context.describe() if chosen.scope.startswith("exact") else None,
            )

        if not stores and not candidate.detailPageToken:
            message = "No retailer page or product details are available for this product."
        elif failures and all(not f.retryable for f in failures):
            message = "No trustworthy product dimensions found. " + "; ".join(f.message for f in failures[:3])
        else:
            message = "No trustworthy product dimensions found in retailer or provider data."
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

    async def _consider(self, rep: PageRepresentation, name: str | None, url: str | None, kind: str,
                        outcomes: list[_Outcome], failures: list[_Failure], attempts: list[Attempt]) -> bool:
        """Fast path, then the model. True when this source yielded three verified values."""
        attempt = Attempt(name, url, kind, rep)
        attempts.append(attempt)
        if rep.is_empty:
            attempt.error = "no page content"
            return False
        fast = _fast_path(rep, name, url)
        if fast is not None:
            if (_is_legacy_extractor(self._extractor) and fast.count >= 2
                    and fast.source_type == "embedded_json"):
                # Historical M5.5 named scoped application-state fields
                # "structured_metadata"; preserve that public result for old callers.
                fast.source_type = "provider_specs"
            attempt.fast_path = fast.labeled
            outcomes.append(fast)
            if fast.count == 3 or (_is_legacy_extractor(self._extractor) and fast.count >= 2):
                return True
        if self._extractor is None:
            # Compatibility for the explicitly labeled M5 page-text API only.
            # Production M6 config always supplies the semantic extractor.
            legacy_fast = _legacy_labeled_page_fast_path(rep, name, url)
            if legacy_fast is not None:
                outcomes.append(legacy_fast)
                return legacy_fast.count == 3
            return False
        if self._extractor is None or self._llm_calls >= self._max_llm_calls:
            return False
        if _is_legacy_extractor(self._extractor):
            return await self._consider_legacy(rep, name, url, outcomes, failures, attempt)
        page_json = rep.to_llm_json(self._page_max_chars)
        attempt.page_json = page_json
        self._llm_calls += 1
        started = time.monotonic()
        try:
            extraction = await self._extractor.extract(rep, page_json)
        except DimensionExtractionError as exc:
            attempt.error = str(exc)
            attempt.raw_llm = getattr(self._extractor, "last_raw", None)
            failures.append(_Failure(f"{name or 'Retailer'}: {exc}", exc.retryable))
            return False
        finally:
            attempt.llm_seconds = round(time.monotonic() - started, 2)
        attempt.raw_llm = getattr(self._extractor, "last_raw", None)
        attempt.extraction = extraction
        validation = validate_semantic(extraction, rep)
        attempt.validation = validation
        if not validation.accepted:
            logger.info("Dimension extraction not accepted (%s): %s", name, "; ".join(validation.errors)[:300])
            failures.append(_Failure(f"{name or 'Retailer'}: {validation.errors[0] if validation.errors else 'not accepted'}", False))
            return False
        outcomes.append(_Outcome(
            values_m=validation.values_m, axis=validation.axis, scope=validation.scope or "page", method="llm",
            source_type=validation.source_type or "page_text", source_path=validation.source_path,
            raw_text=validation.raw_text, source_name=name, source_url=url))
        return validation.count == 3

    async def _consider_legacy(
        self,
        rep: PageRepresentation,
        name: str | None,
        url: str | None,
        outcomes: list[_Outcome],
        failures: list[_Failure],
        attempt: Attempt,
    ) -> bool:
        """Adapter for M5/M5.5 injected extractors; not used by production M6."""
        from .dimension_llm import (
            DimensionExtractionError as LegacyExtractionError,
            validate_extraction,
        )

        evidence = "\n".join(entry.value for entry, _ in rep.all_entries())[:self._page_max_chars]
        self._llm_calls += 1
        started = time.monotonic()
        try:
            extraction = await self._extractor.extract(evidence)  # type: ignore[call-arg]
        except LegacyExtractionError as exc:
            attempt.error = str(exc)
            failures.append(_Failure(f"{name or 'Retailer'}: {exc}", exc.retryable))
            return False
        finally:
            attempt.llm_seconds = round(time.monotonic() - started, 2)
        validated = validate_extraction(extraction, evidence)
        if not validated.known_axes:
            failures.append(_Failure(f"{name or 'Retailer'}: model values were not supported by source labels", False))
            return False
        values: list[float] = []
        axis = AxisAssignment(source="labels", confidence=1.0, reason="axes explicitly labeled in source text")
        labeled: dict[str, float] = {}
        for axis_name in ("width", "depth", "height"):
            value = getattr(validated, axis_name)
            if value is not None:
                setattr(axis, f"{axis_name}Index", len(values))
                values.append(value)
                labeled[axis_name] = value
        matching_entries = [
            (entry, section) for entry, section in rep.all_entries()
            if extraction.evidence_text and extraction.evidence_text.lower() in entry.value.lower()
        ]
        source_entry = min(
            matching_entries,
            key=lambda item: _SCOPE_RANK.get(item[1].scope, 3),
            default=None,
        )
        scope = source_entry[1].scope if source_entry else "page"
        path = source_entry[0].path if source_entry else "legacy labeled page text"
        outcomes.append(_Outcome(
            values_m=values,
            axis=axis,
            scope=scope,
            method="llm",
            source_type="page_text_llm",
            source_path=path,
            raw_text=extraction.raw_dimensions or validated.evidence,
            source_name=name,
            source_url=url,
            labeled=labeled,
        ))
        return len(values) == 3


def _fast_path(rep: PageRepresentation, name: str | None, url: str | None) -> _Outcome | None:
    """Explicit, separately named Width / Depth / Height fields in structured data
    (JSON-LD, spec tables, embedded name/value data, provider specs) of one section.
    Only field *names* that are exactly an axis (e.g. 'Width', 'Overall Height (in)') count."""
    best: _Outcome | None = None
    for section in rep.sections:
        if section.kind == "page_text":
            continue
        if section.scope == "page" and section.owner and len(rep.owners()) > 1:
            continue  # one of several items; the variant is not identified
        found = DimensionCandidate()
        for entry in section.entries:
            axis, _ = axis_for_field_label(entry.name or "")
            if axis:
                found.merge(parse_field(entry.name or "", entry.value))
        if not found.known_axes:
            continue
        labeled = {a: getattr(found, a) for a in ("width", "depth", "height") if getattr(found, a) is not None}
        values = list(labeled.values())
        axis = AxisAssignment(source="structured_fields", confidence=1.0,
                              reason="separately labeled width/depth/height fields")
        for i, a in enumerate(labeled):
            setattr(axis, f"{a}Index", i)
        outcome = _Outcome(values, axis, section.scope, "structured", section.kind,
                           f"{section.heading} > " + ", ".join(a.capitalize() for a in labeled),
                           found.raw_text(), name, url, labeled)
        if best is None or outcome.count > best.count:
            best = outcome
    return best


_SCOPE_RANK = {"exact_record": 0, "exact_page": 1, "family": 2, "page": 3}
_SCOPE_LABEL = {"exact_record": "exact_variant", "exact_page": "exact_variant_page", "family": "product_family",
                "page": "retailer_page"}
_SOURCE_TYPE = {"json_ld": "json_ld", "spec_table": "spec_table", "embedded_json": "embedded_json",
                "page_text_llm": "page_text_llm",
                "page_text": "page_text", "provider_specs": "structured_metadata"}


def _is_legacy_extractor(extractor: Any) -> bool:
    if extractor is None:
        return False
    try:
        return len(inspect.signature(extractor.extract).parameters) == 1
    except (TypeError, ValueError):
        return False


def _legacy_labeled_page_fast_path(
    rep: PageRepresentation,
    name: str | None,
    url: str | None,
) -> _Outcome | None:
    best: _Outcome | None = None
    for entry, section in rep.all_entries():
        candidate = parse_labeled_dimensions(entry.text, require_keyword=True)
        if not candidate.known_axes:
            continue
        labeled = {
            axis_name: getattr(candidate, axis_name)
            for axis_name in ("width", "depth", "height")
            if getattr(candidate, axis_name) is not None
        }
        values = list(labeled.values())
        axis = AxisAssignment(source="labels", confidence=1.0, reason="axes explicitly labeled in source text")
        for index, axis_name in enumerate(labeled):
            setattr(axis, f"{axis_name}Index", index)
        outcome = _Outcome(
            values,
            axis,
            section.scope,
            "structured",
            "page_text",
            entry.path,
            candidate.raw_text(),
            name,
            url,
            labeled,
        )
        if best is None or outcome.count > best.count:
            best = outcome
    return best


def _choose(outcomes: list[_Outcome]) -> _Outcome | None:
    """Three values first; then exact variant > family > other page; structured before model."""
    if not outcomes:
        return None
    return min(outcomes, key=lambda o: (-min(o.count, 3), _SCOPE_RANK.get(o.scope, 3), o.method != "structured"))


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

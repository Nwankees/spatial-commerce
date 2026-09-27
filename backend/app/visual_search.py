from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from .product_models import ProductCandidate, RetrievalSource
from .product_providers import (
    ProductProviderError,
    ProductProviderMalformedResponseError,
    ProductProviderNotConfiguredError,
    ProductProviderTimeoutError,
    ProductProviderUnavailableError,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

SERPAPI_IMAGE_ENDPOINT = "https://serpapi.com/image"
SERPAPI_SEARCH_ENDPOINT = "https://serpapi.com/search.json"
LensMode = Literal["products", "visual_matches", "exact_matches"]

_SOURCE_BY_MODE: dict[LensMode, RetrievalSource] = {
    "products": "lens_products",
    "visual_matches": "lens_visual_match",
    "exact_matches": "lens_exact_match",
}
_TRACKING_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "ref", "ref_", "tag",
}


@dataclass(frozen=True)
class VisualProductCandidate:
    product: ProductCandidate
    mode: LensMode
    rank: int


@dataclass(frozen=True)
class VisualSearchResult:
    candidates: list[VisualProductCandidate]
    upload_ms: int
    search_ms: int
    mode_errors: dict[str, str] = field(default_factory=dict)


class VisualSearchProvider(Protocol):
    name: str

    @property
    def configured(self) -> bool: ...

    async def search(
        self,
        image: bytes,
        category_hint: str | None,
        text_hint: str | None,
    ) -> VisualSearchResult: ...


class SerpApiGoogleLensProvider:
    """Direct-upload Google Lens provider.

    The photographed image is uploaded once to SerpApi's Image API. The
    short-lived image_id is then reused for independent Lens modes; the phone
    image never needs a public URL.
    """

    name = "serpapi_google_lens"

    def __init__(
        self,
        api_key: str | None,
        *,
        timeout_seconds: float = 20.0,
        country: str = "us",
        language: str = "en",
        modes: tuple[LensMode, ...] = ("products", "visual_matches", "exact_matches"),
        limit_per_mode: int = 12,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._country = country.lower()
        self._language = language.lower()
        self._modes = modes
        self._limit = max(1, limit_per_mode)
        self._transport = transport

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    async def search(
        self,
        image: bytes,
        category_hint: str | None,
        text_hint: str | None,
    ) -> VisualSearchResult:
        if not self._api_key:
            raise ProductProviderNotConfiguredError("SERPAPI_API_KEY is not configured.")
        if not image:
            raise ProductProviderMalformedResponseError("The visual-search image is empty.")

        upload_started = time.perf_counter()
        image_id = await self._upload(image)
        upload_ms = _elapsed_ms(upload_started)

        search_started = time.perf_counter()
        outcomes = await asyncio.gather(
            *(self._search_mode(image_id, mode, category_hint, text_hint) for mode in self._modes),
            return_exceptions=True,
        )
        search_ms = _elapsed_ms(search_started)
        candidates: list[VisualProductCandidate] = []
        errors: dict[str, str] = {}
        for mode, outcome in zip(self._modes, outcomes):
            if isinstance(outcome, BaseException):
                safe_reason = _safe_remote_text(str(outcome), self._api_key) or type(outcome).__name__
                errors[mode] = f"{type(outcome).__name__}: {safe_reason}"
                logger.warning(
                    "Google Lens mode=%s failed type=%s reason=%s",
                    mode,
                    type(outcome).__name__,
                    safe_reason,
                )
            else:
                candidates.extend(outcome)
        if errors and len(errors) == len(self._modes):
            first = next(outcome for outcome in outcomes if isinstance(outcome, BaseException))
            if isinstance(first, ProductProviderError):
                raise first
            raise ProductProviderUnavailableError("Every Google Lens mode failed.") from None
        return VisualSearchResult(candidates, upload_ms, search_ms, errors)

    async def _upload(self, image: bytes) -> str:
        content_type = "image/jpeg"
        logger.info(
            "SerpApi image upload request endpoint=%s image_bytes=%s content_type=%s",
            SERPAPI_IMAGE_ENDPOINT,
            len(image),
            content_type,
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(
                    SERPAPI_IMAGE_ENDPOINT,
                    data={"api_key": self._api_key},
                    files={"image": ("camera.jpg", image, content_type)},
                )
        except httpx.TimeoutException:
            raise ProductProviderTimeoutError("The Google Lens image upload timed out.") from None
        except httpx.HTTPError as exc:
            logger.warning("SerpApi image upload failed: %s", type(exc).__name__)
            raise ProductProviderUnavailableError("The Google Lens image upload is unreachable.") from None
        try:
            payload = _json_object(response)
        except ProductProviderMalformedResponseError:
            logger.warning(
                "SerpApi image upload response endpoint=%s status=%s content_type=%s invalid_json=true",
                SERPAPI_IMAGE_ENDPOINT,
                response.status_code,
                response.headers.get("content-type"),
            )
            raise
        image_id = _text(payload.get("image_id"))
        logger.info(
            "SerpApi image upload response endpoint=%s status=%s response=%s returned_image_id=%s",
            SERPAPI_IMAGE_ENDPOINT,
            response.status_code,
            _safe_upload_response(payload, self._api_key),
            image_id,
        )
        if not 200 <= response.status_code < 300:
            detail = _serpapi_error(payload, self._api_key) or _serpapi_message(payload, self._api_key)
            raise ProductProviderUnavailableError(
                f"The Google Lens image upload returned HTTP {response.status_code}"
                + (f": {detail}" if detail else ".")
            )
        if not image_id:
            detail = _serpapi_error(payload, self._api_key) or _serpapi_message(payload, self._api_key)
            if detail:
                raise ProductProviderUnavailableError(f"The Google Lens image upload was rejected: {detail}")
            raise ProductProviderMalformedResponseError("The image upload response had no image_id.")
        return image_id

    async def _search_mode(
        self,
        image_id: str,
        mode: LensMode,
        category_hint: str | None,
        text_hint: str | None,
    ) -> list[VisualProductCandidate]:
        params: dict[str, str] = {
            "engine": "google_lens",
            "type": mode,
            "image_id": image_id,
            "country": self._country,
            "hl": self._language,
            "auto_crop": "true",
            "safe": "active",
            "api_key": self._api_key or "",
        }
        if mode in ("products", "visual_matches"):
            hint = _text(text_hint) or _text(category_hint)
            if hint:
                params["q"] = hint[:200]
        logger.info(
            "SerpApi Lens request endpoint=%s engine=google_lens mode=%s image_id=%s",
            SERPAPI_SEARCH_ENDPOINT,
            mode,
            image_id,
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.get(SERPAPI_SEARCH_ENDPOINT, params=params)
        except httpx.TimeoutException:
            raise ProductProviderTimeoutError(f"Google Lens {mode} timed out.") from None
        except httpx.HTTPError as exc:
            logger.warning("Google Lens mode=%s failed: %s", mode, type(exc).__name__)
            raise ProductProviderUnavailableError(f"Google Lens {mode} is unreachable.") from None
        try:
            payload = _json_object(response)
        except ProductProviderMalformedResponseError:
            logger.warning(
                "SerpApi Lens response engine=google_lens mode=%s image_id=%s status=%s "
                "content_type=%s invalid_json=true accepted=false",
                mode,
                image_id,
                response.status_code,
                response.headers.get("content-type"),
            )
            raise
        metadata = payload.get("search_metadata")
        metadata_status = _text(metadata.get("status")) if isinstance(metadata, dict) else None
        search_id = _text(metadata.get("id")) if isinstance(metadata, dict) else None
        error = _serpapi_error(payload, self._api_key)
        message = _serpapi_message(payload, self._api_key)
        no_results = _is_no_results_error(error) and (metadata_status or "").casefold() == "success"
        accepted = (
            200 <= response.status_code < 300
            and (not error or no_results)
            and (metadata_status or "").casefold() != "error"
        )
        logger.info(
            "SerpApi Lens response engine=google_lens mode=%s image_id=%s status=%s "
            "search_metadata_status=%s search_id=%s error=%s message=%s accepted=%s",
            mode,
            image_id,
            response.status_code,
            metadata_status,
            search_id,
            error,
            message,
            accepted,
        )
        if not 200 <= response.status_code < 300:
            raise ProductProviderUnavailableError(
                f"Google Lens {mode} returned HTTP {response.status_code}"
                + (f": {error or message}" if error or message else ".")
            )
        if no_results:
            logger.info("Google Lens mode=%s completed with no results", mode)
            return []
        if error or (metadata_status or "").casefold() == "error":
            raise ProductProviderUnavailableError(
                f"Google Lens {mode} rejected the request"
                + (f": {error}" if error else ".")
            )
        raw_results = _mode_results(payload, mode)
        normalized = self.normalize(raw_results, mode, self._limit)
        logger.info(
            "Google Lens mode=%s results=%s",
            mode,
            [{"rank": item.rank, "title": item.product.title, "url": item.product.productUrl}
             for item in normalized],
        )
        return normalized

    def normalize(
        self,
        raw_results: list[Any],
        mode: LensMode,
        limit: int,
    ) -> list[VisualProductCandidate]:
        candidates: list[VisualProductCandidate] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_results):
            if len(candidates) >= limit:
                break
            if not isinstance(raw, dict):
                continue
            title = _text(raw.get("title"))
            link = _http_url(raw.get("link")) or _http_url(raw.get("product_link"))
            if not title or not link:
                continue
            canonical = canonical_product_url(link)
            if canonical in seen:
                continue
            seen.add(canonical)
            rank = _positive_int(raw.get("position")) or index + 1
            price, price_text, currency = _price(raw.get("price"), raw)
            provider_id = _text(raw.get("product_id")) or "url-" + hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()[:16]
            identifiers = _identifiers(raw)
            product = ProductCandidate(
                id=f"serpapi-lens:{provider_id}",
                provider=self.name,
                providerProductId=provider_id,
                position=rank,
                title=title,
                price=price,
                priceText=price_text,
                currency=_currency_code(currency, price_text, self._country),
                retailer=_text(raw.get("source")) or _host_label(link),
                imageUrl=_http_url(raw.get("thumbnail")) or _http_url(raw.get("image")),
                productUrl=link,
                rating=_bounded_number(raw.get("rating"), 0, 5),
                reviewCount=_nonnegative_int(raw.get("reviews")),
                inStock=raw.get("in_stock") if isinstance(raw.get("in_stock"), bool) else None,
                retrievalSources=[_SOURCE_BY_MODE[mode]],
                visualRank=rank,
                identifiers=identifiers,
            )
            candidates.append(VisualProductCandidate(product, mode, rank))
        return candidates


def _mode_results(payload: dict[str, Any], mode: LensMode) -> list[Any]:
    # SerpApi currently returns Lens product-tab rows under visual_matches;
    # accept a mode-named list as well to remain compatible with response updates.
    for key in (mode, "visual_matches", "exact_matches"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    if isinstance(payload.get("search_metadata"), dict):
        return []
    raise ProductProviderMalformedResponseError(f"Google Lens {mode} returned no results list.")


def canonical_product_url(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    port = f":{parts.port}" if parts.port and parts.port not in (80, 443) else ""
    path = parts.path.rstrip("/") or "/"
    query = urlencode(
        sorted((key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
               if key.lower() not in _TRACKING_QUERY_KEYS)
    )
    return urlunsplit((parts.scheme.lower(), host + port, path, query, ""))


def _price(value: Any, raw: dict[str, Any]) -> tuple[float | None, str | None, str | None]:
    if isinstance(value, dict):
        number = _number(value.get("extracted_value"))
        text = _text(value.get("value"))
        currency = _text(value.get("currency"))
    else:
        number = _number(raw.get("extracted_price"))
        text = _text(value)
        currency = None
    if number is None or number < 0:
        return None, text, currency
    return number, text, currency


def _currency_code(value: str | None, price_text: str | None, country: str) -> str | None:
    if value and len(value) == 3 and value.isalpha():
        return value.upper()
    symbol = value or ((price_text or "").strip()[:1] or None)
    if symbol == "$":
        return {"us": "USD", "ca": "CAD", "au": "AUD"}.get(country)
    if symbol == "£" and country in {"uk", "gb"}:
        return "GBP"
    if symbol == "€":
        return "EUR"
    if symbol == "₹" and country == "in":
        return "INR"
    if symbol == "¥" and country == "jp":
        return "JPY"
    return None


def _identifiers(raw: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("gtin", "upc", "sku", "model", "model_number", "merchant_product_id", "product_id"):
        value = _text(raw.get(key))
        if value:
            result[key] = value[:160]
    return result


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        raise ProductProviderMalformedResponseError("SerpApi returned invalid JSON.")
    return payload


def _safe_upload_response(payload: dict[str, Any], secret: str | None) -> dict[str, str | None]:
    """Keep the complete documented upload response while excluding unknown fields."""
    return {
        "message": _safe_remote_text(payload.get("message"), secret),
        "error": _safe_remote_text(payload.get("error"), secret),
        "image_id": _text(payload.get("image_id")),
    }


def _serpapi_error(payload: dict[str, Any], secret: str | None) -> str | None:
    return _safe_remote_text(payload.get("error"), secret)


def _serpapi_message(payload: dict[str, Any], secret: str | None) -> str | None:
    return _safe_remote_text(payload.get("message"), secret)


def _is_no_results_error(value: str | None) -> bool:
    normalized = (value or "").casefold()
    return "no results" in normalized or "hasn't returned any results" in normalized


def _safe_remote_text(value: Any, secret: str | None, limit: int = 300) -> str | None:
    text = _text(value)
    if not text:
        return None
    if secret:
        text = text.replace(secret, "[redacted]")
    return text[:limit]


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value or None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _bounded_number(value: Any, minimum: float, maximum: float) -> float | None:
    number = _number(value)
    return number if number is not None and minimum <= number <= maximum else None


def _positive_int(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number >= 1 else None


def _nonnegative_int(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number >= 0 else None


def _http_url(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    parts = urlsplit(text)
    return text if parts.scheme in ("http", "https") and parts.netloc else None


def _host_label(url: str) -> str | None:
    host = (urlsplit(url).hostname or "").removeprefix("www.")
    return host or None


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))

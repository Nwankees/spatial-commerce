from __future__ import annotations

import hashlib
import logging
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from .product_models import ProductCandidate
from .product_providers import (
    ProductProviderMalformedResponseError,
    ProductProviderNotConfiguredError,
    ProductProviderTimeoutError,
    ProductProviderUnavailableError,
)
from .query_builder import ProductQuery

logger = logging.getLogger(__name__)
# httpx logs full request URLs at INFO, and SerpApi takes the key as a query
# parameter, so keep httpx quiet regardless of the server's logging config.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
# SerpApi reports "no results" as a 200 response with an error message.
_NO_RESULTS_MARKERS = ("hasn't returned any results", "returned no results", "no results")
# Prices are only mapped to a currency when the symbol is unambiguous for the
# requested Google country; otherwise currency stays null.
_CURRENCY_BY_COUNTRY_AND_SYMBOL = {
    ("us", "$"): "USD",
    ("ca", "$"): "CAD",
    ("au", "$"): "AUD",
    ("uk", "£"): "GBP",
    ("gb", "£"): "GBP",
    ("in", "₹"): "INR",
    ("jp", "¥"): "JPY",
}
_EURO_COUNTRIES = {"de", "fr", "es", "it", "nl", "ie", "at", "be", "pt", "fi"}


class SerpApiProductSearchProvider:
    name = "serpapi"

    def __init__(
        self,
        api_key: str | None,
        *,
        timeout_seconds: float = 15.0,
        country: str = "us",
        language: str = "en",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._country = country.lower()
        self._language = language.lower()
        self._transport = transport

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    async def search(self, query: ProductQuery, limit: int) -> list[ProductCandidate]:
        if not self._api_key:
            raise ProductProviderNotConfiguredError("SERPAPI_API_KEY is not configured.")

        params = {
            "engine": "google_shopping",
            "q": query.text,
            "gl": self._country,
            "hl": self._language,
            "api_key": self._api_key,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.get(SERPAPI_ENDPOINT, params=params)
        except httpx.TimeoutException:
            raise ProductProviderTimeoutError("The product search provider timed out.") from None
        except httpx.HTTPError as exc:
            # Exception text can include the request URL (and therefore the key).
            logger.warning("SerpApi request failed: %s", type(exc).__name__)
            raise ProductProviderUnavailableError("The product search provider is unreachable.") from None

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if response.status_code != 200:
            logger.warning("SerpApi returned HTTP %s", response.status_code)
            raise ProductProviderUnavailableError(
                f"The product search provider returned HTTP {response.status_code}."
            )
        if not isinstance(payload, dict):
            raise ProductProviderMalformedResponseError("The product search provider returned invalid JSON.")

        error = payload.get("error")
        if isinstance(error, str) and error:
            if any(marker in error.lower() for marker in _NO_RESULTS_MARKERS):
                return []
            logger.warning("SerpApi reported an error for the search request.")
            raise ProductProviderUnavailableError("The product search provider rejected the request.")

        raw_results = _collect_raw_results(payload)
        if raw_results is None:
            raise ProductProviderMalformedResponseError(
                "The product search provider response had no shopping results list."
            )
        return self.normalize(raw_results, limit)

    def normalize(self, raw_results: list[Any], limit: int) -> list[ProductCandidate]:
        products: list[ProductCandidate] = []
        seen: set[str] = set()
        for raw in raw_results:
            if len(products) >= limit:
                break
            candidate = self._normalize_one(raw)
            if candidate is None:
                continue
            # Near-identical listings (same title from the same seller) crowd out
            # variety in a five-result list; keep the provider's highest-ranked one.
            listing_key = "listing:" + " ".join(candidate.title.lower().split()) + "|" + (
                (candidate.retailer or "").lower()
            )
            keys = (candidate.providerProductId, candidate.productUrl, listing_key)
            if any(key in seen for key in keys):
                continue
            seen.update(keys)
            products.append(candidate)
        return products

    def _normalize_one(self, raw: Any) -> ProductCandidate | None:
        if not isinstance(raw, dict):
            return None
        title = _text(raw.get("title"))
        price = _number(raw.get("extracted_price"))
        product_url = _http_url(raw.get("link")) or _http_url(raw.get("product_link"))
        if not title or price is None or price < 0 or not product_url:
            return None

        provider_id = _text(raw.get("product_id"))
        if not provider_id:
            # No provider id: key the candidate by a hash of its real URL rather than inventing one.
            provider_id = "url-" + hashlib.sha256(product_url.encode("utf-8")).hexdigest()[:16]

        price_text = _text(raw.get("price"))
        rating = _number(raw.get("rating"))
        reviews = _number(raw.get("reviews"))
        try:
            return ProductCandidate(
                id=f"{self.name}:{provider_id}",
                provider=self.name,
                providerProductId=provider_id,
                position=_int(raw.get("position")),
                title=title,
                price=price,
                priceText=price_text,
                currency=self._currency(price_text),
                retailer=_text(raw.get("source")),
                imageUrl=_http_url(raw.get("thumbnail")) or _http_url(raw.get("serpapi_thumbnail")),
                productUrl=product_url,
                rating=rating if rating is not None and 0 <= rating <= 5 else None,
                reviewCount=int(reviews) if reviews is not None and reviews >= 0 else None,
                detailPageToken=_text(raw.get("immersive_product_page_token")),
                # Google Shopping search results carry no structured physical
                # dimensions, so they are reported as unavailable, never estimated.
            )
        except ValidationError:
            return None

    def _currency(self, price_text: str | None) -> str | None:
        if not price_text:
            return None
        symbol = price_text.strip()[:1]
        if symbol == "€" and self._country in _EURO_COUNTRIES:
            return "EUR"
        return _CURRENCY_BY_COUNTRY_AND_SYMBOL.get((self._country, symbol))


def _collect_raw_results(payload: dict[str, Any]) -> list[Any] | None:
    results = payload.get("shopping_results")
    if isinstance(results, list):
        return results
    categorized = payload.get("categorized_shopping_results")
    if isinstance(categorized, list):
        merged: list[Any] = []
        for group in categorized:
            if isinstance(group, dict) and isinstance(group.get("shopping_results"), list):
                merged.extend(group["shopping_results"])
        return merged
    if results is None and categorized is None and isinstance(payload.get("search_metadata"), dict):
        # A well-formed response that simply has no shopping results.
        return []
    return None


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = " ".join(value.split())
    return stripped or None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _http_url(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return text
    return None

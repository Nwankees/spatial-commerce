from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .product_models import ProductCandidate
from .product_providers import (
    ProductProviderMalformedResponseError,
    ProductProviderNotConfiguredError,
    ProductProviderTimeoutError,
    ProductProviderUnavailableError,
)
from .serpapi_provider import SERPAPI_ENDPOINT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoreLink:
    name: str | None
    url: str


@dataclass(frozen=True)
class ProductDetail:
    """Provider product-detail data: explicit spec fields and retailer links."""

    features: list[tuple[str, str]] = field(default_factory=list)
    stores: list[StoreLink] = field(default_factory=list)
    source_name: str = "provider product details"


class ProductDetailSource(Protocol):
    name: str

    @property
    def configured(self) -> bool: ...

    async def fetch(self, candidate: ProductCandidate) -> ProductDetail | None:
        """Returns None when the candidate carries no detail reference."""


class SerpApiImmersiveProductDetailSource:
    """SerpApi Google Immersive Product API, keyed by the page token that the
    Google Shopping search result carried. Yields spec 'features' and store links."""

    name = "serpapi_immersive_product"

    def __init__(
        self,
        api_key: str | None,
        *,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._transport = transport

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    async def fetch(self, candidate: ProductCandidate) -> ProductDetail | None:
        if not candidate.detailPageToken:
            return None
        if not self._api_key:
            raise ProductProviderNotConfiguredError("SERPAPI_API_KEY is not configured.")
        params = {
            "engine": "google_immersive_product",
            "page_token": candidate.detailPageToken,
            "api_key": self._api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.get(SERPAPI_ENDPOINT, params=params)
        except httpx.TimeoutException:
            raise ProductProviderTimeoutError("The product details provider timed out.") from None
        except httpx.HTTPError as exc:
            logger.warning("SerpApi product details request failed: %s", type(exc).__name__)
            raise ProductProviderUnavailableError("The product details provider is unreachable.") from None
        if response.status_code != 200:
            raise ProductProviderUnavailableError(
                f"The product details provider returned HTTP {response.status_code}."
            )
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            raise ProductProviderMalformedResponseError("The product details provider returned invalid JSON.")
        if isinstance(payload.get("error"), str) and payload["error"]:
            raise ProductProviderUnavailableError("The product details provider rejected the request.")
        product = payload.get("product_results")
        if not isinstance(product, dict):
            raise ProductProviderMalformedResponseError("The product details response had no product_results.")
        return ProductDetail(
            features=_features(product.get("about_the_product")),
            stores=_stores(product.get("stores")),
            source_name="Google Shopping product details (via SerpApi)",
        )


def _features(about: Any) -> list[tuple[str, str]]:
    if not isinstance(about, dict) or not isinstance(about.get("features"), list):
        return []
    pairs = []
    for feature in about["features"]:
        if isinstance(feature, dict):
            title, value = feature.get("title"), feature.get("value")
            if isinstance(title, str) and isinstance(value, str):
                pairs.append((title, value))
    return pairs


def _stores(stores: Any) -> list[StoreLink]:
    if not isinstance(stores, list):
        return []
    links = []
    for store in stores:
        if isinstance(store, dict) and isinstance(store.get("link"), str):
            name = store.get("name") if isinstance(store.get("name"), str) else None
            links.append(StoreLink(name, store["link"]))
    return links

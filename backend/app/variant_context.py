"""Exact variant identity for a selected product, from SerpApi + retailer URL data only.

Nothing here looks at page order, dimensions, images or titles similarity, and no
LLM is involved. Identity comes from:
  1. the concrete store offer SerpApi lists for the selected result (same retailer,
     same price), and the retailer IDs in that offer's URL;
  2. explicitly selected options (SerpApi `variants[].items[].selected`, or an
     explicit "Color: X" / "Size: Y" in the offer title).
If identity cannot be pinned down, the context says so (identity = product_only).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import parse_qsl, unquote, urlparse

Identity = Literal["exact_item", "exact_offer", "options_only", "product_only"]

# Retailer-specific URL rules: only how to *read* stable IDs out of a store URL.
_PATH_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("target.com", re.compile(r"/-/A-(\d{5,})"), "itemId"),
    ("walmart.com", re.compile(r"/ip/(?:[^/]+/)?(\d{5,})"), "itemId"),
    ("wayfair.com", re.compile(r"~([A-Z0-9]{6,})\.html", re.I), "sku"),
    ("homedepot.com", re.compile(r"/p/(?:[^/]+/)?(\d{6,})"), "itemId"),
    ("kohls.com", re.compile(r"/prd-(\d{5,})"), "itemId"),
    ("staples.com", re.compile(r"/product_(\d{5,})"), "itemId"),
    ("bestbuy.com", re.compile(r"/sku/(\d{5,})"), "sku"),
    ("officedepot.com", re.compile(r"/products/(\d{4,})"), "sku"),
    ("quill.com", re.compile(r"/cbs/(\d{5,})\.html"), "itemId"),
    ("lowes.com", re.compile(r"/pd/[^/]+/(\d{5,})"), "itemId"),
    ("costco.com", re.compile(r"\.product\.(\d{5,})\.html"), "itemId"),
    ("amazon.com", re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})"), "sku"),
]
# Generic query parameters that carry item/variant/offer identity on many stores.
_QUERY_RULES: dict[str, str] = {
    "variant": "variantId", "variantid": "variantId", "vid": "variantId",
    "sku": "sku", "skuid": "sku", "sku_id": "sku",
    "itemid": "itemId", "item_id": "itemId", "pid": "itemId", "productid": "itemId",
    "preselect": "variantId", "selectedofferid": "offerId", "offerid": "offerId",
    "selectedsellerid": "sellerId", "piid": "optionIds", "piid[]": "optionIds",
}
_EXPLICIT_OPTION_RE = re.compile(r"\b(color|colour|size|finish|style|material|pattern)\s*:\s*([^,|()\[\]]{1,40})", re.I)


@dataclass
class RetailerIds:
    itemId: str | None = None
    sku: str | None = None
    variantId: str | None = None
    offerId: str | None = None
    sellerId: str | None = None
    optionIds: list[str] = field(default_factory=list)

    def item_level(self) -> list[str]:
        """IDs that identify one concrete sellable item/variant."""
        return [v for v in (self.variantId, self.itemId, self.sku) if v]

    def all(self) -> list[str]:
        return self.item_level() + [v for v in (self.offerId,) if v] + list(self.optionIds)


@dataclass
class OfferRef:
    storeName: str | None
    storeTitle: str | None
    storeUrl: str
    price: float | None


@dataclass
class VariantContext:
    googleIds: dict[str, str] = field(default_factory=dict)
    offer: OfferRef | None = None
    retailerDomain: str | None = None
    retailerIds: RetailerIds = field(default_factory=RetailerIds)
    selectedOptions: dict[str, str] = field(default_factory=dict)
    optionSources: dict[str, str] = field(default_factory=dict)
    identity: Identity = "product_only"
    reason: str | None = None

    @property
    def exact(self) -> bool:
        return self.identity in ("exact_item", "exact_offer")

    def describe(self) -> str:
        ids = ", ".join(f"{k}={v}" for k, v in vars(self.retailerIds).items() if v)
        return f"{self.identity} ({self.retailerDomain or 'no retailer'}{': ' + ids if ids else ''})"


def parse_google_ids(product_link: str | None) -> dict[str, str]:
    """catalogid / productid / headlineOfferDocid / gpcid / mid from a Google Shopping link."""
    if not product_link:
        return {}
    query = dict(parse_qsl(urlparse(product_link).query))
    ids = {}
    for part in (query.get("prds") or "").split(","):
        key, _, value = part.partition(":")
        if key in ("catalogid", "productid", "headlineOfferDocid", "gpcid", "mid") and value:
            ids[key] = value
    return ids


def extract_retailer_ids(url: str | None) -> tuple[str | None, RetailerIds]:
    ids = RetailerIds()
    if not url:
        return None, ids
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    domain = host[4:] if host.startswith("www.") else host
    path = unquote(parsed.path)
    for suffix, pattern, slot in _PATH_RULES:
        if domain == suffix or domain.endswith("." + suffix):
            match = pattern.search(path)
            if match and getattr(ids, slot) is None:
                setattr(ids, slot, match.group(1))
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        slot = _QUERY_RULES.get(key.lower())
        if not slot or not re.fullmatch(r"[A-Za-z0-9_-]{2,64}", value):
            continue
        if slot == "optionIds":
            ids.optionIds.append(value)
        elif getattr(ids, slot) is None:
            setattr(ids, slot, value)
    return domain or None, ids


def choose_offer(stores: list, retailer: str | None, price: float | None) -> tuple[OfferRef | None, str | None]:
    """The stores[] entry that is the selected result's concrete offer: same retailer
    name and same price. Returns (offer, reason-if-none)."""
    wanted = _norm(retailer)
    if not wanted:
        return None, "selected result has no retailer name"
    same_store = [s for s in stores if _norm(getattr(s, "name", None)) == wanted]
    if not same_store:
        return None, "no product-detail offer from the selected retailer"
    if len(same_store) > 1 and price is not None:
        same_store = [s for s in same_store if getattr(s, "price", None) is not None and abs(s.price - price) < 0.005]
    if len(same_store) != 1:
        return None, "several offers from the selected retailer; cannot tell which was selected"
    s = same_store[0]
    return OfferRef(s.name, getattr(s, "title", None), s.url, getattr(s, "price", None)), None


def build_variant_context(
    product_link: str | None,
    retailer: str | None,
    price: float | None,
    stores: list,
    serpapi_selected_options: dict[str, str] | None = None,
    direct_url: str | None = None,
) -> VariantContext:
    context = VariantContext(googleIds=parse_google_ids(product_link))
    offer, reason = (None, None)
    if direct_url:
        offer = OfferRef(retailer, None, direct_url, price)
    else:
        offer, reason = choose_offer(stores, retailer, price)
    context.offer = offer
    if offer:
        context.retailerDomain, context.retailerIds = extract_retailer_ids(offer.storeUrl)
        for match in _EXPLICIT_OPTION_RE.finditer(offer.storeTitle or ""):
            name, value = match.group(1).lower(), match.group(2).strip()
            context.selectedOptions.setdefault(name, value)
            context.optionSources.setdefault(name, "store_title")
    for name, value in (serpapi_selected_options or {}).items():
        context.selectedOptions.setdefault(name.lower(), value)
        context.optionSources.setdefault(name.lower(), "serpapi_selected")

    if context.retailerIds.item_level():
        context.identity = "exact_item"
    elif context.retailerIds.offerId:
        context.identity = "exact_offer"
    elif context.selectedOptions:
        context.identity = "options_only"
        context.reason = reason or "no retailer item ID in the offer URL"
    else:
        context.identity = "product_only"
        context.reason = reason or "no retailer item ID or selected options available"
    return context


def _norm(text: str | None) -> str:
    return " ".join((text or "").lower().split())

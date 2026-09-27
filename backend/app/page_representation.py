"""Generic, structure-preserving representation of a retailer product page.

The page is parsed into a compact JSON-able structure (product title, selected
variant, JSON-LD product data, spec tables / name-value lists, headed sections,
embedded application-state data and visible text blocks) that the dimension
model reads. Only *noise* is removed (navigation, footers, cookie banners, ads,
recommendations, reviews, Q&A, AI-generated summaries, media, tracking). Nothing
is kept or dropped because it does or does not look like a measurement: that
judgement belongs to the model, and provenance is checked afterwards in code.

Every entry has a stable id (E12 / T3) and a human-readable path so the model's
choice can be verified against exactly this representation.
"""
from __future__ import annotations

import html as html_lib
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Iterable, Literal

from .dimension_evidence import AI_GENERATED_RE
from .variant_context import VariantContext, extract_retailer_ids

EntryKind = Literal["json_ld", "spec_table", "embedded_json", "page_text", "provider_specs"]
Scope = Literal["exact_record", "exact_page", "family", "page"]

MAX_HTML_CHARS = 4_000_000
MAX_SCRIPT_CHARS = 8_000_000
DEFAULT_MAX_CHARS = 12_000
_MAX_VALUE_CHARS = 400
_MAX_TEXT_BLOCK = 400
_MAX_ENTRIES = 1500

# ---- Noise (not semantics) ----------------------------------------------
_DROP_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "footer", "aside", "iframe", "select",
              "option", "textarea", "input", "img", "picture", "video", "audio", "canvas", "map", "object", "head"}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
              "track", "wbr"}
_NOISE_ATTR_RE = re.compile(
    r"review|rating|comment|question|answer|\bqna\b|q-and-a|\bfaq\b|cookie|consent|gdpr|breadcrumb|footer|"
    r"\bnav\b|navbar|nav-|-nav|\bmenu\b|megamenu|recommend|carousel|similar|related|also-?viewed|also-?bought|"
    r"sponsor|advert|\bads?\b|ads-|-ads|promo|newsletter|sign-?up|social|share|chat-?widget|skip-?link",
    re.IGNORECASE,
)
# Sections that are user content or other products, recognised by their heading.
_NOISE_HEADING_RE = re.compile(
    r"review|rating|questions?\b|q\s*&\s*a|customers (?:also|who)|you may also|you might also|similar (?:items|products)|"
    r"recommended|sponsored|related (?:items|products)|frequently bought|compare with|recently viewed",
    re.IGNORECASE,
)
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "summary", "caption", "legend"}
_BLOCK_TAGS = {"p", "div", "li", "ul", "ol", "section", "article", "main", "table", "tr", "td", "th", "dl", "dt",
               "dd", "br", "h1", "h2", "h3", "h4", "h5", "h6", "summary", "details", "header", "form", "fieldset",
               "blockquote", "pre", "figure", "figcaption", "caption", "button", "span_block"}
# Embedded-data keys whose subtrees are user content, media, tracking, other products or
# commerce/app state (prices, promos, fulfilment, config). Matched on whole words of the key
# (camelCase / snake_case split), so e.g. "description" or "catalog" are not caught.
_NOISE_KEY_WORDS = {
    "review", "reviews", "rating", "ratings", "question", "questions", "answer", "answers", "comment", "comments",
    "image", "images", "img", "media", "gallery", "video", "videos", "thumbnail", "thumbnails", "swatch", "swatches",
    "icon", "icons", "logo", "logos", "recommendation", "recommendations", "recommended", "similar", "related",
    "sponsor", "sponsored", "advert", "adverts", "ad", "ads", "tracking", "analytics", "beacon", "pixel",
    "experiment", "experiments", "flag", "flags", "breadcrumb", "breadcrumbs", "seo", "footer", "header",
    "navigation", "nav", "cookie", "cookies", "consent", "i18n", "translation", "translations", "locale", "css",
    "script", "scripts", "token", "tokens", "csrf", "session", "cart", "checkout", "login", "account", "url", "urls",
    "href", "link", "links", "src", "srcset", "badge", "badges", "promo", "promos", "promotion", "promotions",
    "discount", "discounts", "price", "prices", "pricing", "coupon", "coupons", "rebate", "rebates", "return",
    "returns", "warranty", "protection", "addon", "addons", "fulfillment", "fulfilment", "delivery", "pickup",
    "inventory", "availability", "seller", "sellers", "payment", "payments", "financing",
    "subscription", "subscriptions", "config", "configs", "configuration", "telemetry", "metric", "metrics", "log",
    "logs", "logging", "debug", "toast", "banner", "banners",
    "targeting", "sampling", "traceparent", "traceparents", "cacheable", "oidc", "redirection",
    "dynamic", "locales", "params", "membership", "telemetry",
}
# (Generic container words such as module/layout/zone/slot/store/state are deliberately
# not noise: application state often nests the product record inside them.)


def _key_words(key: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", key)]


def _is_noise_key(key: str) -> bool:
    words = _key_words(key)
    joined = "".join(words)
    return (any(w in _NOISE_KEY_WORDS for w in words) or "addon" in joined
            or bool(AI_GENERATED_RE.search(key)))


_ITEM_ID_KEYS = {
    "tcin", "usitemid", "itemid", "item_id", "sku", "skuid", "sku_id", "variantid", "variant_id",
    "productid", "product_id", "offerid", "offer_id", "partnumber", "itemnumber", "catentryid",
}
_ID_KEYS = _ITEM_ID_KEYS | {"id"}
_NAME_KEYS = ("name", "label", "displayName", "display_name", "title", "attributeName", "key", "specName")
_VALUE_KEYS = ("value", "values", "displayValue", "display_value", "attributeValue", "attributeValues", "specValue")
_UNIT_KEYS = ("unit", "unitText", "unitCode", "uom", "unitOfMeasure", "unit_of_measure", "dimension_unit_of_measure",
              "measurementUnit", "units")
_UNIT_CODES = {"INH": "in", "CMT": "cm", "MMT": "mm", "MTR": "m", "FOT": "ft", "LBR": "lb", "KGM": "kg", "GRM": "g"}
_PRODUCT_TYPES = {"product", "productgroup", "individualproduct", "productmodel"}
_JSONLD_KEYS = ("name", "sku", "productID", "mpn", "gtin", "gtin12", "gtin13", "gtin14", "model", "color", "size",
                "material", "pattern", "width", "depth", "height", "length", "weight", "additionalProperty",
                "description", "category", "brand")


@dataclass
class Entry:
    id: str
    name: str | None
    value: str
    kind: EntryKind
    path: str
    scope: Scope
    owner: str | None = None

    @property
    def text(self) -> str:
        return f"{self.name}: {self.value}" if self.name else self.value


@dataclass
class Section:
    id: str
    heading: str
    kind: EntryKind
    scope: Scope
    priority: int
    owner: str | None = None
    entries: list[Entry] = field(default_factory=list)
    text: list[Entry] = field(default_factory=list)


@dataclass
class PageRepresentation:
    url: str | None
    retailer: str | None
    product_title: str | None
    page_title: str | None = None
    page_product_title: str | None = None
    selected_variant: dict[str, Any] = field(default_factory=dict)
    page_ids: list[str] = field(default_factory=list)
    variant_mismatch: bool = False
    sections: list[Section] = field(default_factory=list)
    included_ids: set[str] = field(default_factory=set)
    truncated: bool = False

    # -- lookup ---------------------------------------------------------------
    def entry(self, entry_id: str) -> tuple[Entry, Section] | None:
        for section in self.sections:
            for entry in section.entries + section.text:
                if entry.id == entry_id:
                    return entry, section
        return None

    def all_entries(self) -> Iterable[tuple[Entry, Section]]:
        for section in self.sections:
            for entry in section.entries + section.text:
                yield entry, section

    def owners(self) -> set[str]:
        return {s.owner for s in self.sections if s.owner}

    @property
    def is_empty(self) -> bool:
        return not any(s.entries or s.text for s in self.sections)

    # -- LLM view -------------------------------------------------------------
    def to_llm_dict(self, max_chars: int = DEFAULT_MAX_CHARS) -> dict[str, Any]:
        """Compact dict for the model. Sections are added by structural priority
        (structured data first, free text last) until the budget is used; the ids of
        what was included are remembered so validation only accepts shown entries."""
        head: dict[str, Any] = {
            "productTitle": self.product_title,
            "pageProductTitle": self.page_product_title,
            "pageTitle": self.page_title,
            "retailer": self.retailer,
            "selectedVariant": self.selected_variant or None,
        }
        head = {k: v for k, v in head.items() if v}
        used = len(_dumps(head)) + 40
        out_sections: list[dict[str, Any]] = []
        included: set[str] = set()
        truncated = False
        for section in sorted(self.sections, key=lambda s: s.priority):
            shown: dict[str, Any] = {"id": section.id, "heading": section.heading, "type": section.kind}
            if section.scope == "exact_record":
                shown["scope"] = "selected item"
            elif section.scope == "family":
                shown["scope"] = "product family (all variants)"
            if section.owner:
                shown["itemId"] = section.owner
            base = len(_dumps(shown)) + 20
            if used + base > max_chars:
                truncated = True
                break
            entries, texts = [], []
            section_used = base
            for entry in section.entries:
                item = {"id": entry.id, "name": entry.name, "value": entry.value}
                size = len(_dumps(item)) + 1
                if used + section_used + size > max_chars:
                    truncated = True
                    break
                entries.append(item)
                included.add(entry.id)
                section_used += size
            for entry in section.text:
                item = {"id": entry.id, "text": entry.value}
                size = len(_dumps(item)) + 1
                if used + section_used + size > max_chars:
                    truncated = True
                    break
                texts.append(item)
                included.add(entry.id)
                section_used += size
            if not entries and not texts:
                continue
            if entries:
                shown["entries"] = entries
            if texts:
                shown["text"] = texts
            out_sections.append(shown)
            used += section_used
        self.included_ids = included
        self.truncated = truncated
        head["sections"] = out_sections
        return head

    def to_llm_json(self, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        return _dumps(self.to_llm_dict(max_chars))


def build_text_representation(
    text: str,
    url: str | None,
    *,
    product_title: str | None,
    retailer: str | None,
    heading: str = "Rendered product content",
) -> PageRepresentation:
    """Wrap rendered markdown/search evidence in the same provenance model as HTML.

    Crawl4AI and Browser Use deliberately live in an isolated helper process. They
    return compact rendered text, so this adapter gives every block a stable source
    id before Qwen sees it. Validation can then prove each returned number and unit
    came from the cited block instead of trusting model prose.
    """
    rep = PageRepresentation(url=url, retailer=retailer, product_title=product_title)
    section = Section(id="S1", heading=heading, kind="page_text", scope="page", priority=0)
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    # One rendered/markdown line per evidence id. This keeps "Width: ..." and
    # "Length: ..." semantics local instead of blending neighboring labels.
    blocks = [chunk for line in cleaned.splitlines() for chunk in _chunks(line, _MAX_TEXT_BLOCK)]
    for index, block in enumerate(blocks[:80], start=1):
        entry = Entry(
            id=f"T{index}",
            name=None,
            value=block,
            kind="page_text",
            path=f"{heading} > block {index}",
            scope="page",
        )
        section.text.append(entry)
    if section.text:
        rep.sections.append(section)
    return rep


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ---- Builder -----------------------------------------------------------------

class _Builder:
    def __init__(self, rep: PageRepresentation, exact_ids: set[str], exact: bool, page_scope: Scope) -> None:
        self.rep = rep
        self.exact_ids = exact_ids
        self.exact = exact
        self.page_scope = page_scope
        self._entry_counter = 0
        self._text_counter = 0
        self._section_counter = 0
        self._seen: set[tuple[str, str, str, str]] = set()
        self._entries = 0
        self.found_exact_record = False

    def section(self, heading: str, kind: EntryKind, scope: Scope, priority: int, owner: str | None = None) -> Section:
        self._section_counter += 1
        section = Section(f"S{self._section_counter}", _squash(heading)[:120] or "(untitled)", kind, scope, priority, owner)
        self.rep.sections.append(section)
        return section

    def add_entry(self, section: Section, name: str | None, value: str, path: str) -> None:
        name = _squash(name or "")[:120] or None
        value = _squash(value)
        if not value or self._entries >= _MAX_ENTRIES:
            return
        key = (section.scope, section.owner or "", (name or "").lower(), value.lower())
        if key in self._seen:
            return  # the same name/value is often repeated (highlights, spec lists, v2 lists)
        self._seen.add(key)
        for chunk in _chunks(value, _MAX_VALUE_CHARS):
            self._entry_counter += 1
            self._entries += 1
            section.entries.append(Entry(f"E{self._entry_counter}", name, chunk, section.kind,
                                         f"{section.heading} > {name}" if name else f"{section.heading} > {path}",
                                         section.scope, section.owner))

    def add_text(self, section: Section, text: str) -> None:
        text = _squash(text)
        if len(text) < 2 or self._entries >= _MAX_ENTRIES:
            return
        key = (section.scope, section.owner or "", "", text.lower())
        if key in self._seen:
            return
        self._seen.add(key)
        for chunk in _chunks(text, _MAX_TEXT_BLOCK):
            self._text_counter += 1
            self._entries += 1
            section.text.append(Entry(f"T{self._text_counter}", None, chunk, "page_text",
                                      f"{section.heading} > text", section.scope, section.owner))


def build_page_representation(
    page_html: str,
    url: str | None,
    *,
    product_title: str | None = None,
    retailer: str | None = None,
    context: VariantContext | None = None,
    exact_ids: Iterable[str] = (),
) -> PageRepresentation:
    """Parses one retailer page. ``exact_ids`` are the selected item's retailer ids
    (only when identity is exact): objects belonging to other item ids (sibling
    variants) are excluded, and the selected item's own record is scoped as such."""
    ids = {str(i) for i in exact_ids if i}
    exact = bool(ids)
    rep = PageRepresentation(url=url, retailer=retailer, product_title=product_title)
    if context is not None:
        rep.selected_variant = {k: v for k, v in {
            "identity": context.identity,
            "retailerIds": {k: v for k, v in vars(context.retailerIds).items() if v} or None,
            "selectedOptions": context.selectedOptions or None,
            "offerTitle": context.offer.storeTitle if context.offer else None,
        }.items() if v}
    page_scope: Scope = "exact_page" if exact else "page"
    builder = _Builder(rep, ids, exact, page_scope)

    parser = _TreeParser()
    try:
        parser.feed(page_html[:MAX_HTML_CHARS])
        parser.close()
    except Exception:
        pass  # html.parser is tolerant; use whatever was parsed
    rep.page_title = _squash(parser.title)[:200] or None

    url_ids: set[str] = set()
    if url:
        url_ids.update(extract_retailer_ids(url)[1].item_level())
    declared_ids: set[str] = set()  # the page's own statement of which item it shows
    for declared_url in (parser.canonical, parser.og_url):
        if declared_url:
            declared_ids.update(extract_retailer_ids(declared_url)[1].item_level())

    for block in parser.jsonld:
        _add_jsonld(builder, _load_json(block), declared_ids)
    for body in parser.json_scripts:
        data = _parse_embedded(body)
        if data is not None:
            _add_embedded(builder, data)
    _add_html(builder, parser.root)

    # Sections that declare themselves machine-generated (e.g. a trailing "Generated by AI"
    # line under a feature summary) are not the retailer's own product data.
    for section in rep.sections:
        if section.kind == "page_text" and any(AI_GENERATED_RE.search(t.value) and len(t.value) <= 60 for t in section.text):
            section.text.clear()
            section.entries.clear()
    rep.page_ids = sorted(url_ids | declared_ids)
    if exact:
        # The fetched (final) URL names another item, or the page carries no record of the
        # selected item and declares itself (canonical / JSON-LD) to be a different one.
        redirected_away = bool(url_ids) and not (url_ids & ids)
        declared_other = bool(declared_ids) and not (declared_ids & ids) and not builder.found_exact_record
        rep.variant_mismatch = redirected_away or declared_other
    rep.sections = [s for s in rep.sections if s.entries or s.text]
    return rep


def build_provider_representation(
    features: list[tuple[str, str]],
    *,
    product_title: str | None,
    retailer: str | None,
    source_name: str | None,
) -> PageRepresentation:
    """SerpApi product-detail specs (product-family scope) in the same shape."""
    rep = PageRepresentation(url=None, retailer=retailer, product_title=product_title)
    builder = _Builder(rep, set(), False, "family")
    section = builder.section(f"{source_name or 'Provider'} specifications", "provider_specs", "family", 0)
    for name, value in features:
        builder.add_entry(section, name, value, name)
    rep.sections = [s for s in rep.sections if s.entries]
    return rep


# ---- JSON-LD -------------------------------------------------------------------

def _add_jsonld(builder: _Builder, data: Any, declared_ids: set[str]) -> None:
    for node, is_variant in _jsonld_products(data):
        owner = next((str(node[k]) for k in ("sku", "productID") if isinstance(node.get(k), (str, int))), None)
        if not is_variant and owner:
            declared_ids.add(owner)
        if builder.exact and owner in builder.exact_ids:
            builder.found_exact_record = True
        if is_variant:
            if builder.exact and owner and owner not in builder.exact_ids:
                continue  # a sibling variant listed under hasVariant
            scope: Scope = "exact_record" if builder.exact and owner in builder.exact_ids else builder.page_scope
        else:
            scope = "exact_record" if builder.exact and owner in builder.exact_ids else builder.page_scope
        if not is_variant and isinstance(node.get("name"), str) and not builder.rep.page_product_title:
            builder.rep.page_product_title = _squash(node["name"])[:200]
        section = builder.section("JSON-LD " + ("variant" if is_variant else "product"), "json_ld", scope, 0,
                                  owner if is_variant else None)
        for key in _JSONLD_KEYS:
            if key not in node:
                continue
            value = node[key]
            if key == "additionalProperty":
                for prop in value if isinstance(value, list) else [value]:
                    if isinstance(prop, dict) and isinstance(prop.get("name"), str):
                        text = _quantity_text(prop)
                        if text:
                            builder.add_entry(section, prop["name"], text, f"additionalProperty.{prop['name']}")
                continue
            if key == "brand" and isinstance(value, dict):
                value = value.get("name")
            if key == "description" and isinstance(value, str):
                value = _strip_tags(value)
            text = _quantity_text(value) if isinstance(value, dict) else (
                str(value) if isinstance(value, (str, int, float)) and not isinstance(value, bool) else None)
            if text:
                builder.add_entry(section, key, text, key)


def _jsonld_products(data: Any, depth: int = 0, variant: bool = False) -> Iterable[tuple[dict, bool]]:
    if depth > 12:
        return
    if isinstance(data, list):
        for item in data:
            yield from _jsonld_products(item, depth + 1, variant)
    elif isinstance(data, dict):
        types = data.get("@type")
        names = {str(t).lower() for t in (types if isinstance(types, list) else [types]) if t}
        if names & _PRODUCT_TYPES:
            yield data, variant
        for key in ("@graph", "mainEntity", "itemListElement", "item"):
            if key in data:
                yield from _jsonld_products(data[key], depth + 1, variant)
        if "hasVariant" in data:
            yield from _jsonld_products(data["hasVariant"], depth + 1, True)


def _quantity_text(value: Any) -> str | None:
    """schema.org QuantitativeValue / PropertyValue -> '30 in', keeping the literal number."""
    if isinstance(value, dict):
        number = value.get("value")
        if isinstance(number, (dict, list)) or number is None or isinstance(number, bool):
            return None
        unit = value.get("unitText") or value.get("unitCode")
        unit_text = _UNIT_CODES.get(str(unit).upper(), str(unit)) if unit else ""
        return f"{number} {unit_text}".strip()
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    return None


# ---- Embedded application data -------------------------------------------------

def _parse_embedded(body: str) -> Any:
    body = body.strip()
    if not body or len(body) > MAX_SCRIPT_CHARS:
        return None
    candidates = [body]
    assignment = re.match(r"^(?:window\.|var\s+|let\s+|const\s+)?[\w.$\[\]'\"]+\s*=\s*(.*?);?\s*$", body, re.S)
    if assignment:
        candidates.append(assignment.group(1))
    for text in candidates:
        if text[:1] in "{[":
            try:
                return json.loads(text)
            except (ValueError, RecursionError):
                continue
    return None


def _node_ids(node: dict) -> list[tuple[str, str]]:
    return [(k, str(v)) for k, v in node.items()
            if k.lower() in _ID_KEYS and isinstance(v, (str, int)) and not isinstance(v, bool)]


def _is_item_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{4,}", value))


def _add_embedded(builder: _Builder, data: Any) -> None:
    groups: dict[tuple[str, str | None, Scope], Section] = {}

    def group(path: str, owner: str | None, scope: Scope, flat: bool) -> Section:
        key = (path, owner, scope)
        if key not in groups:
            heading = _path_heading(path)
            # Name/value lists are spec-like by structure. Within that broad class,
            # generic product/spec/detail containers precede unrelated application
            # state. This is structural prioritization, not measurement matching:
            # no field is included/excluded because its value looks like dimensions.
            product_structure = bool(re.search(
                r"(?:^|[.>_])(product|specifications?|specs?|details?|attributes?|highlights?|features?|description)(?:$|[.>_])",
                path,
                re.IGNORECASE,
            ))
            if not flat:
                priority = 0 if product_structure else 2
            elif scope == "exact_record":
                priority = 1
            else:
                priority = 2 if product_structure else 4
            if scope == "family":
                priority += 1
            groups[key] = builder.section(heading, "embedded_json", scope, priority, owner)
        return groups[key]

    def walk(node: Any, path: str, owner: str | None, scope: Scope, depth: int) -> None:
        if depth > 45 or builder._entries >= _MAX_ENTRIES:
            return
        if isinstance(node, list):
            for child in node:
                walk(child, path + "[]", owner, scope, depth + 1)
            return
        if not isinstance(node, dict):
            return
        ids = _node_ids(node)
        item_ids = [v for k, v in ids if k.lower() in _ITEM_ID_KEYS]
        own_scope = scope
        if builder.exact:
            if any(v in builder.exact_ids for _, v in ids):
                scope = own_scope = "exact_record"
                owner = next(v for _, v in ids if v in builder.exact_ids)
                builder.found_exact_record = True
            elif any(_is_item_id(v) and v not in builder.exact_ids for v in item_ids):
                if not _contains_id(node, builder.exact_ids):
                    return  # another item's data (sibling variant, related product)
                # a parent/family object that contains the selected item: its own fields
                # describe the product family; its children are walked normally
                own_scope = "family"
        elif item_ids:
            owner = item_ids[0]
        # name/value pair objects: {"name": "Dimensions", "value": "..."}
        name = next((node[k] for k in _NAME_KEYS if isinstance(node.get(k), str)), None)
        value = next((node[k] for k in _VALUE_KEYS if k in node), None)
        if isinstance(value, list) and value and all(isinstance(v, (str, int, float)) for v in value):
            value = ", ".join(str(v) for v in value)
        if isinstance(name, str) and isinstance(value, (str, int, float)) and not isinstance(value, bool):
            if not AI_GENERATED_RE.search(name) and not _is_blob(str(value)):
                unit = next((node[k] for k in _UNIT_KEYS if isinstance(node.get(k), str)), None)
                text = _clean(str(value))
                if unit and unit.strip().lower() not in text.lower():
                    text = f"{text} {_UNIT_CODES.get(unit.upper(), unit)}"
                builder.add_entry(group(path, owner, own_scope, False), _clean(name), text, path)
            return
        # small flat objects: {"width": 21.65, "depth": 22.44, "dimension_unit_of_measure": "INCH"}
        leaves = {k: v for k, v in node.items() if _informative_leaf(k, v)}
        nested = {k: v for k, v in node.items() if isinstance(v, (dict, list))}
        if leaves and len(leaves) <= 14 and not nested and any(isinstance(v, (int, float)) for v in leaves.values()):
            text = ", ".join(f"{_humanize(k)}: {_leaf_text(v)}" for k, v in leaves.items())
            builder.add_entry(group(path, owner, own_scope, True), _humanize(path.rsplit(".", 1)[-1].rstrip("[]")), text, path)
        else:
            for key, value in leaves.items():
                if isinstance(value, str) and len(value) > 3 or isinstance(value, (int, float)):
                    builder.add_entry(group(path, owner, own_scope, True), _humanize(key), _leaf_text(value), f"{path}.{key}")
        for key, child in nested.items():
            if _is_noise_key(key):
                continue
            if isinstance(child, list) and child and all(isinstance(v, (str, int, float)) and not isinstance(v, bool)
                                                         for v in child):
                # lists of plain values, e.g. bullet points / feature lines
                section = group(f"{path}.{key}", owner, own_scope, False)
                for value in child[:200]:
                    text = _leaf_text(value)
                    if text and not _is_blob(text) and not AI_GENERATED_RE.search(text[:200]):
                        builder.add_entry(section, _humanize(key), text, f"{path}.{key}")
                continue
            # below a family/parent object, everything not in the selected item's own record
            # is family-level data
            walk(child, f"{path}.{key}", owner, "family" if own_scope == "family" else scope, depth + 1)

    walk(data, "$", None, builder.page_scope, 0)


def _is_blob(text: str) -> bool:
    """Serialized JSON/markup inside a string value (machine state, not page content)."""
    t = text.strip()
    return len(t) > 80 and t[:1] in "{[<" and t[-1:] in "}]>"


def _contains_id(node: Any, wanted: set[str], depth: int = 0) -> bool:
    if depth > 45:
        return False
    if isinstance(node, dict):
        if any(v in wanted for _, v in _node_ids(node)):
            return True
        return any(_contains_id(v, wanted, depth + 1) for v in node.values() if isinstance(v, (dict, list)))
    if isinstance(node, list):
        return any(_contains_id(v, wanted, depth + 1) for v in node if isinstance(v, (dict, list)))
    return False


def _informative_leaf(key: str, value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
        return False
    if _is_noise_key(key) or key.startswith(("__", "@")):
        return False
    if key.lower() in _ID_KEYS or key.lower().endswith(("id", "ids", "_id", "uuid", "hash", "key")):
        return False
    if isinstance(value, str):
        v = value.strip()
        if not v or v.startswith(("http://", "https://", "//", "/", "data:", "#")):
            return False
        if re.fullmatch(r"[0-9a-f-]{16,}|[A-Za-z0-9+/=_-]{32,}", v):
            return False  # hashes, tokens, opaque ids
        if AI_GENERATED_RE.search(v[:200]) or _is_blob(v):
            return False
    return True


def _leaf_text(value: Any) -> str:
    return _clean(str(value)) if isinstance(value, str) else str(value)


def _path_heading(path: str) -> str:
    parts = [p.rstrip("[]") for p in path.split(".") if p and p != "$"]
    parts = [p for p in parts if p not in ("props", "pageProps", "initialData", "data", "__NEXT_DATA__")]
    return "embedded: " + (" > ".join(parts[-3:]) if parts else "root")


def _humanize(key: str) -> str:
    key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", key).replace("_", " ")
    return _squash(key)


# ---- HTML ------------------------------------------------------------------------

class _Node:
    __slots__ = ("tag", "attrs", "children", "parent")

    def __init__(self, tag: str, attrs: dict[str, str], parent: _Node | None) -> None:
        self.tag, self.attrs, self.parent = tag, attrs, parent
        self.children: list[_Node | str] = []


class _TreeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("root", {}, None)
        self._stack = [self.root]
        self.jsonld: list[str] = []
        self.json_scripts: list[str] = []
        self.title = ""
        self.canonical: str | None = None
        self.og_url: str | None = None
        self._script: str | None = None
        self._buf: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        attributes = {k.lower(): (v or "") for k, v in attrs}
        if tag == "script":
            kind = attributes.get("type", "").lower()
            self._script = "jsonld" if "ld+json" in kind else ("json" if (not kind or "json" in kind or "javascript" in kind) else None)
            self._buf = []
            return
        if tag == "link" and "canonical" in attributes.get("rel", "").lower():
            self.canonical = attributes.get("href") or None
        if tag == "meta" and attributes.get("property", "").lower() == "og:url":
            self.og_url = attributes.get("content") or None
        if tag == "title":
            self._in_title = True
        node = _Node(tag, attributes, self._stack[-1])
        self._stack[-1].children.append(node)
        if tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS and self._stack[-1].tag == tag:
            self._stack.pop()

    def handle_endtag(self, tag):
        if tag == "script":
            if self._script == "jsonld":
                self.jsonld.append("".join(self._buf))
            elif self._script == "json":
                self.json_scripts.append("".join(self._buf))
            self._script = None
            return
        if tag == "title":
            self._in_title = False
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                break

    def handle_data(self, data):
        if self._script is not None:
            self._buf.append(data)
            return
        if self._in_title:
            self.title += data
            return
        self._stack[-1].children.append(data)


def _is_noise(node: _Node) -> bool:
    if node.tag in _DROP_TAGS or node.tag == "title":
        return True
    if node.attrs.get("aria-hidden") == "true" or "hidden" in node.attrs:
        return True
    marker = " ".join(node.attrs.get(k, "") for k in ("id", "class", "role", "aria-label", "data-testid", "data-test"))
    return bool(marker.strip()) and bool(_NOISE_ATTR_RE.search(marker) or AI_GENERATED_RE.search(marker))


def _node_text(node: _Node) -> str:
    parts: list[str] = []

    def collect(n: _Node | str) -> None:
        if isinstance(n, str):
            parts.append(n)
        elif not _is_noise(n):
            if n.tag in _BLOCK_TAGS:
                parts.append(" ")
            for child in n.children:
                collect(child)
            if n.tag in _BLOCK_TAGS:
                parts.append(" ")

    collect(node)
    return _squash("".join(parts))


def _is_heading(node: _Node) -> bool:
    if node.tag in _HEADING_TAGS or node.attrs.get("role") == "heading":
        return True
    if node.tag == "button" and ("aria-expanded" in node.attrs or "aria-controls" in node.attrs):
        return True  # accordion toggles ("Specifications", "Details")
    return False


def _add_html(builder: _Builder, root: _Node) -> None:
    state = {"section": builder.section("Page", "page_text", builder.page_scope, 3), "headings": []}
    inline: list[str] = []

    def flush() -> None:
        text = _squash("".join(inline))
        inline.clear()
        if text and not state.get("noise"):
            builder.add_text(state["section"], text)

    def start_section(title: str) -> None:
        flush()
        state["section"] = builder.section(title, "page_text", builder.page_scope, 3)
        state["noise"] = bool(_NOISE_HEADING_RE.search(title))

    def visit(node: _Node | str) -> None:
        if isinstance(node, str):
            inline.append(node)
            return
        if _is_noise(node):
            return
        if node.tag == "h1" and not builder.rep.page_product_title:
            builder.rep.page_product_title = _node_text(node)[:200] or None
        if _is_heading(node):
            title = _node_text(node)
            if title and len(title) <= 120:
                start_section(title)
                return
        if state.get("noise") and node.tag in ("table", "dl"):
            return
        if node.tag == "table":
            flush()
            _add_table(builder, node, state["section"].heading)
            return
        if node.tag == "dl":
            flush()
            _add_dl(builder, node, state["section"].heading)
            return
        block = node.tag in _BLOCK_TAGS
        if block:
            flush()
        for child in node.children:
            visit(child)
        if block:
            flush()

    visit(root)
    flush()


def _add_table(builder: _Builder, table: _Node, heading: str) -> None:
    caption = next((c for c in table.children if isinstance(c, _Node) and c.tag == "caption"), None)
    section = builder.section(_node_text(caption) if caption else heading, "spec_table", builder.page_scope, 1)
    rows: list[_Node] = []

    def find_rows(node: _Node) -> None:
        for child in node.children:
            if isinstance(child, _Node) and not _is_noise(child):
                if child.tag == "tr":
                    rows.append(child)
                elif child.tag != "table":
                    find_rows(child)

    find_rows(table)
    for row in rows:
        cells = [_node_text(c) for c in row.children if isinstance(c, _Node) and c.tag in ("td", "th")]
        cells = [c for c in cells if c]
        if len(cells) >= 2:
            builder.add_entry(section, cells[0], " | ".join(cells[1:]), cells[0])
        elif len(cells) == 1:
            builder.add_text(section, cells[0])


def _add_dl(builder: _Builder, dl: _Node, heading: str) -> None:
    section = builder.section(heading, "spec_table", builder.page_scope, 1)
    term: str | None = None
    items: list[_Node] = []

    def flatten(node: _Node) -> None:
        for child in node.children:
            if isinstance(child, _Node) and not _is_noise(child):
                if child.tag in ("dt", "dd"):
                    items.append(child)
                else:
                    flatten(child)

    flatten(dl)
    for item in items:
        text = _node_text(item)
        if item.tag == "dt":
            term = text
        elif text:
            if term:
                builder.add_entry(section, term, text, term)
            else:
                builder.add_text(section, text)


# ---- helpers ------------------------------------------------------------------------

def _load_json(block: str) -> Any:
    text = block.strip()
    text = re.sub(r"^<!--|-->$", "", text).strip()
    text = re.sub(r"^//\s*<!\[CDATA\[|//\s*\]\]>$", "", text).strip()
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]{0,300}>", " ", text)


def _clean(text: str) -> str:
    return _squash(html_lib.unescape(_strip_tags(text)))


def _squash(text: str) -> str:
    return " ".join(str(text).replace(" ", " ").split())


def _chunks(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks, current = [], ""
    for word in text.split(" "):
        if current and len(current) + 1 + len(word) > size:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks[:12]

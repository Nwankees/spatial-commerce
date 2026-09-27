"""Extracts explicitly stated dimensions from retailer HTML and provider specs.

Each extractor returns a DimensionCandidate for one source type so the
resolver can apply the documented priority order. Parsing is tolerant of
malformed HTML (stdlib html.parser) and never raises on bad input.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Iterable

from .dimension_parsing import (
    DimensionCandidate,
    parse_field,
    parse_labeled_dimensions,
    parse_length,
    unit_code_to_meters,
)

logger = logging.getLogger(__name__)

SOURCE_PRIORITY = ("json_ld", "structured_metadata", "spec_table", "page_text")
MAX_HTML_CHARS = 3_000_000
_PRODUCT_TYPES = {"product", "productgroup", "individualproduct", "productmodel"}
_AXES = ("width", "depth", "height")
_BLOCK_TAGS = {"p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6", "section", "td", "th", "dd", "dt", "span"}
# Untyped name/value pairs inside arbitrary page scripts are deliberately not
# used: pages embed data for related products, so their owner is ambiguous.


@dataclass(frozen=True)
class SourceFinding:
    source_type: str
    candidate: DimensionCandidate
    source_url: str | None


class _PageCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.jsonld: list[str] = []
        self.scripts: list[str] = []
        self.itemprops: list[tuple[str, str]] = []
        self.rows: list[list[str]] = []
        self.dl_pairs: list[tuple[str, str]] = []
        self.text_parts: list[str] = []
        self._script_kind: str | None = None
        self._script_buf: list[str] = []
        self._skip_depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._dt: list[str] | None = None
        self._dd: list[str] | None = None
        self._last_dt: str | None = None
        self._itemprop_stack: list[tuple[str, str, list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {k.lower(): (v or "") for k, v in attrs}
        if tag == "script":
            kind = attributes.get("type", "").lower()
            self._script_kind = "jsonld" if "ld+json" in kind else "script"
            self._script_buf = []
            return
        if tag in ("style", "noscript", "template", "svg"):
            self._skip_depth += 1
            return
        prop = attributes.get("itemprop", "").lower()
        if prop in _AXES:
            content = attributes.get("content")
            if content:
                self.itemprops.append((prop, content))
            else:
                self._itemprop_stack.append((tag, prop, []))
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "dt":
            self._dt = []
        elif tag == "dd":
            self._dd = []
        if tag in _BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script_kind is not None:
            content = "".join(self._script_buf)
            (self.jsonld if self._script_kind == "jsonld" else self.scripts).append(content)
            self._script_kind = None
            return
        if tag in ("style", "noscript", "template", "svg"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._itemprop_stack and self._itemprop_stack[-1][0] == tag:
            _, prop, buf = self._itemprop_stack.pop()
            self.itemprops.append((prop, " ".join("".join(buf).split())))
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag == "dt" and self._dt is not None:
            self._last_dt = " ".join("".join(self._dt).split())
            self._dt = None
        elif tag == "dd" and self._dd is not None:
            if self._last_dt:
                self.dl_pairs.append((self._last_dt, " ".join("".join(self._dd).split())))
            self._dd = None
            self._last_dt = None
        if tag in _BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._script_kind is not None:
            self._script_buf.append(data)
            return
        if self._skip_depth:
            return
        for buffer in (self._cell, self._dt, self._dd):
            if buffer is not None:
                buffer.append(data)
        for _, _, buf in self._itemprop_stack:
            buf.append(data)
        self.text_parts.append(data)


def extract_from_html(html: str, source_url: str | None) -> list[SourceFinding]:
    """Returns one finding per source type that yielded at least one axis."""
    collector = _PageCollector()
    try:
        collector.feed(html[:MAX_HTML_CHARS])
        collector.close()
    except Exception:  # html.parser is tolerant, but never let a bad page escape.
        logger.info("Retailer HTML could not be fully parsed; using what was collected.")

    findings: list[SourceFinding] = []

    json_ld = DimensionCandidate()
    for block in collector.jsonld:
        for node in _iter_jsonld_products(_load_json(block)):
            json_ld.merge(_candidate_from_jsonld_product(node))
    findings.append(SourceFinding("json_ld", json_ld, source_url))

    metadata = DimensionCandidate()
    for axis, value in collector.itemprops:
        meters = parse_length(value)
        if meters is not None:
            metadata.add(axis, meters, f"itemprop {axis}: {value}")
    findings.append(SourceFinding("structured_metadata", metadata, source_url))

    spec = DimensionCandidate()
    for row in collector.rows:
        if len(row) == 2:
            spec.merge(parse_field(row[0], row[1]))
    for label, value in collector.dl_pairs:
        spec.merge(parse_field(label, value))
    findings.append(SourceFinding("spec_table", spec, source_url))

    text = re.sub(r"[ \t]+", " ", "".join(collector.text_parts))
    findings.append(
        SourceFinding("page_text", parse_labeled_dimensions(text, require_keyword=True), source_url)
    )
    return [finding for finding in findings if finding.candidate.known_axes]


def extract_from_features(features: Iterable[tuple[str, str]], source_url: str | None) -> list[SourceFinding]:
    """Provider product-detail specs (e.g. SerpApi 'about_the_product.features')."""
    candidate = DimensionCandidate()
    for label, value in features:
        candidate.merge(parse_field(label, value))
    return [SourceFinding("structured_metadata", candidate, source_url)] if candidate.known_axes else []


def _load_json(block: str) -> Any:
    text = block.strip()
    text = re.sub(r"^<!--|-->$", "", text).strip()
    text = re.sub(r"^//\s*<!\[CDATA\[|//\s*\]\]>$", "", text).strip()
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def _iter_jsonld_products(data: Any, depth: int = 0) -> Iterable[dict[str, Any]]:
    if depth > 12:
        return
    if isinstance(data, list):
        for item in data:
            yield from _iter_jsonld_products(item, depth + 1)
    elif isinstance(data, dict):
        types = data.get("@type")
        type_names = {str(t).lower() for t in (types if isinstance(types, list) else [types]) if t}
        if type_names & _PRODUCT_TYPES:
            yield data
        for key in ("@graph", "hasVariant", "isVariantOf", "mainEntity", "itemListElement", "item"):
            if key in data:
                yield from _iter_jsonld_products(data[key], depth + 1)


def _candidate_from_jsonld_product(node: dict[str, Any]) -> DimensionCandidate:
    candidate = DimensionCandidate()
    for axis in _AXES:
        meters, raw = _quantity_to_meters(node.get(axis))
        if meters is not None:
            candidate.add(axis, meters, f"{axis}: {raw}")
    properties = node.get("additionalProperty")
    for prop in properties if isinstance(properties, list) else [properties]:
        if not isinstance(prop, dict):
            continue
        name = prop.get("name")
        value = prop.get("value")
        if not isinstance(name, str) or value is None or isinstance(value, (dict, list, bool)):
            continue
        unit = prop.get("unitText") or prop.get("unitCode")
        value_text = str(value)
        if isinstance(value, (int, float)) and isinstance(unit, str):
            factor = unit_code_to_meters(unit)
            if factor is None:
                continue
            value_text = f"{float(value) * factor * 1000:g} mm"
        candidate.merge(parse_field(name, value_text))
    return candidate


def _quantity_to_meters(value: Any) -> tuple[float | None, str | None]:
    if isinstance(value, str):
        return parse_length(value), value
    if isinstance(value, dict):
        number = value.get("value")
        unit = value.get("unitCode") or value.get("unitText")
        if isinstance(number, str):
            if unit is None:
                return parse_length(number), number
            try:
                number = float(number)
            except ValueError:
                return None, None
        if isinstance(number, (int, float)) and not isinstance(number, bool) and isinstance(unit, str):
            factor = unit_code_to_meters(unit)
            if factor is not None:
                return parse_length(f"{float(number) * factor * 1000:g} mm"), f"{number} {unit}"
    # Bare numbers carry no unit and are not trustworthy.
    return None, None

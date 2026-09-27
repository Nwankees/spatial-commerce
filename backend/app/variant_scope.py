"""Scopes retailer-page evidence to the exact selected variant's own data record.

Generic: embedded page data (JSON in <script> tags) is searched for objects whose
identifier-like key (tcin, usItemId, sku, variantId, …) *exactly* equals a retailer ID
taken from the selected offer's URL. Only that object's own content is used; nested
objects carrying a different identifier (sibling/child variants), AI-generated
summaries, reviews, media and package/shipping data are skipped. No page order,
proximity, dimension values or LLM judgement is used to pick the variant.
"""
from __future__ import annotations

import html as html_lib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .dimension_evidence import AI_GENERATED_RE

ID_KEYS = {
    "tcin", "usitemid", "itemid", "item_id", "sku", "skuid", "sku_id", "variantid", "variant_id",
    "productid", "product_id", "id", "offerid", "offer_id", "partnumber", "itemnumber", "catentryid",
}
_SKIP_KEY_RE = re.compile(
    r"review|rating|question|answer|image|media|gallery|video|swatch|package|shipping|recommend|similar|related|"
    r"sponsor|advert|breadcrumb|seo|promotion",
    re.IGNORECASE,
)
_NAME_KEYS = ("name", "label", "displayName", "display_name", "title", "attributeName", "key")
_VALUE_KEYS = ("value", "values", "displayValue", "display_value", "attributeValue", "attributeValues")
_KEYWORD_RE = re.compile(r"dimension|\bwidth\b|\bdepth\b|\bheight\b|\boverall\b|product size|assembled|\bsize\b", re.I)
_MAX_SCRIPT_CHARS = 6_000_000


@dataclass
class VariantRecord:
    path: str
    id_key: str
    id_value: str
    pairs: list[tuple[str, str]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.pairs or self.lines)


def find_variant_records(page_html: str, ids: list[str]) -> list[VariantRecord]:
    wanted = {str(i) for i in ids if i}
    if not wanted:
        return []
    records: list[VariantRecord] = []
    for data in _embedded_json(page_html):
        _walk(data, "$", wanted, records, depth=0)
    return records


def record_evidence(records: list[VariantRecord], max_chars: int = 6000) -> str:
    lines: list[str] = []
    for record in records:
        lines.extend(f"{label}: {value}" for label, value in record.pairs)
        lines.extend(record.lines)
    kept, seen, total = [], set(), 0
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        if total + len(line) + 1 > max_chars:
            break
        seen.add(key)
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)


def _walk(node: Any, path: str, wanted: set[str], records: list[VariantRecord], depth: int) -> None:
    if depth > 40:
        return
    if isinstance(node, dict):
        match = _matching_id(node, wanted)
        if match:
            record = VariantRecord(path, match[0], match[1])
            _collect(node, record, wanted_id=match[1], depth=0, top=True)
            records.append(record)
            return  # the record's own subtree has been consumed
        for key, value in node.items():
            _walk(value, f"{path}.{key}", wanted, records, depth + 1)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, f"{path}[{index}]", wanted, records, depth + 1)


def _matching_id(node: dict, wanted: set[str]) -> tuple[str, str] | None:
    for key, value in node.items():
        if key.lower() in ID_KEYS and isinstance(value, (str, int)) and not isinstance(value, bool) and str(value) in wanted:
            return key, str(value)
    return None


def _other_id(node: dict, wanted_id: str) -> bool:
    """True if the object belongs to a different item (e.g. a sibling variant)."""
    for key, value in node.items():
        if key.lower() in ID_KEYS and isinstance(value, (str, int)) and not isinstance(value, bool):
            if str(value) != wanted_id and re.fullmatch(r"[A-Za-z0-9_-]{4,}", str(value)):
                return True
    return False


def _collect(node: Any, record: VariantRecord, wanted_id: str, depth: int, top: bool = False) -> None:
    if depth > 25:
        return
    if isinstance(node, dict):
        if not top and _other_id(node, wanted_id):
            return
        label = next((node[k] for k in _NAME_KEYS if isinstance(node.get(k), str)), None)
        value = next((node[k] for k in _VALUE_KEYS if k in node), None)
        if isinstance(value, list) and value and all(isinstance(v, (str, int, float)) for v in value):
            value = ", ".join(str(v) for v in value)
        if isinstance(label, str) and isinstance(value, (str, int, float)) and not isinstance(value, bool):
            label, text = _clean(label), _clean(str(value))
            if label and text and len(label) <= 80 and len(text) <= 200 and not AI_GENERATED_RE.search(label):
                record.pairs.append((label, text))
        for key, child in node.items():
            if _SKIP_KEY_RE.search(key) or AI_GENERATED_RE.search(key):
                continue
            _collect(child, record, wanted_id, depth + 1)
    elif isinstance(node, list):
        for child in node:
            _collect(child, record, wanted_id, depth + 1)
    elif isinstance(node, str) and 6 <= len(node) <= 4000:
        text = _clean(node)
        if _KEYWORD_RE.search(text) and re.search(r"\d", text) and not AI_GENERATED_RE.search(text):
            record.lines.append(text[:400])


def _clean(text: str) -> str:
    text = html_lib.unescape(re.sub(r"<[^>]{0,200}>", " ", text))
    return " ".join(text.replace(" ", " ").split())


def _embedded_json(page_html: str) -> list[Any]:
    found = []
    for match in re.finditer(r"<script\b[^>]*>(.*?)</script>", page_html[:20_000_000], re.S | re.I):
        body = match.group(1).strip()
        if not body or len(body) > _MAX_SCRIPT_CHARS:
            continue
        candidates = [body]
        assignment = re.match(r"^(?:window\.|var\s+|let\s+|const\s+)?[\w.$\[\]'\"]+\s*=\s*(.*?);?\s*$", body, re.S)
        if assignment:
            candidates.append(assignment.group(1))
        for text in candidates:
            if text[:1] in "{[":
                try:
                    found.append(json.loads(text))
                    break
                except (ValueError, RecursionError):
                    continue
    return found


def ids_for_selected_options(page_html: str, options: dict[str, str]) -> set[str]:
    """Item IDs whose own data explicitly carries *every* selected option value.

    Looks for objects such as {"name": "Color", "value": "Gray", "tcin": "123"} or
    {"color": "Gray", "sku": "123"}. Exact, case-insensitive equality only. The caller
    uses the result only when exactly one ID matches all options.
    """
    wanted = {name.lower(): value.strip().lower() for name, value in options.items() if value}
    if not wanted:
        return set()
    per_option: dict[str, set[str]] = {name: set() for name in wanted}

    def visit(node: Any, depth: int) -> None:
        if depth > 40:
            return
        if isinstance(node, dict):
            ids = {str(v) for k, v in node.items() if k.lower() in ID_KEYS and isinstance(v, (str, int)) and not isinstance(v, bool)}
            if ids:
                name = next((node[k] for k in _NAME_KEYS if isinstance(node.get(k), str)), None)
                value = next((node[k] for k in _VALUE_KEYS if isinstance(node.get(k), str)), None)
                for option, wanted_value in wanted.items():
                    direct = node.get(option) or node.get(option.capitalize())
                    if (isinstance(name, str) and name.strip().lower() == option and isinstance(value, str)
                            and value.strip().lower() == wanted_value) or (
                            isinstance(direct, str) and direct.strip().lower() == wanted_value):
                        per_option[option].update(ids)
            for child in node.values():
                visit(child, depth + 1)
        elif isinstance(node, list):
            for child in node:
                visit(child, depth + 1)

    for data in _embedded_json(page_html):
        visit(data, 0)
    sets = list(per_option.values())
    return set.intersection(*sets) if sets and all(sets) else set()

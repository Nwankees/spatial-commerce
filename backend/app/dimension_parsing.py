"""Conservative parsing of explicitly stated product dimensions.

Everything here is pure (no I/O). A value is only accepted when both its
number and its unit are explicit (or the unit is explicitly declared by the
surrounding label), and it is only mapped to width/depth/height when the
source labels it as such. Anything ambiguous is rejected rather than guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

# ---- Units ---------------------------------------------------------------

_UNIT_METERS: dict[str, float] = {
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001, "millimetre": 0.001, "millimetres": 0.001,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01, "centimetre": 0.01, "centimetres": 0.01,
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "in": 0.0254, "in.": 0.0254, "inch": 0.0254, "inches": 0.0254,
    '"': 0.0254, "”": 0.0254, "″": 0.0254, "''": 0.0254, "′′": 0.0254,
    "ft": 0.3048, "ft.": 0.3048, "foot": 0.3048, "feet": 0.3048,
}
# UN/CEFACT codes used by schema.org QuantitativeValue.unitCode.
_UNIT_CODES: dict[str, float] = {"MMT": 0.001, "CMT": 0.01, "MTR": 1.0, "INH": 0.0254, "FOT": 0.3048}

MIN_DIMENSION_M = 0.01
MAX_DIMENSION_M = 10.0

_NUM = r"\d+(?:\.\d+)?(?:\s+\d+/\d+)?|\d+/\d+|\d*\.\d+"
_ALPHA_UNIT = r"(?:millimet(?:er|re)s?|centimet(?:er|re)s?|met(?:er|re)s?|inch(?:es)?|feet|foot|mm|cm|in\.?|ft\.?|m)(?:(?![a-z])|(?=[wdhl]\b))"
_SYMBOL_UNIT = r"""(?:"|”|″|''|′′)"""
_UNIT = rf"(?:{_ALPHA_UNIT}|{_SYMBOL_UNIT})"
_LABEL = r"(?:width|depth|height|length|w|d|h|l)"
_TOKEN = rf"(?:\b{_LABEL}\b\s*[:.]?\s*)?(?:{_NUM})\s*(?:{_UNIT})?\s*(?:{_LABEL}\b\.?)?"
_SEP = r"\s*(?:x|×|\*|by)\s*"
_TRAILING_UNIT = r"(?:\s*\(?\s*(?:inches|inch|in\.?|cm|mm|ft|feet|meters|m)\s*\)?(?![a-z]))?"
# Groups must not start in the middle of another number/unit (e.g. inside
# "2 ft 6 in") and must not be followed by a fourth value, which is ambiguous.
_TRIPLET_RE = re.compile(
    rf"(?<![\w.\"”″'′/]){_TOKEN}{_SEP}{_TOKEN}(?:{_SEP}{_TOKEN})?{_TRAILING_UNIT}(?!\s*(?:x|×|\*|by)\s*\d)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    rf"^\s*(?:(?P<pre>{_LABEL})\b\s*[:.]?\s*)?(?P<num>{_NUM})\s*(?P<unit>{_UNIT})?\s*(?:(?P<post>{_LABEL})\b\.?)?\s*$",
    re.IGNORECASE,
)
_SPLIT_RE = re.compile(_SEP, re.IGNORECASE)
_ORDER_RE = re.compile(r"\b([WDHL])\s*[x×*]\s*([WDHL])(?:\s*[x×*]\s*([WDHL]))?\b", re.IGNORECASE)
_CONTEXT_UNIT_RE = re.compile(
    r"\(\s*(inches|inch|in\.?|cm|mm|ft|feet|meters|m)\s*\)|\bin\s+(inches|centimeters|millimeters|feet)\b",
    re.IGNORECASE,
)
_FEET_INCHES_RE = re.compile(
    rf"^\s*(?P<ft>\d+)\s*(?:'|’|′|ft\.?|feet|foot)\s*(?P<inch>{_NUM})\s*(?:\"|”|″|''|in\.?|inch(?:es)?)\s*$",
    re.IGNORECASE,
)
_SINGLE_RE = re.compile(rf"^\s*(?P<num>{_NUM})\s*(?P<unit>{_UNIT})?\s*$", re.IGNORECASE)
# Contexts that describe something other than the assembled product footprint.
_EXCLUDED_CONTEXT = re.compile(
    r"\b(package|packaging|packaged|shipping|shipment|carton|box|boxed|parcel|seat|arm|armrest|back|backrest|"
    r"leg|shelf|shelves|drawer|interior|inside|inner|opening|cushion|clearance|minimum|maximum|min|max|"
    r"adjustable|mattress|screen|display|monitor|keyboard tray)\b",
    re.IGNORECASE,
)
_LABEL_TO_AXIS = {"w": "width", "width": "width", "d": "depth", "depth": "depth", "h": "height", "height": "height"}
_FIELD_LABEL_RE = re.compile(
    r"^(?:overall|product|item|assembled|assembled product|outside|exterior|total|external)?\s*"
    r"(width|depth|height)(?:\s*overall)?$"
)
_DIMENSIONS_LABEL_RE = re.compile(
    r"^(?:overall|product|item|assembled|assembled product|furniture|outside|exterior)?\s*"
    r"(?:dimensions?|size|measurements?)(?:\s*overall)?$"
)


def unit_to_meters(unit: str | None) -> float | None:
    if unit is None:
        return None
    return _UNIT_METERS.get(unit.strip().lower())


def unit_code_to_meters(code: str | None) -> float | None:
    if not code:
        return None
    return _UNIT_CODES.get(code.strip().upper()) or unit_to_meters(code)


def parse_number(text: str) -> float | None:
    text = text.strip()
    try:
        if " " in text:
            whole, frac = text.split(None, 1)
            return float(Fraction(whole) + Fraction(frac))
        return float(Fraction(text)) if "/" in text else float(text)
    except (ValueError, ZeroDivisionError):
        return None


def _in_range(meters: float | None) -> float | None:
    if meters is None or not (MIN_DIMENSION_M <= meters <= MAX_DIMENSION_M):
        return None
    return round(meters, 4)


def parse_length(text: str, unit_hint: str | None = None) -> float | None:
    """Parses one explicit length ('30 in', '76.2 cm', '2 ft 6 in') to meters.

    A bare number is only accepted when ``unit_hint`` comes from an explicit
    label such as 'Width (in)'. Returns None for anything else.
    """
    if not text:
        return None
    text = " ".join(text.replace(" ", " ").split())
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    feet_inches = _FEET_INCHES_RE.match(text)
    if feet_inches:
        inches = parse_number(feet_inches.group("inch"))
        if inches is None:
            return None
        return _in_range(int(feet_inches.group("ft")) * 0.3048 + inches * 0.0254)
    single = _SINGLE_RE.match(text)
    if not single:
        return None
    number = parse_number(single.group("num"))
    factor = unit_to_meters(single.group("unit")) or unit_to_meters(unit_hint)
    if number is None or factor is None:
        return None
    return _in_range(number * factor)


@dataclass
class DimensionCandidate:
    """Values explicitly found in one source; each axis may be missing."""

    width: float | None = None
    depth: float | None = None
    height: float | None = None
    raw: list[str] = field(default_factory=list)
    conflicts: set[str] = field(default_factory=set)

    @property
    def known_axes(self) -> int:
        return sum(value is not None for value in (self.width, self.depth, self.height))

    def add(self, axis: str, meters: float, raw: str) -> None:
        if axis in self.conflicts:
            return
        current = getattr(self, axis)
        if current is not None and abs(current - meters) > 0.002:
            # Two different explicit values for the same axis: refuse to pick one.
            setattr(self, axis, None)
            self.conflicts.add(axis)
            return
        setattr(self, axis, meters)
        if raw and raw not in self.raw:
            self.raw.append(raw)

    def merge(self, other: DimensionCandidate) -> None:
        for axis in ("width", "depth", "height"):
            if axis in other.conflicts:
                setattr(self, axis, None)
                self.conflicts.add(axis)
            value = getattr(other, axis)
            if value is not None:
                self.add(axis, value, "")
        for raw in other.raw:
            if raw and raw not in self.raw:
                self.raw.append(raw)

    def raw_text(self) -> str | None:
        text = " | ".join(r for r in self.raw if r)
        return text[:300] or None


def normalize_label(label: str) -> tuple[str, str | None]:
    """Lowercases a label and extracts an explicit unit hint like '(in)'."""
    label = " ".join(label.replace(" ", " ").split()).strip().rstrip(":").strip()
    unit_hint = None
    hint = re.search(r"\(\s*([a-z.\"”']+)\s*\)\s*$", label, re.IGNORECASE)
    if hint and unit_to_meters(hint.group(1)) is not None:
        unit_hint = hint.group(1)
        label = label[: hint.start()].strip()
    return label.lower(), unit_hint


def axis_for_field_label(label: str) -> tuple[str | None, str | None]:
    """Maps an explicit field label to an axis ('Overall Width (in)' -> width)."""
    normalized, unit_hint = normalize_label(label)
    if _EXCLUDED_CONTEXT.search(normalized):
        return None, None
    match = _FIELD_LABEL_RE.match(normalized)
    return (match.group(1), unit_hint) if match else (None, None)


def is_dimensions_label(label: str) -> bool:
    normalized, _ = normalize_label(re.sub(r"\([^)]*[x×][^)]*\)", "", label))
    return bool(_DIMENSIONS_LABEL_RE.match(normalized)) and not _EXCLUDED_CONTEXT.search(label)


_VALUE_DESIGNATOR_RE = re.compile(
    r"\s*\b(w|wide|width|d|deep|depth|h|high|tall|height|l|long|length)\b\.?\s*$", re.IGNORECASE
)
_DESIGNATOR_AXIS = {
    "w": "width", "wide": "width", "width": "width",
    "d": "depth", "deep": "depth", "depth": "depth",
    "h": "height", "high": "height", "tall": "height", "height": "height",
}
_PAGE_TEXT_KEYWORD_RE = re.compile(
    r"(dimension|overall|assembled|\bsize\b|measures|measurements)", re.IGNORECASE
)


def parse_field(label: str, value: str) -> DimensionCandidate:
    """Parses one label/value pair from a spec table or structured property."""
    candidate = DimensionCandidate()
    if not label or not value or _EXCLUDED_CONTEXT.search(value):
        return candidate
    axis, unit_hint = axis_for_field_label(label)
    if axis:
        text = value.strip()
        designator = _VALUE_DESIGNATOR_RE.search(text)
        if designator:
            # '30.25'' H' under a Height label is fine; '38" L' or a mismatched axis is not.
            if _DESIGNATOR_AXIS.get(designator.group(1).lower()) != axis:
                return candidate
            text = text[: designator.start()]
        meters = parse_length(text, unit_hint)
        if meters is not None:
            candidate.add(axis, meters, f"{label.strip()}: {value.strip()}")
        return candidate
    if is_dimensions_label(label):
        return parse_labeled_dimensions(value, context=label)
    return candidate


def parse_labeled_dimensions(text: str, context: str = "", require_keyword: bool = False) -> DimensionCandidate:
    """Finds explicit W/D/H dimension groups such as '30" W x 28" D x 35" H'.

    Each number must be labeled W/D/H itself, or the context must declare the
    order explicitly (e.g. 'Dimensions (W x D x H)'). Unlabeled groups, groups
    using 'L', and package/seat/etc. contexts are rejected. With
    ``require_keyword`` (free page text) a dimensions keyword must appear just
    before the group, so unrelated text such as 'fits 12" W laptops' is ignored.
    """
    result = DimensionCandidate()
    if not text or (context and _EXCLUDED_CONTEXT.search(context)):
        return result
    text = text.replace("\u00a0", " ")
    for match in _TRIPLET_RE.finditer(text):
        before = text[max(0, match.start() - 50): match.start()]
        if _CONTINUES_MEASUREMENT_RE.search(before):
            continue  # e.g. the '6 in W' inside '2 ft 6 in W': never parse a fragment.
        if _EXCLUDED_CONTEXT.search(before):
            continue
        if require_keyword and not _PAGE_TEXT_KEYWORD_RE.search(before[-40:]):
            continue
        group = _parse_group(match.group(0), f"{context} {before}".strip())
        if group is not None:
            result.merge(group)
    return result


_CONTINUES_MEASUREMENT_RE = re.compile(
    r"""(?:\d|\b(?:ft|feet|foot|in|inch|inches|cm|mm|m)\.?|["”″'′])\s*$""", re.IGNORECASE
)
_GROUP_TRAILING_UNIT_RE = re.compile(
    r"(?:\s+|\s*\()(inches|inch|in\.?|cm|mm|ft|feet|meters|m)\s*\)?\s*$", re.IGNORECASE
)


def _parse_group(group_text: str, context: str) -> DimensionCandidate | None:
    body = " ".join(group_text.split())
    trailing_unit = None
    trailing = _GROUP_TRAILING_UNIT_RE.search(body)
    if trailing:
        trailing_unit = trailing.group(1)
        body = body[: trailing.start()]
    tokens = []
    for part in _SPLIT_RE.split(body):
        token = _TOKEN_RE.match(part)
        if token is None:
            return None
        tokens.append(token)

    context_unit = None
    for unit_match in _CONTEXT_UNIT_RE.finditer(context):
        context_unit = unit_match.group(1) or unit_match.group(2)

    labels = [(t.group("pre") or t.group("post") or "").lower() for t in tokens]
    if all(labels):
        axes = [_LABEL_TO_AXIS.get(label) for label in labels]
    elif not any(labels):
        order = None
        for order in _ORDER_RE.finditer(context):
            pass
        if order is None:
            return None
        letters = [letter.lower() for letter in order.groups() if letter]
        if len(letters) != len(tokens):
            return None
        axes = [_LABEL_TO_AXIS.get(letter) for letter in letters]
    else:
        return None
    if None in axes or len(set(axes)) != len(axes):
        return None  # 'L' labels, repeated labels, or mixed schemes are ambiguous.

    candidate = DimensionCandidate()
    for token, axis in zip(tokens, axes):
        unit = token.group("unit") or trailing_unit or context_unit
        number = parse_number(token.group("num"))
        factor = unit_to_meters(unit)
        if number is None or factor is None:
            return None  # A number without an explicit unit is not trustworthy.
        meters = _in_range(number * factor)
        if meters is None:
            return None
        candidate.add(axis, meters, "")
    candidate.raw.append(" ".join(group_text.split()))
    return candidate

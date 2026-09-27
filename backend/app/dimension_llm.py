"""Local-LLM extraction of explicitly stated dimensions, with deterministic validation.

The model (OLLAMA_DIMENSION_MODEL, text only) is an extractor, never a source of
truth: every value it returns must be re-found in the supplied evidence with its
unit and an explicit axis label (or an explicitly declared axis order) before it is
accepted, and unit conversion is done here in code.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .dimension_parsing import MAX_DIMENSION_M, MIN_DIMENSION_M, parse_number

logger = logging.getLogger(__name__)

Unit = Literal["mm", "cm", "m", "in", "ft"]
_UNIT_METERS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}
_ALPHA_UNIT = r"millimet(?:er|re)s?|centimet(?:er|re)s?|met(?:er|re)s?|inch(?:es)?|feet|foot|mm|cm|in\.?|ft\.?|m"
_SYMBOL_UNIT = r"""''|"|”|″|'|’|′"""
# Alphabetic units must be whole words (or directly followed by an axis letter, e.g. "60cmW").
_UNIT_TOKEN_RE = re.compile(
    rf"(?:(?<![a-z])({_ALPHA_UNIT})(?:(?![a-z])|(?=[wdhl](?![a-z])))|({_SYMBOL_UNIT}))", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?(?:\s+\d+/\d+)?|\d+/\d+)(?![\d.])")
_SEGMENT_SPLIT_RE = re.compile(r"\s[x×]\s|[x×](?=\s*\d)|[,;|\n]|\s/\s", re.IGNORECASE)
_AXIS_LABEL_RE = {
    "width": re.compile(r"\bwidth\b|\bwide\b|\(\s*w\s*\)|(?<![a-z])w(?![a-z])", re.IGNORECASE),
    "depth": re.compile(r"\bdepth\b|\bdeep\b|\(\s*d\s*\)|(?<![a-z])d(?![a-z])", re.IGNORECASE),
    "height": re.compile(r"\bheight\b|\bhigh\b|\btall\b|\(\s*h\s*\)|(?<![a-z])h(?![a-z])", re.IGNORECASE),
    "length": re.compile(r"\blength\b|\blong\b|\(\s*l\s*\)|(?<![a-z])l(?![a-z])", re.IGNORECASE),
}
_ORDER_DECL_RE = re.compile(r"\(?\s*\b([WDHL])\s*[x×]\s*([WDHL])(?:\s*[x×]\s*([WDHL]))?\b\s*\)?", re.IGNORECASE)
# Letter-based boundaries so JSON keys such as "package_dimensions" are caught too.
_EXCLUDED_RE = re.compile(
    r"(?<![a-z])(package|packaging|shipping|carton|box|seat|arm|armrest|back|backrest|leg|shelf|drawer|interior|"
    r"inside|opening|cushion|clearance|minimum|maximum|min|max|adjustable)(?![a-z])",
    re.IGNORECASE,
)
AXES = ("width", "depth", "height")

SYSTEM_PROMPT = """You extract product dimensions from retailer page excerpts for a furniture-fit checker.
You are an extractor, not an estimator. Output is validated by software against the excerpt;
any value that is not literally present in the excerpt with its unit and axis label is discarded."""

USER_PROMPT = """Extract the product's OVERALL width, depth and height from the excerpt below.

Rules:
- Use ONLY dimensions explicitly stated in the excerpt. Never use the product title, type, images,
  typical sizes or world knowledge. Never invent or compute a missing axis.
- Map a number to width/depth/height ONLY when it is explicitly labeled, e.g.
  "39.4 inches (H) x 21.65 inches (W) x 22 inches (D)", "30\\"W x 28\\"D x 35\\"H",
  or separate "Width: ...", "Depth: ...", "Height: ..." fields, or when the excerpt explicitly
  declares the order, e.g. "Dimensions (W x D x H): 76 x 71 x 89 cm".
- An unlabeled triple like "39.4 x 21.65 x 22" is NOT enough: leave those axes null.
- "L x W x H" style labels are ambiguous: do not map L to width or depth.
- Ignore package/shipping/box dimensions and parts (seat, arm, leg, shelf, drawer, interior).
- Copy numbers exactly as written (no conversion) with their unit: one of mm, cm, m, in, ft.
- evidence_text: copy the exact excerpt text the values came from, verbatim.
- Always fill all three axis objects: {{"value": number, "unit": "in"}} when stated, or
  {{"value": null, "unit": null}} when that axis is not explicitly stated.
- Prefer lines labeled as overall/product dimensions; seat, back, base or adjustable-range
  measurements are parts, not the overall size.
- If nothing qualifies, set status "unavailable" and give a short reason.

Excerpt:
<<<
{evidence}
>>>"""


class ExtractedLength(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float | None = None
    unit: Unit | None = None


class DimensionExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Axis objects are always present (value/unit null when not stated): small models
    # otherwise tend to answer a whole axis with null even when it is stated.
    width: ExtractedLength = Field(default_factory=ExtractedLength)
    depth: ExtractedLength = Field(default_factory=ExtractedLength)
    height: ExtractedLength = Field(default_factory=ExtractedLength)
    raw_dimensions: str | None = Field(default=None, max_length=400)
    evidence_text: str | None = Field(default=None, max_length=600)
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["verified", "partial", "unavailable"]
    reason: str | None = Field(default=None, max_length=300)

    @field_validator("width", "depth", "height", mode="before")
    @classmethod
    def null_axis_means_not_stated(cls, value: Any) -> Any:
        return {} if value is None else value


class DimensionExtractionError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass
class ValidatedDimensions:
    width: float | None = None
    depth: float | None = None
    height: float | None = None
    evidence: str | None = None
    rejected: list[str] = field(default_factory=list)

    @property
    def known_axes(self) -> int:
        return sum(v is not None for v in (self.width, self.depth, self.height))


class OllamaDimensionExtractor:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 60.0,
        keep_alive: str | None = "15m",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._keep_alive = keep_alive
        self._transport = transport

    @property
    def model(self) -> str:
        return self._model

    def build_request(self, evidence: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(evidence=evidence)},
            ],
            "stream": False,
            "think": False,
            "format": extraction_json_schema(),
            "options": {"temperature": 0},
        }
        if self._keep_alive:
            body["keep_alive"] = self._keep_alive
        return body

    async def extract(self, evidence: str) -> DimensionExtraction:
        started = time.monotonic()
        outcome = "error"
        try:
            try:
                async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                    response = await client.post(f"{self._base_url}/api/chat", json=self.build_request(evidence))
            except httpx.TimeoutException:
                raise DimensionExtractionError("The local dimension model timed out.", retryable=True) from None
            except httpx.HTTPError as exc:
                logger.warning("Dimension model request failed: %s", type(exc).__name__)
                raise DimensionExtractionError("The local dimension model (Ollama) is not reachable.", retryable=True) from None
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if response.status_code != 200:
                error = payload.get("error") if isinstance(payload, dict) else None
                if response.status_code == 404 or (isinstance(error, str) and "not found" in error.lower()):
                    raise DimensionExtractionError(
                        f"The dimension model '{self._model}' is not installed in Ollama.", retryable=False
                    )
                raise DimensionExtractionError(
                    f"The dimension model returned HTTP {response.status_code}.", retryable=True
                )
            content = payload.get("message", {}).get("content") if isinstance(payload, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise DimensionExtractionError("The dimension model returned no content.", retryable=True)
            try:
                extraction = DimensionExtraction.model_validate(json.loads(_strip_fences(content)))
            except (ValueError, ValidationError):
                raise DimensionExtractionError("The dimension model returned malformed output.", retryable=True) from None
            outcome = extraction.status
            return extraction
        finally:
            logger.info("Dimension extraction model=%s outcome=%s duration=%.1fs",
                        self._model, outcome, time.monotonic() - started)


def validate_extraction(extraction: DimensionExtraction, source_text: str) -> ValidatedDimensions:
    """Accepts an axis only if the source text itself proves it. Never trusts model math or labels."""
    result = ValidatedDimensions()
    if extraction.status == "unavailable":
        return result
    source = _normalize_text(source_text)
    evidence = _normalize_text(extraction.evidence_text or "")
    if not evidence or evidence.lower() not in source.lower():
        # The quoted evidence must be verbatim source text; otherwise search the whole source.
        result.rejected.append("evidence_text is not verbatim source text")
        evidence = source
    used: set[tuple[int, int]] = set()
    for axis in AXES:
        claimed: ExtractedLength = getattr(extraction, axis)
        if claimed.value is None:
            continue
        if claimed.unit is None:
            result.rejected.append(f"{axis}: no unit")
            continue
        if claimed.value <= 0:
            result.rejected.append(f"{axis}: non-positive value")
            continue
        span = _find_supported_occurrence(evidence, axis, claimed.value, claimed.unit, used)
        if span is None:
            result.rejected.append(f"{axis}: {claimed.value:g} {claimed.unit} with an explicit {axis} label is not in the source")
            continue
        meters = claimed.value * _UNIT_METERS[claimed.unit]
        if not (MIN_DIMENSION_M <= meters <= MAX_DIMENSION_M):
            result.rejected.append(f"{axis}: out of plausible range")
            continue
        conflict = _conflicting_statement(source, evidence, axis, claimed.value, claimed.unit)
        if conflict:
            # e.g. product variants listing different overall sizes: refuse to pick one.
            result.rejected.append(f"{axis}: conflicting statement in source: {conflict[:120]}")
            continue
        used.add(span)
        setattr(result, axis, round(meters, 5))
    if result.known_axes:
        result.evidence = (extraction.evidence_text or extraction.raw_dimensions or "")[:300] or None
    return result


def _find_supported_occurrence(
    text: str, axis: str, value: float, unit: str, used: set[tuple[int, int]]
) -> tuple[int, int] | None:
    for found_value, found_unit, span in _labeled_values(text, axis):
        if abs(found_value - value) <= 1e-9 and found_unit == unit and span not in used:
            return span
    return None


def _labeled_values(text: str, axis: str):
    """Yields (value, unit, span) for every number the text explicitly labels as ``axis``."""
    order = _declared_order(text)
    declaration = _ORDER_DECL_RE.search(text) if order else None
    if declaration:
        # Mask the "(W x D x H)" declaration so its letters are not read as value labels.
        text = text[: declaration.start()] + " " * (declaration.end() - declaration.start()) + text[declaration.end():]
    numbers = list(_NUMBER_RE.finditer(text))
    for match in numbers:
        number = parse_number(match.group(1))
        if number is None:
            continue
        segment_start, segment_end = _segment_bounds(text, match.start(), match.end())
        segment = text[segment_start:segment_end]
        before = text[max(0, segment_start - 40):segment_start]
        if _EXCLUDED_RE.search(segment) or _EXCLUDED_RE.search(before):
            continue
        unit = _unit_for(text, match.end(), segment_end)
        if unit is None:
            continue
        labels = {name for name, regex in _AXIS_LABEL_RE.items() if regex.search(_strip_number_and_unit(segment))}
        if labels:
            if labels == {axis}:
                yield number, unit, match.span()
            continue  # labeled with another axis, L, or several labels: ambiguous
        if order and declaration and _position_in_group(text, numbers, match, declaration.end()) == order.get(axis):
            yield number, unit, match.span()


def _conflicting_statement(source: str, evidence: str, axis: str, value: float, unit: str) -> str | None:
    """Returns a same-labeled statement elsewhere in the source (e.g. another variant's
    "Dimensions (Overall): ...") that gives this axis a different value, if any."""
    prefix_match = re.match(r"^\s*([^\d]{6,60}?)\s*\d", evidence)
    if not prefix_match:
        return None
    prefix = prefix_match.group(1).strip()
    lowered = source.lower()
    start = 0
    while (index := lowered.find(prefix.lower(), start)) != -1:
        start = index + len(prefix)
        statement = re.split(r'["\n]', source[index: index + 200])[0]
        others = {(round(v, 6), u) for v, u, _ in _labeled_values(statement, axis)}
        if others and (round(value, 6), unit) not in others:
            return statement
    return None


def _segment_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    for sep in _SEGMENT_SPLIT_RE.finditer(text[:start]):
        left = sep.end()
    right_match = _SEGMENT_SPLIT_RE.search(text, end)
    right = right_match.start() if right_match else len(text)
    # A "Width:" style label may sit just before a line's value.
    return left, right


def _unit_for(text: str, number_end: int, segment_end: int) -> str | None:
    """Unit right after the number, else the group's trailing unit (e.g. '30 W x 28 D x 35 H in')."""
    for scope_end in (segment_end, min(len(text), number_end + 80)):
        unit = _UNIT_TOKEN_RE.search(text, number_end, scope_end)
        if unit:
            return _canonical_unit(unit.group(1) or unit.group(2))
    return None


def _canonical_unit(token: str) -> str | None:
    t = token.lower().rstrip(".")
    if t in ('"', "”", "″", "''", "in", "inch", "inches"):
        return "in"
    if t in ("'", "’", "′", "ft", "feet", "foot"):
        return "ft"
    if t.startswith("millimet") or t == "mm":
        return "mm"
    if t.startswith("centimet") or t == "cm":
        return "cm"
    if t.startswith("met") or t == "m":
        return "m"
    return None


def _strip_number_and_unit(segment: str) -> str:
    segment = _NUMBER_RE.sub(" ", segment)
    return _UNIT_TOKEN_RE.sub(" ", segment)


def _declared_order(text: str) -> dict[str, int] | None:
    match = _ORDER_DECL_RE.search(text)
    if not match:
        return None
    letters = [g.lower() for g in match.groups() if g]
    if "l" in letters or len(set(letters)) != len(letters):
        return None
    mapping = {"w": "width", "d": "depth", "h": "height"}
    return {mapping[letter]: index for index, letter in enumerate(letters)}


def _position_in_group(
    text: str, numbers: list[re.Match[str]], match: re.Match[str], declaration_end: int
) -> int | None:
    """Index of a number within an 'a x b x c' group that follows an order declaration."""
    if match.start() < declaration_end:
        return None
    group = [m for m in numbers if m.start() >= declaration_end]
    chained: list[re.Match[str]] = []
    for m in group:
        if chained and not re.fullmatch(r"\s*(?:[a-z\"”″']+\.?\s*)?[x×]\s*", text[chained[-1].end():m.start()], re.IGNORECASE):
            break
        chained.append(m)
    for index, m in enumerate(chained):
        if m.span() == match.span():
            return index
    return None


def _normalize_text(text: str) -> str:
    """Unifies quotes/spaces but keeps line breaks, which separate labeled fields."""
    for fancy in ("\u2033", "\u201d", "\u201c"):
        text = text.replace(fancy, '"')
    text = text.replace("\u00a0", " ").replace("\r", "\n")
    lines = (" ".join(line.split()) for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _strip_fences(content: str) -> str:
    text = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    return fenced.group(1) if fenced else text


_SCHEMA: dict[str, Any] | None = None


def extraction_json_schema() -> dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None:
        schema = DimensionExtraction.model_json_schema()
        defs = schema.pop("$defs", {})

        def inline(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node:
                    return inline(defs[node["$ref"].split("/")[-1]])
                out = {k: inline(v) for k, v in node.items() if k not in ("title", "default", "additionalProperties")}
                if out.get("type") == "object" and "properties" in out:
                    out["required"] = list(out["properties"])
                return out
            if isinstance(node, list):
                return [inline(i) for i in node]
            return node

        _SCHEMA = inline(schema)
    return json.loads(json.dumps(_SCHEMA))


async def extract_and_validate(extractor: Any, evidence: str) -> ValidatedDimensions:
    """One extraction pass; if it quoted real source text but some axes did not validate,
    a second pass sees only that quoted statement (small models misread long excerpts).
    Both passes are validated against the full evidence; the better one is kept, never merged."""
    first = await extractor.extract(evidence)
    best = validate_extraction(first, evidence)
    quoted = _normalize_text(first.evidence_text or "")
    needs_retry = best.known_axes < 3 and (best.rejected or best.known_axes < 2)
    if needs_retry and quoted and quoted.lower() in _normalize_text(evidence).lower() and quoted != _normalize_text(evidence):
        second = validate_extraction(await extractor.extract(quoted), evidence)
        if second.known_axes > best.known_axes:
            second.rejected = best.rejected + second.rejected
            best = second
    return best

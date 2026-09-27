"""Semantic dimension extraction over a structured page representation.

qwen3:4b-instruct (OLLAMA_DIMENSION_MODEL) reads the parsed product page and points
at the field that states the selected product's overall physical dimensions. It
never is a source of truth: afterwards code checks, deterministically, that
  - the referenced entries exist in the representation that was shown,
  - every returned number literally occurs in those entries, each once,
  - the unit written there matches the claimed unit (conversion happens here),
  - all values come from one section of the selected variant's data,
  - axis labels, if any, come from the source text itself.
Regular expressions are used only for that verification (numbers, units, axis
letters) after the model has chosen a source -- never to decide what the model sees.
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

from .dimension_parsing import parse_number
from .page_representation import Entry, PageRepresentation, Section

logger = logging.getLogger(__name__)

Unit = Literal["mm", "cm", "m", "in", "ft"]
UNIT_METERS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}
MIN_M, MAX_M = 0.001, 10.0

SYSTEM_PROMPT = """You read a structured representation of one retailer product page and identify the
selected product's overall physical dimensions. You are an extractor, not an estimator.
Your answer is checked by software against the page data: every value must literally
appear in the entry you cite, with its unit."""

USER_PROMPT = """Selected product: {title}
Retailer: {retailer}
Selected variant / options: {variant}

Task: find the OVERALL physical dimensions (the whole assembled product) of the selected product in
the page data below.

Rules:
- Use only values present in the page data. Do not invent, estimate, convert or compute values.
- Do not use the product category, the title or typical sizes.
- Do not use package, packaging, shipping, box or carton dimensions.
- Do not use seat, arm, back, leg, shelf, drawer, interior, opening, cushion or other part/sub-component
  measurements, and not weight, capacity or screen size.
- Prefer fields that are explicitly overall / product / assembled / item dimensions. A field simply named
  "Dimensions", "Size", "Measurements", "Overall Size" etc. with 2-3 lengths usually is the overall size.
- Retailers use many names (e.g. Product Size, Assembled Size, Size Details, Item Measurements): judge by meaning.
- If the same main-product measurements appear both in an unlabeled field and elsewhere with explicit W/D/H
  labels, cite the labeled occurrence so the supported axis order is preserved. Do not substitute dimensions of an
  included component (for example an ottoman) for the selected product's own dimensions.
- If one clear overall-product field contains unlabeled dimensions (e.g. "13.39 x 4.13 x 4.13 in"), status MUST be
  "verified": return all values in the order written and leave axisMapping indices null. "Verified" means the
  measurements and their product-level meaning are clear; it does NOT mean their width/depth/height order is known.
- Map an axis only when the page labels it (e.g. 30"W x 20"D x 35"H, "(H)", "Width: ...",
  or a declared order such as "W x D x H"). Retailer Length is the second horizontal footprint axis and should
  be returned as depth for Check Fit (for example W x L or Width + Length).
- Separate Width / Depth / Height fields may be cited together (one sourceId per value) if they are in the same section.
- Sections with scope "selected item" are the selected variant's own data: prefer them over
  "product family" sections. Ignore data of other items (a different itemId) and AI-generated summaries.
- Use status "uncertain" ONLY when several different plausible overall dimension sets remain for the same selected
  product (or the selected product/variant itself is ambiguous). Never use "uncertain" merely because one clear
  three-value overall measurement has unknown axis order; that case is "verified" with null axis indices.
- If none are present, answer status "unavailable". Give a short reason in both cases.
- For every value give: value (number exactly as written), unit (mm, cm, m, in or ft), sourceId (the entry/text id,
  e.g. "E12" or "T3"), label (the axis label written next to it on the page, or null).
- rawText: copy the exact source value text you used.
- axisMapping: indices into your dimensions array, or null when not labeled; confidence 0-1; reason.

Page data (JSON):
{page}"""


class SemanticDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    unit: Unit
    sourceId: str = Field(max_length=16)
    label: str | None = Field(default=None, max_length=40)


class AxisMappingOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    widthIndex: int | None = None
    depthIndex: int | None = None
    heightIndex: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = Field(default=None, max_length=300)


class SemanticExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["verified", "partial", "unavailable", "uncertain"]
    dimensions: list[SemanticDimension] = Field(default_factory=list, max_length=6)
    rawText: str | None = Field(default=None, max_length=600)
    axisMapping: AxisMappingOut = Field(default_factory=AxisMappingOut)
    reason: str | None = Field(default=None, max_length=400)

    @field_validator("dimensions", mode="before")
    @classmethod
    def none_is_empty(cls, value: Any) -> Any:
        return [] if value is None else value

    @field_validator("axisMapping", mode="before")
    @classmethod
    def none_mapping(cls, value: Any) -> Any:
        return {} if value is None else value


class DimensionExtractionError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass
class AxisAssignment:
    widthIndex: int | None = None
    depthIndex: int | None = None
    heightIndex: int | None = None
    confidence: float = 0.0
    reason: str | None = None
    source: Literal["labels", "structured_fields", "none"] = "none"

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass
class SemanticResult:
    accepted: bool = False
    values_m: list[float] = field(default_factory=list)
    values_raw: list[tuple[float, str]] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    source_type: str | None = None
    source_path: str | None = None
    scope: str | None = None
    raw_text: str | None = None
    axis: AxisAssignment = field(default_factory=AxisAssignment)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    model_status: str | None = None
    model_reason: str | None = None

    @property
    def count(self) -> int:
        return len(self.values_m) if self.accepted else 0


# ---- Model client --------------------------------------------------------------------

class OllamaSemanticDimensionExtractor:
    def __init__(self, base_url: str, model: str, *, timeout_seconds: float = 90.0, num_ctx: int = 8192,
                 keep_alive: str | None = "15m", transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._transport = transport
        self.last_raw: str | None = None
        self.last_seconds: float | None = None

    @property
    def model(self) -> str:
        return self._model

    def build_request(self, page_json: str, title: str | None, retailer: str | None, variant: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(
                    title=title or "(unknown)", retailer=retailer or "(unknown)",
                    variant=json.dumps(variant, ensure_ascii=False) if variant else "(not specified)",
                    page=page_json)},
            ],
            "stream": False,
            "think": False,
            "format": semantic_json_schema(),
            "options": {"temperature": 0, "num_ctx": self._num_ctx},
        }
        if self._keep_alive:
            body["keep_alive"] = self._keep_alive
        return body

    async def extract(self, rep: PageRepresentation, page_json: str) -> SemanticExtraction:
        started = time.monotonic()
        outcome = "error"
        self.last_raw = None
        try:
            request = self.build_request(page_json, rep.product_title, rep.retailer, rep.selected_variant)
            try:
                async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                    response = await client.post(f"{self._base_url}/api/chat", json=request)
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
                    raise DimensionExtractionError(f"The dimension model '{self._model}' is not installed in Ollama.",
                                                   retryable=False)
                raise DimensionExtractionError(f"The dimension model returned HTTP {response.status_code}.", retryable=True)
            content = payload.get("message", {}).get("content") if isinstance(payload, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise DimensionExtractionError("The dimension model returned no content.", retryable=True)
            self.last_raw = content
            try:
                extraction = SemanticExtraction.model_validate(json.loads(_strip_fences(content)))
            except (ValueError, ValidationError):
                raise DimensionExtractionError("The dimension model returned malformed output.", retryable=True) from None
            outcome = extraction.status
            return extraction
        finally:
            self.last_seconds = round(time.monotonic() - started, 2)
            logger.info("Dimension extraction model=%s outcome=%s duration=%.1fs", self._model, outcome, self.last_seconds)


def _strip_fences(content: str) -> str:
    text = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    return fenced.group(1) if fenced else text


_SCHEMA: dict[str, Any] | None = None


def semantic_json_schema() -> dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None:
        schema = SemanticExtraction.model_json_schema()
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


# ---- Deterministic validation ----------------------------------------------------------

_NUMBER_RE = re.compile(r"(?<![\d.,/])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:\s+\d+/\d+)?|\d+/\d+|\.\d+)(?![\d/]|\.\d)")
_ALPHA_UNIT = r"millimet(?:er|re)s?|centimet(?:er|re)s?|met(?:er|re)s?|inch(?:es)?|feet|foot|mm|cm|in\.?|ft\.?|m"
_UNIT_TOKEN_RE = re.compile(
    rf"(?:(?<![a-z])({_ALPHA_UNIT})(?:(?![a-z])|(?=[wdhl](?![a-z])))|(''|\"|”|″|'|’|′))", re.IGNORECASE
)
_SEPARATOR_RE = re.compile(r"\s[x×*]\s|(?<=[\d\"”″'a-z)\]])\s*[x×*]\s*(?=\d)|\bby\b|[,;|]", re.IGNORECASE)
_AXIS_WORDS = {
    "width": re.compile(r"\bwidth\b|\bwide\b|(?<![a-z])w(?![a-z])", re.IGNORECASE),
    "depth": re.compile(r"\bdepth\b|\bdeep\b|(?<![a-z])d(?![a-z])", re.IGNORECASE),
    "height": re.compile(r"\bheight\b|\bhigh\b|\btall\b|(?<![a-z])h(?![a-z])", re.IGNORECASE),
    "length": re.compile(r"\blength\b|\blong\b|(?<![a-z])l(?![a-z])", re.IGNORECASE),
}
_NAME_AXIS_WORDS = {
    "width": re.compile(r"\bwidth\b|\bwide\b", re.I), "depth": re.compile(r"\bdepth\b|\bdeep\b", re.I),
    "height": re.compile(r"\bheight\b|\bhigh\b|\btall\b", re.I), "length": re.compile(r"\blength\b|\blong\b", re.I),
}
_ORDER_DECL_RE = re.compile(r"(?<![a-z])([WDHL])\s*[x×*]\s*([WDHL])(?:\s*[x×*]\s*([WDHL]))?(?![a-z])", re.IGNORECASE)
# Post-choice guard on the chosen field's *name/heading* only (never used to select evidence).
_NON_OVERALL_NAME_RE = re.compile(
    r"(?<![a-z])(package|packaging|packaged|shipping|shipment|carton|box|boxed|parcel|seat|arm|armrest|back|"
    r"backrest|leg|legs|shelf|shelves|drawer|interior|inside|inner|opening|cushion|clearance)(?![a-z])",
    re.IGNORECASE,
)


def canonical_unit(token: str) -> str | None:
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


def _normalize(text: str) -> str:
    for fancy in ("″", "”", "“"):
        text = text.replace(fancy, '"')
    return " ".join(text.replace(" ", " ").split())


@dataclass
class _Occurrence:
    value: float
    start: int
    end: int
    unit: str | None
    label: str | None  # width/depth/height/length from the text next to the number


def source_occurrences(entry: Entry) -> list[_Occurrence]:
    """Numbers in an entry with the unit and axis label written next to them."""
    name = _normalize(entry.name or "")
    value = _normalize(entry.value)
    prefix = f"{name}: " if name else ""
    text = prefix + value
    numbers = [m for m in _NUMBER_RE.finditer(text) if m.start() >= len(prefix)]
    occurrences = []
    for i, match in enumerate(numbers):
        number = parse_number(match.group(1).replace(",", ""))
        if number is None:
            continue
        next_start = numbers[i + 1].start() if i + 1 < len(numbers) else len(text)
        unit_match = _UNIT_TOKEN_RE.search(text, match.end(), next_start) or _UNIT_TOKEN_RE.search(text, match.end())
        unit = canonical_unit(unit_match.group(1) or unit_match.group(2)) if unit_match else None
        if unit is None and name:
            name_unit = _UNIT_TOKEN_RE.search(name)
            unit = canonical_unit(name_unit.group(1) or name_unit.group(2)) if name_unit else None
        seg_start = max([s.end() for s in _SEPARATOR_RE.finditer(text, len(prefix), match.start())] + [len(prefix)])
        seg_end_match = _SEPARATOR_RE.search(text, match.end())
        seg_end = min(seg_end_match.start() if seg_end_match else len(text), next_start)
        segment = text[seg_start:match.start()] + " " + text[match.end():seg_end]
        segment = _UNIT_TOKEN_RE.sub(" ", _NUMBER_RE.sub(" ", segment))
        labels = {axis for axis, rx in _AXIS_WORDS.items() if rx.search(segment)}
        occurrences.append(_Occurrence(number, match.start(), match.end(), unit,
                                       next(iter(labels)) if len(labels) == 1 else None))
    return occurrences


def validate_semantic(extraction: SemanticExtraction, rep: PageRepresentation) -> SemanticResult:
    result = SemanticResult(model_status=extraction.status, model_reason=extraction.reason)
    if extraction.status in ("unavailable", "uncertain"):
        result.errors.append(f"model answered {extraction.status}" + (f": {extraction.reason}" if extraction.reason else ""))
        return result
    dims = extraction.dimensions
    if not dims:
        result.errors.append("model returned no values")
        return result
    if len(dims) > 3:
        result.errors.append("more than three values returned")
        return result
    if rep.variant_mismatch:
        result.errors.append("page does not belong to the selected variant")
        return result

    located: list[tuple[SemanticDimension, Entry, Section]] = []
    for dim in dims:
        found = rep.entry(dim.sourceId)
        if found is None or (rep.included_ids and dim.sourceId not in rep.included_ids):
            result.errors.append(f"source {dim.sourceId} is not in the page data shown to the model")
            return result
        entry, section = found
        # Small local models occasionally point one row/block beside the exact value
        # while still selecting the correct specification section. Keep provenance
        # deterministic but hackathon-practical: repair the citation only when the
        # exact number+unit occurs in exactly one shown entry of that same section.
        if not _entry_has(dim, entry):
            raw_hint = _normalize(extraction.rawText or "").lower()
            alternatives = [
                candidate
                for candidate in section.entries + section.text
                if (not rep.included_ids or candidate.id in rep.included_ids)
                and (
                    candidate.path == entry.path
                    or (entry.name is not None and candidate.name == entry.name)
                    or (raw_hint and raw_hint in _normalize(candidate.text).lower())
                )
                and _entry_has(dim, candidate)
            ]
            if len(alternatives) == 1:
                result.notes.append(f"corrected adjacent source id {dim.sourceId} to {alternatives[0].id}")
                entry = alternatives[0]
        located.append((dim, entry, section))
    sections = {section.id for _, _, section in located}
    if len(sections) != 1:
        result.errors.append("values come from different sections")
        return result
    section = located[0][2]
    entries = list(dict.fromkeys(entry.id for _, entry, _ in located))
    entry_objs = {entry.id: entry for _, entry, _ in located}
    for entry in entry_objs.values():
        if _NON_OVERALL_NAME_RE.search(entry.name or "") or _NON_OVERALL_NAME_RE.search(section.heading):
            result.errors.append(f"chosen field '{entry.name or section.heading}' is not an overall product dimension")
            return result
    if section.scope in ("page",) and section.owner and len(rep.owners()) > 1:
        result.errors.append("chosen data belongs to one of several items on the page (variant not identified)")
        return result

    # Every value must occur (once each) in its cited entry, with the same unit.
    used: dict[str, set[int]] = {eid: set() for eid in entries}
    occurrence_for: list[_Occurrence] = []
    for index, (dim, entry, _) in enumerate(located):
        if dim.value <= 0:
            result.errors.append(f"value {index + 1}: not positive")
            return result
        match = None
        for occ in source_occurrences(entry):
            if occ.start in used[entry.id] or abs(occ.value - dim.value) > 1e-9 * max(1.0, abs(dim.value)):
                continue
            match = occ
            break
        if match is None:
            result.errors.append(f"value {dim.value:g} does not occur in {dim.sourceId}")
            return result
        if match.unit is None:
            result.errors.append(f"value {dim.value:g} in {dim.sourceId} has no unit in the source")
            return result
        if match.unit != dim.unit:
            result.errors.append(f"value {dim.value:g}: source unit is {match.unit}, model said {dim.unit}")
            return result
        meters = dim.value * UNIT_METERS[dim.unit]
        if not (MIN_M <= meters <= MAX_M):
            result.errors.append(f"value {dim.value:g} {dim.unit} is outside the plausible range")
            return result
        used[entry.id].add(match.start)
        occurrence_for.append(match)
        result.values_m.append(round(meters, 5))
        result.values_raw.append((dim.value, dim.unit))

    raw_sources = " | ".join(_normalize(entry_objs[e].value) for e in entries)
    if extraction.rawText and _normalize(extraction.rawText).lower() not in " ".join(
            _normalize(entry_objs[e].text) for e in entries).lower():
        result.notes.append("model rawText is not verbatim source text; using the source entry text")
    result.raw_text = raw_sources[:300]
    result.source_ids = entries
    result.source_type = section.kind
    result.scope = section.scope
    names = [entry_objs[e].name for e in entries if entry_objs[e].name]
    result.source_path = f"{section.heading} > {', '.join(names)}" if names else f"{section.heading} > text"
    result.axis = _axis_assignment(located, occurrence_for, extraction.axisMapping, entry_objs)
    if len(result.values_m) == 3 and any(index is None for index in (
            result.axis.widthIndex, result.axis.depthIndex, result.axis.heightIndex)):
        corroborated = _corroborating_labeled_axes(rep, result.values_raw, set(entries), section.scope)
        if corroborated is not None:
            result.axis, corroborating_path = corroborated
            result.notes.append(
                "the chosen measurements are duplicated with explicit axis labels at " + corroborating_path
            )
    result.accepted = True
    return result


def _entry_has(dim: SemanticDimension, entry: Entry) -> bool:
    return any(
        abs(occ.value - dim.value) <= 1e-9 * max(1.0, abs(dim.value)) and occ.unit == dim.unit
        for occ in source_occurrences(entry)
    )


def _corroborating_labeled_axes(
    rep: PageRepresentation,
    chosen_values: list[tuple[float, str]],
    chosen_ids: set[str],
    chosen_scope: str,
) -> tuple[AxisAssignment, str] | None:
    """Complete an unlabeled chosen triple from an exact labeled duplicate.

    This runs only *after* Qwen has selected and provenance-validated the overall
    measurements. It never discovers measurement candidates. A corroborating source
    must have one contiguous three-number group with exactly the same values/units and
    explicit W/D/H labels, must have been shown to Qwen, and must be at least as tightly
    variant-scoped as the chosen source. Conflicting duplicate mappings remain unresolved.
    """
    if len(chosen_values) != 3:
        return None
    scope_rank = {"exact_record": 0, "exact_page": 1, "family": 2, "page": 3}
    chosen_rank = scope_rank.get(chosen_scope, 3)
    candidates: list[tuple[int, tuple[int, int, int], str]] = []
    for entry, section in rep.all_entries():
        if entry.id in chosen_ids or (rep.included_ids and entry.id not in rep.included_ids):
            continue
        if scope_rank.get(section.scope, 3) > chosen_rank:
            continue
        if _NON_OVERALL_NAME_RE.search(entry.name or "") or _NON_OVERALL_NAME_RE.search(section.heading):
            continue
        occurrences = source_occurrences(entry)
        for start in range(max(0, len(occurrences) - 2)):
            window = occurrences[start:start + 3]
            if len(window) != 3 or {occ.label for occ in window} != {"width", "depth", "height"}:
                continue
            used: set[int] = set()
            mapping: dict[str, int] = {}
            for occ in window:
                matches = [
                    index for index, (value, unit) in enumerate(chosen_values)
                    if index not in used and unit == occ.unit
                    and abs(value - occ.value) <= 1e-9 * max(1.0, abs(value))
                ]
                if not matches:
                    break
                index = matches[0]
                used.add(index)
                mapping[occ.label] = index  # type: ignore[index]
            if len(mapping) == 3:
                permutation = (mapping["width"], mapping["depth"], mapping["height"])
                candidates.append((scope_rank.get(section.scope, 3), permutation, entry.path))
    if not candidates:
        return None
    best_rank = min(rank for rank, _, _ in candidates)
    best = [(permutation, path) for rank, permutation, path in candidates if rank == best_rank]
    permutations = {permutation for permutation, _ in best}
    if len(permutations) != 1:
        return None
    width, depth, height = next(iter(permutations))
    paths = sorted({path for permutation, path in best if permutation == (width, depth, height)})
    assignment = AxisAssignment(
        widthIndex=width,
        depthIndex=depth,
        heightIndex=height,
        confidence=1.0,
        reason="same verified measurements explicitly labeled W/D/H in corroborating source: " + paths[0],
        source="labels",
    )
    return assignment, paths[0]


def _axis_assignment(located, occurrences: list[_Occurrence], model: AxisMappingOut,
                     entries: dict[str, Entry]) -> AxisAssignment:
    """Axes only from labels in the source text; the model's mapping is advisory."""
    labels: list[str | None] = []
    for (dim, entry, _), occ in zip(located, occurrences):
        label = occ.label
        if label is None and len(located) > 1 and len({e.id for _, e, _ in located}) > 1:
            # separate fields: "Overall Width: 30 in"
            found = {axis for axis, rx in _NAME_AXIS_WORDS.items() if rx.search(entry.name or "")}
            label = next(iter(found)) if len(found) == 1 else None
        labels.append(label)
    if not any(labels):
        # a declared order such as "Dimensions (W x D x H)" in the (single) cited entry
        if len(entries) == 1:
            entry = next(iter(entries.values()))
            decl = _ORDER_DECL_RE.search(entry.name or "") or _ORDER_DECL_RE.search(entry.value)
            if decl:
                letters = [g.lower() for g in decl.groups() if g]
                if len(letters) == len(located):
                    order = sorted(range(len(occurrences)), key=lambda i: occurrences[i].start)
                    axis_of = {"w": "width", "d": "depth", "h": "height", "l": "length"}
                    labels = [None] * len(located)
                    for position, index in enumerate(order):
                        labels[index] = axis_of[letters[position]]
    assignment = AxisAssignment()
    if len(located) == 1 and labels[0] is None:
        found = {axis for axis, rx in _NAME_AXIS_WORDS.items() if rx.search(located[0][1].name or "")}
        if len(found) == 1:
            labels = [next(iter(found))]
    index_of: dict[str, int] = {}
    labels = ["depth" if label == "length" else label for label in labels]
    for i, label in enumerate(labels):
        if label:
            if label in index_of:
                index_of.pop(label)  # the same label twice: ambiguous
                labels[i] = None
            else:
                index_of[label] = i
    if len(located) == 3 and len(index_of) == 2 and "height" in index_of:
        rest = ({"width", "depth"} - set(index_of)).pop()
        index_of[rest] = ({0, 1, 2} - set(index_of.values())).pop()
    assignment.widthIndex = index_of.get("width")
    assignment.depthIndex = index_of.get("depth")
    assignment.heightIndex = index_of.get("height")
    mapped = sum(i is not None for i in (assignment.widthIndex, assignment.depthIndex, assignment.heightIndex))
    if mapped == 0:
        assignment.reason = "axis order not labeled in the source"
        assignment.source = "none"
    else:
        assignment.source = "labels"
        assignment.confidence = 1.0 if mapped == len(located) else 0.5
        assignment.reason = "axes labeled in the source" if mapped == len(located) else \
            "only some axes labeled in the source (" + ", ".join(k for k in ("width", "depth", "height") if k in index_of) + ")"
    claimed = {"width": model.widthIndex, "depth": model.depthIndex, "height": model.heightIndex}
    disagreements = [axis for axis, idx in claimed.items() if idx is not None and index_of.get(axis) != idx]
    if disagreements:
        assignment.reason += "; model mapping for " + ", ".join(disagreements) + " not supported by source labels (ignored)"
    return assignment

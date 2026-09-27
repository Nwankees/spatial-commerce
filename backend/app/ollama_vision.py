from __future__ import annotations

import base64
import copy
import json
import logging
import re
import time
from typing import Any

import httpx
from pydantic import ValidationError

from .models import Hypothesis, VisualProductAnalysis
from .vision_errors import (
    VisionMalformedResponseError,
    VisionNotConfiguredError,
    VisionTimeoutError,
    VisionUnavailableError,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the vision stage of a shopping search system. Your JSON output is parsed by
software and used to search real online stores for the product the user is pointing their phone at.
Be accurate and conservative: a wrong confident guess sends the shopper to the wrong product,
while an honest "unknown" is always acceptable."""

USER_PROMPT = """Look at the photo and describe the single main product the user is intentionally pointing at
(usually the most central, in-focus foreground object). Ignore people, walls, floors, and background clutter.

Report what will help find this exact product, or the closest real products, in online stores:
- category (broad) and subcategory (specific product type / form factor in shopper language).
- brand ONLY if a logo, printed name, or a very distinctive design supports it; give your confidence
  and the evidence type. Leave brand null if you cannot tell.
- modelFamily ONLY if printed text or unmistakable design supports it (product line, not a guessed
  exact model number). Leave null otherwise.
- visibleText: legible text/logos exactly as printed on the product. Empty if none is legible.
- color, apparent materials, style, and shape/form factor.
- distinctiveFeatures: the visual details that separate this product from similar ones.
- visibleSpecifications: ONLY values printed on the product or its label (e.g. text reading "65W").
  Never state dimensions, wattage, capacity, connector standards, or model numbers you cannot read.
- searchQueries: 3 to 5 diverse shopping queries, most specific first, e.g. one with the brand/line
  if known, one describing the type with its distinctive features, and one broader fallback.
  Do not put guessed specs or guessed model numbers in queries.
- confidence: how sure you are about the identification overall (0 to 1); uncertaintyNotes: what is unsure.

If no clear product is visible, set objectDetected to false, leave the other fields empty/null,
set confidence low, and give a short retry instruction in message.{context}
Return only JSON matching the schema."""


class OllamaVisionService:
    """Local visual product analysis through the Ollama HTTP API (/api/chat)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 180.0,
        keep_alive: str | None = "15m",
        think: bool | None = False,
        min_confidence: float = 0.2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._keep_alive = keep_alive
        self._think = think
        self._min_confidence = min_confidence
        self._transport = transport

    @property
    def model(self) -> str:
        return self._model

    @property
    def configured(self) -> bool:
        return bool(self._base_url and self._model)

    def build_request(self, image_bytes: bytes, user_request: str | None) -> dict[str, Any]:
        context = f"\nThe user's optional shopping intent: {user_request}" if user_request else ""
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(context=context),
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                },
            ],
            "stream": False,
            "format": analysis_json_schema(),
            "options": {"temperature": 0.1},
        }
        if self._think is not None:
            body["think"] = self._think
        if self._keep_alive:
            body["keep_alive"] = self._keep_alive
        return body

    async def analyze(self, image_bytes: bytes, user_request: str | None) -> VisualProductAnalysis:
        if not self.configured:
            raise VisionNotConfiguredError("OLLAMA_BASE_URL and OLLAMA_MODEL must be configured.")
        started = time.monotonic()
        outcome = "error"
        try:
            payload = await self._post(self.build_request(image_bytes, user_request))
            analysis = self._parse(payload)
            outcome = "ok" if analysis.objectDetected else "no_product"
            return analysis
        finally:
            logger.info(
                "Local vision analysis model=%s outcome=%s duration=%.1fs",
                self._model, outcome, time.monotonic() - started,
            )

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(f"{self._base_url}/api/chat", json=body)
        except httpx.TimeoutException:
            raise VisionTimeoutError("The local vision model timed out.") from None
        except httpx.HTTPError as exc:
            logger.warning("Ollama request failed: %s", type(exc).__name__)
            raise VisionUnavailableError("The local vision service (Ollama) is not reachable.") from None

        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.status_code != 200:
            error = payload.get("error") if isinstance(payload, dict) else None
            if response.status_code == 404 or (isinstance(error, str) and "not found" in error.lower()):
                raise VisionUnavailableError(f"The local vision model '{self._model}' is not installed in Ollama.")
            logger.warning("Ollama returned HTTP %s: %s", response.status_code, str(error)[:200])
            raise VisionUnavailableError(f"The local vision service returned HTTP {response.status_code}.")
        if not isinstance(payload, dict):
            raise VisionMalformedResponseError("The local vision service returned invalid JSON.")
        _log_timings(payload)
        return payload

    def _parse(self, payload: dict[str, Any]) -> VisualProductAnalysis:
        message = payload.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise VisionMalformedResponseError("The local vision model returned no content.")
        try:
            raw = json.loads(_strip_fences(content))
        except ValueError:
            raise VisionMalformedResponseError("The local vision model returned malformed JSON.") from None
        try:
            analysis = VisualProductAnalysis.model_validate(raw)
        except ValidationError:
            raise VisionMalformedResponseError(
                "The local vision model's output did not match the product schema."
            ) from None
        return enforce_conservative_analysis(analysis, self._min_confidence)


def enforce_conservative_analysis(analysis: VisualProductAnalysis, min_confidence: float) -> VisualProductAnalysis:
    """Deterministic guards applied after schema validation.

    - A detection below ``min_confidence`` becomes a safe "no product" result.
    - A detection with no product type and no queries is incomplete (retryable).
    - Visible specifications must be backed by legible visibleText; otherwise dropped.
    - Hypotheses claiming visible_text evidence must appear in visibleText; otherwise
      they are downgraded to design_resemblance (confidence kept, evidence corrected).
    """
    if not analysis.objectDetected:
        return analysis.model_copy(update={"searchQueries": [], "visibleSpecifications": []})
    if analysis.confidence < min_confidence:
        return analysis.model_copy(update={
            "objectDetected": False,
            "searchQueries": [],
            "message": analysis.message or "The product could not be identified confidently. Center it and retry.",
        })
    if not (analysis.category or analysis.subcategory) and not analysis.searchQueries:
        raise VisionMalformedResponseError("The local vision model returned an incomplete analysis.")

    text = _normalize(" ".join(analysis.visibleText))
    specs = [spec for spec in analysis.visibleSpecifications if _normalize(spec.value) and _normalize(spec.value) in text]
    return analysis.model_copy(update={
        "visibleSpecifications": specs,
        "brand": _check_hypothesis(analysis.brand, text),
        "modelFamily": _check_hypothesis(analysis.modelFamily, text),
    })


def _check_hypothesis(hypothesis: Hypothesis | None, visible_text: str) -> Hypothesis | None:
    if hypothesis is None:
        return None
    if hypothesis.evidence == "visible_text" and _normalize(hypothesis.value) not in visible_text:
        return hypothesis.model_copy(update={"evidence": "design_resemblance"})
    return hypothesis


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _strip_fences(content: str) -> str:
    text = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    return fenced.group(1) if fenced else text


def _log_timings(payload: dict[str, Any]) -> None:
    def seconds(key: str) -> float:
        value = payload.get(key)
        return value / 1e9 if isinstance(value, (int, float)) else 0.0

    logger.info(
        "Ollama timings total=%.1fs load=%.1fs prompt_eval=%.1fs eval=%.1fs eval_tokens=%s",
        seconds("total_duration"), seconds("load_duration"), seconds("prompt_eval_duration"),
        seconds("eval_duration"), payload.get("eval_count"),
    )


_SCHEMA_CACHE: dict[str, Any] | None = None


def analysis_json_schema() -> dict[str, Any]:
    """The VisualProductAnalysis JSON schema with $refs inlined for Ollama's `format`."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        schema = VisualProductAnalysis.model_json_schema()
        definitions = schema.pop("$defs", {})

        def inline(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node:
                    return inline(copy.deepcopy(definitions[node["$ref"].split("/")[-1]]))
                return {k: inline(v) for k, v in node.items() if k not in ("title", "additionalProperties")}
            if isinstance(node, list):
                return [inline(item) for item in node]
            return node

        schema = inline(schema)
        for node in _walk_objects(schema):
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                # Ask for every key (null/[] when unknown) so omissions are explicit.
                node["required"] = list(node["properties"])
        _SCHEMA_CACHE = schema
    return copy.deepcopy(_SCHEMA_CACHE)


def _walk_objects(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_objects(item)


async def ollama_health(base_url: str, model: str, timeout_seconds: float = 2.0,
                        transport: httpx.AsyncBaseTransport | None = None) -> dict[str, object]:
    """Checks Ollama reachability and whether the configured model is installed."""
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, transport=transport) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/tags")
        models = [m.get("name") for m in response.json().get("models", []) if isinstance(m, dict)]
    except Exception as exc:  # health must never fail
        return {"reachable": False, "modelInstalled": False, "error": type(exc).__name__}
    wanted = model if ":" in model else f"{model}:latest"
    return {"reachable": True, "modelInstalled": wanted in models or model in models, "installedModels": models[:20]}

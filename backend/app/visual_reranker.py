from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import math
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol

import httpx
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .product_image import ProductImageError, download_product_image
from .product_models import ProductCandidate
from .vision_errors import VisionMalformedResponseError, VisionTimeoutError, VisionUnavailableError

logger = logging.getLogger(__name__)


class CandidateVisualScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidateId: str
    sameProductProbability: float = Field(ge=0, le=1)
    visualSimilarity: float = Field(ge=0, le=1)
    categoryMatch: float = Field(ge=0, le=1)
    reason: str = Field(max_length=240)


class CandidateVisualScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[CandidateVisualScore]


@dataclass(frozen=True)
class VisualRerankResult:
    scores: dict[str, CandidateVisualScore]
    duration_ms: int
    compared: int


class VisualCandidateReranker(Protocol):
    async def rerank(
        self,
        original_image: bytes,
        candidates: list[ProductCandidate],
    ) -> VisualRerankResult: ...


class OllamaCandidateVisualReranker:
    """Bounded Qwen3-VL comparisons for a small merged shortlist.

    Each candidate is evaluated separately: the 8B model is reliable with two
    images, while a multi-image batch can consume the full context reasoning
    before emitting its schema-constrained JSON.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 180.0,
        max_candidates: int = 3,
        num_ctx: int = 16384,
        image_timeout_seconds: float = 10.0,
        keep_alive: str | None = "15m",
        transport: httpx.AsyncBaseTransport | None = None,
        image_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._max_candidates = max(1, min(max_candidates, 12))
        self._num_ctx = max(4096, num_ctx)
        self._image_timeout = image_timeout_seconds
        self._keep_alive = keep_alive
        self._transport = transport
        self._image_transport = image_transport

    async def rerank(
        self,
        original_image: bytes,
        candidates: list[ProductCandidate],
    ) -> VisualRerankResult:
        started = time.perf_counter()
        shortlist = [candidate for candidate in candidates if candidate.imageUrl][: self._max_candidates]
        logger.info(
            "Local visual rerank model=%s shortlist=%s",
            self._model,
            [{"candidate_id": candidate.id, "image_url": candidate.imageUrl} for candidate in shortlist],
        )
        downloaded = await asyncio.gather(
            *(download_product_image(
                candidate.imageUrl,
                timeout_seconds=self._image_timeout,
                transport=self._image_transport,
            ) for candidate in shortlist),
            return_exceptions=True,
        )
        usable: list[tuple[ProductCandidate, bytes]] = []
        for candidate, outcome in zip(shortlist, downloaded):
            if isinstance(outcome, ProductImageError):
                logger.warning(
                    "Local visual rerank image candidate=%s url=%s status=failed error_type=%s "
                    "reason=%s retryable=%s",
                    candidate.id,
                    candidate.imageUrl,
                    type(outcome).__name__,
                    str(outcome),
                    outcome.retryable,
                )
            elif isinstance(outcome, BaseException):
                logger.warning(
                    "Local visual rerank image candidate=%s url=%s status=failed error_type=%s reason=%s",
                    candidate.id,
                    candidate.imageUrl,
                    type(outcome).__name__,
                    str(outcome)[:300],
                )
            else:
                usable.append((candidate, outcome.data))
                logger.info(
                    "Local visual rerank image candidate=%s url=%s status=ok bytes=%s pixels=%sx%s "
                    "format=%s download_seconds=%s",
                    candidate.id,
                    candidate.imageUrl,
                    len(outcome.data),
                    outcome.width,
                    outcome.height,
                    outcome.format,
                    outcome.download_seconds,
                )
        if not usable:
            logger.warning(
                "Local visual rerank model=%s status=skipped reason=no_candidate_images_downloaded",
                self._model,
            )
            return VisualRerankResult({}, _elapsed_ms(started), 0)

        original = _prepare_comparison_image(original_image)
        scores: dict[str, CandidateVisualScore] = {}
        attempted = 0
        last_error: Exception | None = None
        deadline = time.monotonic() + self._timeout
        async with httpx.AsyncClient(transport=self._transport) as client:
            for candidate, data in usable:
                remaining = deadline - time.monotonic()
                if remaining <= 1:
                    last_error = VisionTimeoutError("The local visual reranker timed out.")
                    break
                prompt = f"""Image 1 is the user's original photographed object.
Image 2 is this shopping candidate:
id={candidate.id!r}; title={candidate.title!r}; retailer={candidate.retailer!r}

Compare the physical product itself, ignoring background, image angle, lighting, price
graphics, and packaging where possible. Return exactly one score object for this candidate:
- sameProductProbability: likelihood it is the exact same commercial product/model.
- visualSimilarity: similarity of silhouette, color, materials, construction, and distinctive features.
- categoryMatch: whether it is the same kind of product.
Use the exact candidateId supplied. Be conservative. Return only schema-valid JSON."""
                images = [original, _prepare_comparison_image(data)]
                body: dict[str, Any] = {
                    "model": self._model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are the visual matching stage of a shopping search system. "
                                "Return the requested schema immediately and do not include analysis or prose."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [base64.b64encode(image).decode("ascii") for image in images],
                        },
                    ],
                    "stream": False,
                    "format": visual_scores_json_schema(),
                    "think": False,
                    "options": {"temperature": 0.0, "num_ctx": self._num_ctx},
                }
                if self._keep_alive:
                    body["keep_alive"] = self._keep_alive
                logger.info(
                    "Local visual rerank Qwen request model=%s candidate=%s images=2 original_image_bytes=%s",
                    self._model,
                    candidate.id,
                    len(original_image),
                )
                attempted += 1
                try:
                    response = await client.post(
                        f"{self._base_url}/api/chat",
                        json=body,
                        timeout=max(1.0, remaining),
                    )
                    parsed = _parse_ollama_scores(response, self._model, {candidate.id})
                    scores.update(parsed)
                except (VisionMalformedResponseError, VisionTimeoutError, VisionUnavailableError) as exc:
                    last_error = exc
                    logger.warning(
                        "Local visual rerank candidate failed open model=%s candidate=%s type=%s reason=%s",
                        self._model,
                        candidate.id,
                        type(exc).__name__,
                        str(exc),
                    )
                    continue
                except httpx.TimeoutException:
                    last_error = VisionTimeoutError("The local visual reranker timed out.")
                    break
                except httpx.HTTPError as exc:
                    last_error = VisionUnavailableError(
                        f"The local visual reranker is unreachable ({type(exc).__name__})."
                    )
                    break
        if not scores and last_error is not None:
            raise last_error
        logger.info(
            "Local visual rerank Qwen success model=%s compared=%s returned_scores=%s duration_ms=%s",
            self._model,
            attempted,
            len(scores),
            _elapsed_ms(started),
        )
        return VisualRerankResult(scores, _elapsed_ms(started), attempted)


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _parse_ollama_scores(
    response: httpx.Response,
    model: str,
    allowed_ids: set[str],
) -> dict[str, CandidateVisualScore]:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if response.status_code != 200 or not isinstance(payload, dict):
        remote_reason = payload.get("error") if isinstance(payload, dict) else None
        reason = f"The local visual reranker returned HTTP {response.status_code}"
        reason += f": {' '.join(remote_reason.split())[:300]}" if isinstance(remote_reason, str) and remote_reason.strip() else "."
        raise VisionUnavailableError(reason)
    message = payload.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        thinking = message.get("thinking") if isinstance(message, dict) else None
        logger.warning(
            "Local visual rerank Qwen failed model=%s reason=response_missing_content "
            "done_reason=%r prompt_eval_count=%r eval_count=%r message_keys=%s thinking_preview=%r",
            model,
            payload.get("done_reason"),
            payload.get("prompt_eval_count"),
            payload.get("eval_count"),
            sorted(message) if isinstance(message, dict) else [],
            thinking[:600] if isinstance(thinking, str) else None,
        )
        raise VisionMalformedResponseError("The local visual reranker returned no content.")
    try:
        return parse_visual_scores(content, allowed_ids)
    except VisionMalformedResponseError as exc:
        logger.warning(
            "Local visual rerank Qwen failed model=%s reason=%s content_length=%s content_preview=%r",
            model,
            str(exc),
            len(content),
            content[:1200],
        )
        raise


def visual_scores_json_schema() -> dict[str, Any]:
    """Inline Pydantic references for Ollama's schema-constrained format."""
    schema = CandidateVisualScores.model_json_schema()
    definitions = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(copy.deepcopy(definitions[node["$ref"].split("/")[-1]]))
            return {key: inline(value) for key, value in node.items() if key not in ("title", "default")}
        if isinstance(node, list):
            return [inline(value) for value in node]
        return node

    return inline(schema)


def parse_visual_scores(content: str, allowed_ids: set[str]) -> dict[str, CandidateVisualScore]:
    try:
        raw = json.loads(content)
        if isinstance(raw, dict) and isinstance(raw.get("scores"), list):
            for score in raw["scores"]:
                if not isinstance(score, dict):
                    continue
                for field in ("sameProductProbability", "visualSimilarity", "categoryMatch"):
                    value = score.get(field)
                    if isinstance(value, (int, float)) and math.isfinite(value) and -1e-9 <= value <= 1 + 1e-9:
                        score[field] = min(1.0, max(0.0, float(value)))
        parsed = CandidateVisualScores.model_validate(raw)
    except (ValueError, ValidationError):
        raise VisionMalformedResponseError("The local visual reranker returned malformed scores.") from None
    return {score.candidateId: score for score in parsed.scores if score.candidateId in allowed_ids}


def _prepare_comparison_image(data: bytes, max_side: int = 256) -> bytes:
    """Bounds a candidate image before including several of them in one VL call."""
    try:
        with Image.open(BytesIO(data)) as source:
            source.load()
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise VisionMalformedResponseError("A visual-rerank image was unreadable.") from exc
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    output = BytesIO()
    image.save(output, "JPEG", quality=82, optimize=True)
    return output.getvalue()

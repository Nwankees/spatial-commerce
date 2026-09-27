"""Replaceable image-to-3D reconstruction providers.

The main backend never imports a reconstruction model. Providers are reached over
a local HTTP boundary so heavy GPU dependencies live in their own environment.
"""
from __future__ import annotations

import base64
import binascii
import logging
from dataclasses import dataclass, field
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class ReconstructionError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, code: str = "failed") -> None:
        super().__init__(message)
        self.retryable = retryable
        self.code = code


@dataclass
class ReconstructionOutput:
    glb: bytes
    processed_image_png: bytes | None = None
    timings: dict[str, float] = field(default_factory=dict)
    provider_meta: dict[str, object] = field(default_factory=dict)


class ReconstructionProvider(Protocol):
    name: str
    version: str

    async def health(self) -> dict[str, object]: ...

    async def reconstruct(self, image_bytes: bytes) -> ReconstructionOutput: ...


class SidecarReconstructionProvider:
    """Talks to the local reconstruction service (reconstruction_service/server.py).

    Wire format (POST {base}/reconstruct, multipart field "image"):
      200 {"glbBase64", "processedImageBase64"?, "timings": {...}, "model", "version"}
      4xx/5xx {"code": "gpu_out_of_memory" | "invalid_image" | "reconstruction_failed" | ..., "detail"}
    """

    def __init__(self, base_url: str, *, name: str = "sf3d", version: str = "stabilityai/stable-fast-3d",
                 timeout_seconds: float = 180.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self.name = name
        self.version = version
        self._timeout = timeout_seconds
        self._transport = transport

    async def health(self) -> dict[str, object]:
        try:
            async with httpx.AsyncClient(timeout=3.0, transport=self._transport) as client:
                response = await client.get(f"{self._base_url}/health")
            body = response.json() if response.status_code == 200 else {}
            return {"reachable": response.status_code == 200, **(body if isinstance(body, dict) else {})}
        except Exception as exc:
            return {"reachable": False, "error": type(exc).__name__}

    async def reconstruct(self, image_bytes: bytes) -> ReconstructionOutput:
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(
                    f"{self._base_url}/reconstruct",
                    files={"image": ("product-image", image_bytes, "application/octet-stream")},
                )
        except httpx.TimeoutException:
            raise ReconstructionError("3D reconstruction timed out.", retryable=True, code="timeout") from None
        except httpx.HTTPError as exc:
            logger.warning("Reconstruction service unreachable: %s", type(exc).__name__)
            raise ReconstructionError("The local 3D reconstruction service is not running.", retryable=True,
                                      code="service_unavailable") from None
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code != 200:
            code = body.get("code") if isinstance(body, dict) else None
            detail = body.get("detail") if isinstance(body, dict) else None
            if code == "gpu_out_of_memory":
                raise ReconstructionError("The GPU ran out of memory while generating the 3D preview.",
                                          retryable=True, code=code)
            if code == "invalid_image":
                raise ReconstructionError(f"The product image could not be used: {detail or 'invalid image'}.",
                                          retryable=False, code=code)
            if code == "model_not_ready":
                raise ReconstructionError("The 3D model is still loading on the reconstruction service.",
                                          retryable=True, code=code)
            raise ReconstructionError(f"3D reconstruction failed (HTTP {response.status_code}).",
                                      retryable=response.status_code >= 500, code=code or "failed")
        if not isinstance(body, dict) or not isinstance(body.get("glbBase64"), str):
            raise ReconstructionError("The reconstruction service returned an invalid response.", retryable=True,
                                      code="malformed_response")
        try:
            glb = base64.b64decode(body["glbBase64"], validate=True)
            processed = base64.b64decode(body["processedImageBase64"], validate=True) \
                if isinstance(body.get("processedImageBase64"), str) else None
        except (binascii.Error, ValueError):
            raise ReconstructionError("The reconstruction service returned undecodable data.", retryable=True,
                                      code="malformed_response") from None
        timings = {k: float(v) for k, v in (body.get("timings") or {}).items() if isinstance(v, (int, float))}
        meta = {k: body[k] for k in ("model", "version", "device", "peakMemoryMb") if k in body}
        return ReconstructionOutput(glb, processed, timings, meta)

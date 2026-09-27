"""Local SF3D reconstruction sidecar. Run with run_service.ps1 (listens on 127.0.0.1 only)."""
from __future__ import annotations

import asyncio
import base64
import logging

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from sf3d_runner import InvalidImage, Sf3dRunner

MAX_UPLOAD_BYTES = 15_000_000
logger = logging.getLogger("reconstruction")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")
app = FastAPI(title="Spatial Commerce reconstruction service", docs_url=None, redoc_url=None)
runner = Sf3dRunner()


@app.on_event("startup")
async def warm_up() -> None:
    # Load the model in the background so /health answers immediately.
    asyncio.get_running_loop().run_in_executor(None, _safe_load)


def _safe_load() -> None:
    try:
        runner.load()
        logger.info("SF3D loaded: %s", runner.info())
    except Exception:
        logger.exception("SF3D failed to load")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", **runner.info()}


@app.post("/reconstruct")
async def reconstruct(image: UploadFile = File(...)):
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not data or len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=400, content={"code": "invalid_image", "detail": "empty or too large"})
    try:
        result = await asyncio.get_running_loop().run_in_executor(None, runner.reconstruct, data)
    except InvalidImage as exc:
        return JSONResponse(status_code=400, content={"code": "invalid_image", "detail": str(exc)})
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning("GPU out of memory during reconstruction")
        return JSONResponse(status_code=507, content={"code": "gpu_out_of_memory", "detail": "GPU out of memory"})
    except Exception as exc:
        logger.exception("Reconstruction failed")
        if not runner.ready:
            return JSONResponse(status_code=503, content={"code": "model_not_ready", "detail": type(exc).__name__})
        return JSONResponse(status_code=500, content={"code": "reconstruction_failed", "detail": type(exc).__name__})
    logger.info("Reconstruction ok: %s glb=%d bytes", result["timings"], len(result["glb"]))
    return {
        "glbBase64": base64.b64encode(result["glb"]).decode("ascii"),
        "processedImageBase64": base64.b64encode(result["processed_png"]).decode("ascii"),
        "timings": result["timings"],
        "peakMemoryMb": result["peakMemoryMb"],
        **{k: v for k, v in runner.info().items() if k in ("model", "version", "device")},
    }

"""Loads Stable Fast 3D once and turns one product image into a textured GLB.

Runs inside the isolated SF3D environment (Python 3.11, CUDA PyTorch). Uses SF3D's
own preprocessing utilities (rembg background removal + resize_foreground).
"""
from __future__ import annotations

import io
import os
import threading
import time
from contextlib import nullcontext
from typing import Any

import torch
from PIL import Image

MODEL_ID = os.environ.get("SF3D_MODEL", "stabilityai/stable-fast-3d")
FOREGROUND_RATIO = float(os.environ.get("SF3D_FOREGROUND_RATIO", "0.85"))
TEXTURE_RESOLUTION = int(os.environ.get("SF3D_TEXTURE_RESOLUTION", "1024"))
MIN_IMAGE_SIDE = 64


class InvalidImage(ValueError):
    pass


def _cuda_kernel_registered(op: str) -> bool:
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(op, "CUDA")
    except Exception:
        return False


class _CpuBakerAdapter(torch.nn.Module):
    """If texture_baker was built without CUDA kernels (CPU-only build on Windows),
    run its ops on CPU and move results back. SF3D's code is not modified.
    An nn.Module because SF3D registers the baker as a child module."""

    def __init__(self, baker: Any) -> None:
        super().__init__()
        self._baker = baker

    def rasterize(self, uv, face_indices, bake_resolution):
        return self._baker.rasterize(uv.cpu(), face_indices.cpu(), bake_resolution).to(uv.device)

    def get_mask(self, rast):
        return self._baker.get_mask(rast)

    def interpolate(self, attr, rast, face_indices):
        return self._baker.interpolate(attr.cpu(), rast.cpu(), face_indices.cpu()).to(attr.device)


class Sf3dRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model = None
        self.rembg_session = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.baker_mode = "unknown"
        self.load_seconds: float | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        with self._lock:
            if self.model is not None:
                return
            started = time.perf_counter()
            import rembg
            from sf3d.system import SF3D

            model = SF3D.from_pretrained(MODEL_ID, config_name="config.yaml", weight_name="model.safetensors")
            model.to(self.device)
            model.eval()
            if self.device == "cuda" and not _cuda_kernel_registered("texture_baker_cpp::rasterize"):
                model.baker = _CpuBakerAdapter(model.baker)
                self.baker_mode = "cpu (texture_baker built without CUDA)"
            else:
                self.baker_mode = self.device
            self.rembg_session = rembg.new_session()
            self.model = model
            self.load_seconds = round(time.perf_counter() - started, 2)

    def info(self) -> dict[str, Any]:
        return {
            "model": "sf3d",
            "version": MODEL_ID,
            "device": self.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "cudaArchList": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
            "bakerMode": self.baker_mode,
            "modelLoaded": self.ready,
            "loadSeconds": self.load_seconds,
            "textureResolution": TEXTURE_RESOLUTION,
        }

    def reconstruct(self, image_bytes: bytes) -> dict[str, Any]:
        from sf3d.utils import remove_background, resize_foreground

        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                source.load()
                image = source.convert("RGBA")
        except Exception as exc:
            raise InvalidImage(f"unreadable image ({type(exc).__name__})") from None
        if min(image.size) < MIN_IMAGE_SIDE:
            raise InvalidImage(f"image too small ({image.size[0]}x{image.size[1]})")
        self.load()
        with self._lock:  # one GPU job at a time
            image = remove_background(image, self.rembg_session)
            alpha = image.getchannel("A")
            if alpha.getbbox() is None:
                raise InvalidImage("background removal left no foreground object")
            image = resize_foreground(image, FOREGROUND_RATIO)
            processed = io.BytesIO()
            image.save(processed, format="PNG")
            timings["preprocessSeconds"] = round(time.perf_counter() - t0, 3)

            t1 = time.perf_counter()
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
            try:
                with torch.no_grad(), autocast:
                    mesh, _ = self.model.run_image(image, bake_resolution=TEXTURE_RESOLUTION, remesh="none", vertex_count=-1)
            finally:
                peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if self.device == "cuda" else None
            timings["reconstructionSeconds"] = round(time.perf_counter() - t1, 3)

            t2 = time.perf_counter()
            glb = mesh.export(file_type="glb", include_normals=True)
            timings["exportSeconds"] = round(time.perf_counter() - t2, 3)
        timings["totalSeconds"] = round(time.perf_counter() - t0, 3)
        return {"glb": glb, "processed_png": processed.getvalue(), "timings": timings,
                "peakMemoryMb": round(peak, 1) if peak else None}

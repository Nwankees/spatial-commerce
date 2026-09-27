# Reconstruction service (Milestone 6)

A local sidecar that runs [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d)
(SF3D) to turn one product image into a textured GLB. It runs in its own Python 3.11
environment so its pinned dependencies never touch `backend\.venv`. The backend calls it
at `http://127.0.0.1:8010` (`RECONSTRUCTION_SERVICE_URL`).

Nothing large lives in this repository: the venv, SF3D source, model weights, Hugging Face
and rembg caches, build temp files and outputs all live under `D:\SpatialCommerce`.

## Requirements

- Windows, NVIDIA GPU with a recent driver (developed on an RTX 5080, sm_120 → PyTorch cu128).
- Visual Studio Build Tools with the C++ workload (`vcvars64.bat`; pass `-VcVars` if it is not
  the VS 2026 Community path in `setup_sf3d.ps1`).
- `py -3.12` (only to bootstrap `uv`), git, ~15 GB free on D:.
- A Hugging Face account that has accepted the SF3D license (gated model).

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File reconstruction_service\setup_sf3d.ps1
# accept the license at https://huggingface.co/stabilityai/stable-fast-3d, create a read token, then:
powershell -ExecutionPolicy Bypass -File reconstruction_service\login_and_verify.ps1
```

The login prompt is interactive; type the token yourself. It is stored under
`D:\SpatialCommerce\hf` and never in this repository. `login_and_verify.ps1` runs SF3D once on the
bundled demo chair and writes `D:\SpatialCommerce\out\chair1.glb`.

Build notes: SF3D's `texture_baker` is built CPU-only with MSVC `/openmp` (its CUDA kernel is not
built on Windows) and is wrapped so the rest of SF3D runs on the GPU; `rembg` uses the CPU
onnxruntime to avoid CUDA runtime conflicts with PyTorch.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File reconstruction_service\run_service.ps1
```

`GET /health` reports `loading | ready | error`; `POST /reconstruct` takes multipart `image`
and returns the GLB (base64), the preprocessed image, timings, device and peak GPU memory.
Errors are typed: `invalid_image` (400), `model_not_ready` (503), `gpu_out_of_memory` (507),
`reconstruction_failed` (500).

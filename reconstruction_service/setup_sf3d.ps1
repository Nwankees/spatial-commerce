# One-time setup of the isolated Stable Fast 3D (SF3D) environment on D:.
# Nothing here touches backend\.venv. Re-running is safe (steps are skipped when done).
#   powershell -ExecutionPolicy Bypass -File reconstruction_service\setup_sf3d.ps1
param(
    [string]$Root = "D:\SpatialCommerce",
    [string]$Sf3dCommit = "ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2",
    [string]$VcVars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$log = Join-Path $repo "tmp\m6-sf3d-setup.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
Start-Transcript -Path $log -Force | Out-Null
try {
    foreach ($d in "tools", "python", "uv-cache", "pip-cache", "hf", "u2net", "tmp", "out") { New-Item -ItemType Directory -Force -Path "$Root\$d" | Out-Null }
    # Keep all downloads/caches/build temp on D: (C: is nearly full).
    $env:UV_CACHE_DIR = "$Root\uv-cache"; $env:UV_PYTHON_INSTALL_DIR = "$Root\python"; $env:PIP_CACHE_DIR = "$Root\pip-cache"
    $env:HF_HOME = "$Root\hf"; $env:U2NET_HOME = "$Root\u2net"; $env:TMP = "$Root\tmp"; $env:TEMP = "$Root\tmp"
    $venv = "$Root\sf3d-venv"; $py = "$venv\Scripts\python.exe"; $src = "$Root\stable-fast-3d"

    "== 1. uv (in its own tools venv)"
    if (-not (Test-Path "$Root\tools\venv\Scripts\uv.exe")) {
        py -3.12 -m venv "$Root\tools\venv"; & "$Root\tools\venv\Scripts\python.exe" -m pip install -q uv
    }
    $uv = "$Root\tools\venv\Scripts\uv.exe"; & $uv --version

    "== 2. Python 3.11 + SF3D venv (gpytoolbox==0.2.0 has Windows wheels only up to 3.11)"
    & $uv python install 3.11
    if (-not (Test-Path $py)) { & $uv venv $venv --python 3.11 }
    & $py --version

    "== 3. SF3D source at pinned commit"
    if (-not (Test-Path "$src\.git")) { git clone https://github.com/Stability-AI/stable-fast-3d $src }
    git -C $src fetch -q origin; git -C $src checkout -q $Sf3dCommit; git -C $src log -1 --format="%H %cd"

    "== 4. PyTorch 2.7.1 + CUDA 12.8 (RTX 50-series / sm_120 needs cu128+)"
    & $uv pip install -p $py "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
    & $py -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, torch.cuda.get_arch_list())"

    "== 5. SF3D requirements (rembg CPU build to avoid onnxruntime-gpu CUDA conflicts) + service deps"
    $reqs = Get-Content "$src\requirements.txt" | Where-Object { $_ -and $_ -notmatch '^\./' -and $_ -notmatch '^rembg' }
    & $uv pip install -p $py "setuptools==69.5.1" wheel ninja "rembg==2.0.57" @reqs -r "$PSScriptRoot\requirements-service.txt" --constraint (New-Item -Force -Path "$Root\tmp\torch-constraint.txt" -Value "torch==2.7.1`ntorchvision==0.22.1").FullName --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match

    "== 6. Build SF3D C++ extensions with MSVC (texture_baker CPU build + /openmp; uv_unwrapper CPU)"
    if (-not (Test-Path $VcVars)) { throw "vcvars64.bat not found at $VcVars" }
    cmd /c "`"$VcVars`" >nul && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] } }
    $env:DISTUTILS_USE_SDK = "1"; $env:USE_CUDA = "0"; $env:USE_NATIVE_ARCH = "0"; $env:CL = "/openmp"
    & $uv pip install -p $py --no-build-isolation "$src\texture_baker" "$src\uv_unwrapper"
    Remove-Item Env:CL

    "== 7. Import check"
    Push-Location $src
    & $py -c "import torch, texture_baker, uv_unwrapper, rembg, sf3d.system; print('imports ok; texture_baker CUDA kernel:', torch._C._dispatch_has_kernel_for_dispatch_key('texture_baker_cpp::rasterize','CUDA'))"
    Pop-Location
    "SETUP-OK (next: accept the SF3D license on Hugging Face and log in; see reconstruction_service\README.md)"
} catch {
    "ERROR: $_"
} finally {
    Stop-Transcript | Out-Null
}

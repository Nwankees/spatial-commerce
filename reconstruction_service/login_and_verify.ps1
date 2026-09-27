# After accepting the SF3D license at https://huggingface.co/stabilityai/stable-fast-3d :
#   powershell -ExecutionPolicy Bypass -File reconstruction_service\login_and_verify.ps1
# Prompts for your Hugging Face read token (typed by you; stored under D:\SpatialCommerce\hf),
# downloads the weights to D:, and runs SF3D once on its own demo chair.
param([string]$Root = "D:\SpatialCommerce")
$repo = Split-Path -Parent $PSScriptRoot
$env:HF_HOME = "$Root\hf"; $env:U2NET_HOME = "$Root\u2net"; $env:TMP = "$Root\tmp"; $env:TEMP = "$Root\tmp"
$env:PYTHONPATH = "$Root\stable-fast-3d;$PSScriptRoot"
$py = "$Root\sf3d-venv\Scripts\python.exe"
if (-not (Test-Path "$Root\hf\token")) { & "$Root\sf3d-venv\Scripts\huggingface-cli.exe" login }
Start-Transcript -Path (Join-Path $repo "tmp\m6-sf3d-verify.log") -Force | Out-Null
$ErrorActionPreference = "Continue"
# Pipe native output through PowerShell so the transcript records it (and stderr), then check exit codes.
function Native([scriptblock]$b) { $global:LASTEXITCODE = 0; & $b 2>&1 | ForEach-Object { "$_" }; if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" } }
try {
    Native { & $py -X faulthandler -c "from huggingface_hub import hf_hub_download as d; print(d('stabilityai/stable-fast-3d','config.yaml')); print(d('stabilityai/stable-fast-3d','model.safetensors'))" }
    Native { & $py -X faulthandler "$PSScriptRoot\verify_sf3d.py" "$Root\stable-fast-3d\demo_files\examples\chair1.png" "$Root\out\chair1.glb" }
    "VERIFY-OK"
} catch {
    "VERIFY-ERROR: $_"
} finally { Stop-Transcript | Out-Null }

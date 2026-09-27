# Starts the SF3D reconstruction sidecar on http://127.0.0.1:8010 (local only).
#   powershell -ExecutionPolicy Bypass -File reconstruction_service\run_service.ps1
param([string]$Root = "D:\SpatialCommerce", [int]$Port = 8010)
$env:HF_HOME = "$Root\hf"; $env:U2NET_HOME = "$Root\u2net"; $env:TMP = "$Root\tmp"; $env:TEMP = "$Root\tmp"
$env:HF_HUB_OFFLINE = if (Test-Path "$Root\hf\hub\models--stabilityai--stable-fast-3d") { "1" } else { "0" }
$env:PYTHONPATH = "$Root\stable-fast-3d;$PSScriptRoot"
Set-Location $PSScriptRoot
& "$Root\sf3d-venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port $Port

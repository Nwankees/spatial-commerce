$ErrorActionPreference = "Stop"
$helperRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $helperRoot ".venv"

if (-not (Test-Path $venv)) {
    py -3.12 -m venv $venv
}
& "$venv\Scripts\python.exe" -m pip install --upgrade pip
& "$venv\Scripts\python.exe" -m pip install -r "$helperRoot\requirements.txt"
& "$venv\Scripts\python.exe" -m playwright install chromium
Write-Host "Dimension helper ready. Run .\run.ps1"

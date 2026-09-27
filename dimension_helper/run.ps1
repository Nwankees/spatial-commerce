$ErrorActionPreference = "Stop"
$helperRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $helperRoot
try {
    & ".\.venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8020
} finally {
    Pop-Location
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Set-Location $ProjectRoot
if (Test-Path $VenvPython) {
    & $VenvPython -m uvicorn backend.app.main:app --reload --port 8000
} else {
    python -m uvicorn backend.app.main:app --reload --port 8000
}

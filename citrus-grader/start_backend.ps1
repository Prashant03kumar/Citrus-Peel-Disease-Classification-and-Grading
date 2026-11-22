$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$ActivateScript = Join-Path $ProjectRoot '.venv\Scripts\Activate.ps1'

if (-not (Test-Path $ActivateScript)) {
    throw "Project virtual environment not found. Create it with: py -3.11 -m venv .venv"
}

$healthUrl = 'http://127.0.0.1:8000/api/health'
if (Test-NetConnection -ComputerName '127.0.0.1' -Port 8000 -InformationLevel Quiet) {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
    if ($health.status -eq 'ok' -and $health.model_loaded) {
        Write-Host 'A healthy Citrus Grader API is already running on port 8000.'
        return
    }
    throw 'Port 8000 is already in use by a service that is not a ready Citrus Grader API.'
}

Set-Location $ProjectRoot
. $ActivateScript
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) {
    throw "Backend server exited with code $LASTEXITCODE."
}

$ErrorActionPreference = 'Stop'
$FrontendRoot = Join-Path $PSScriptRoot 'frontend'
$frontendUrl = 'http://localhost:5173'

if (-not (Test-Path (Join-Path $FrontendRoot 'node_modules'))) {
    throw 'Frontend dependencies are missing. Run npm install from the frontend folder first.'
}

if (Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue) {
    try {
        $response = Invoke-WebRequest -Uri $frontendUrl -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200 -and $response.Content -match '<div id="root"></div>') {
            Write-Host "A Citrus Grader frontend is already responding at $frontendUrl."
            return
        }
    } catch {
        throw 'Port 5173 is already in use by a service that is not the Citrus Grader frontend.'
    }

    throw 'Port 5173 is already in use by a service that is not the Citrus Grader frontend.'
}

Set-Location $FrontendRoot
npm run dev -- --port 5173 --strictPort
if ($LASTEXITCODE -ne 0) {
    throw "Frontend server exited with code $LASTEXITCODE."
}

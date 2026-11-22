$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$PowerShellExe = Join-Path $PSHOME 'powershell.exe'
$BackendScript = Join-Path $ProjectRoot 'start_backend.ps1'
$FrontendScript = Join-Path $ProjectRoot 'start_frontend.ps1'

$backendArguments = '-NoExit -ExecutionPolicy Bypass -File "{0}"' -f $BackendScript
$frontendArguments = '-NoExit -ExecutionPolicy Bypass -File "{0}"' -f $FrontendScript

Start-Process -FilePath $PowerShellExe -ArgumentList $backendArguments -WorkingDirectory $ProjectRoot
Start-Process -FilePath $PowerShellExe -ArgumentList $frontendArguments -WorkingDirectory $ProjectRoot

Write-Host ''
Write-Host 'Citrus Grader is starting in two PowerShell windows.'
Write-Host 'Dashboard:  http://localhost:5173'
Write-Host 'API docs:   http://localhost:8000/docs'

# Project NOIR Windows Launch Script
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not detected. Running setup.ps1 first..." -ForegroundColor Yellow
    & .\setup.ps1
}

& .\.venv\Scripts\Activate.ps1
python -m noir.main $args

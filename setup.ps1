# Project NOIR Windows Setup Script
param (
    [switch]$Dev = $false
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "             PROJECT NOIR - SETUP SYSTEM                  " -ForegroundColor Yellow
Write-Host "   Real-Time AI & Machine Learning Research Environment   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python installation
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Green
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Gray
} catch {
    Write-Host "Error: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ and re-run this script." -ForegroundColor Red
    exit 1
}

# 2. Create Virtual Environment
Write-Host "[2/6] Setting up virtual environment (.venv)..." -ForegroundColor Green
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "  Virtual environment created." -ForegroundColor Gray
} else {
    Write-Host "  Virtual environment already exists." -ForegroundColor Gray
}

# 3. Activate Virtual Environment
Write-Host "[3/6] Activating virtual environment..." -ForegroundColor Green
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "  Activated .venv" -ForegroundColor Gray
} else {
    Write-Host "Error: Activation script not found at $activateScript" -ForegroundColor Red
    exit 1
}

# 4. Upgrade pip
Write-Host "[4/6] Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# 5. Install Dependencies
Write-Host "[5/6] Installing dependencies..." -ForegroundColor Green
if ($Dev) {
    Write-Host "  Installing core + development dependencies from requirements-dev.txt..." -ForegroundColor Gray
    pip install -r requirements-dev.txt
} else {
    Write-Host "  Installing core dependencies from requirements.txt..." -ForegroundColor Gray
    pip install -r requirements.txt
}

# 6. Create Required Directories & .env
Write-Host "[6/6] Ensuring project directories exist..." -ForegroundColor Green
$dirs = @("experiments", "data", "checkpoints", "logs", "memory", "config")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        New-Item -ItemType File -Path (Join-Path $dir ".gitkeep") -Force | Out-Null
        Write-Host "  Created directory: $dir" -ForegroundColor Gray
    }
}

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "  Created .env from .env.example" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "         PROJECT NOIR SETUP COMPLETED SUCCESSFULLY!       " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "To launch Project NOIR, run:" -ForegroundColor Yellow
Write-Host "  .\run.ps1" -ForegroundColor White
Write-Host ""

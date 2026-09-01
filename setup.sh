#!/usr/bin/env bash
# Project NOIR Linux/macOS Setup Script
set -e

DEV_MODE=false
for arg in "$@"; do
    if [ "$arg" == "--dev" ] || [ "$arg" == "-d" ]; then
        DEV_MODE=true
    fi
done

echo "=========================================================="
echo "             PROJECT NOIR - SETUP SYSTEM                  "
echo "   Real-Time AI & Machine Learning Research Environment   "
echo "=========================================================="

# 1. Check Python installation
echo "[1/6] Checking Python installation..."
if command -v python3 &>/dev/null; then
    PYTHON_BIN=python3
elif command -v python &>/dev/null; then
    PYTHON_BIN=python
else
    echo "Error: Python 3 is not installed or not in PATH."
    exit 1
fi
echo "  Found: $($PYTHON_BIN --version)"

# 2. Create Virtual Environment
echo "[2/6] Setting up virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    $PYTHON_BIN -m venv .venv
    echo "  Virtual environment created."
else
    echo "  Virtual environment already exists."
fi

# 3. Activate Virtual Environment
echo "[3/6] Activating virtual environment..."
source .venv/bin/activate
echo "  Activated .venv"

# 4. Upgrade pip
echo "[4/6] Upgrading pip..."
python -m pip install --upgrade pip

# 5. Install Dependencies
echo "[5/6] Installing dependencies..."
if [ "$DEV_MODE" = true ]; then
    echo "  Installing core + development dependencies from requirements-dev.txt..."
    pip install -r requirements-dev.txt
else
    echo "  Installing core dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

# 6. Create Required Directories & .env
echo "[6/6] Ensuring project directories exist..."
mkdir -p experiments data checkpoints logs memory config
touch experiments/.gitkeep data/.gitkeep checkpoints/.gitkeep logs/.gitkeep memory/.gitkeep config/.gitkeep

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
fi

echo ""
echo "=========================================================="
echo "         PROJECT NOIR SETUP COMPLETED SUCCESSFULLY!       "
echo "=========================================================="
echo "To launch Project NOIR, run:"
echo "  ./run.sh"
echo ""

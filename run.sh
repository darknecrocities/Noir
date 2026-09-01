#!/usr/bin/env bash
# Project NOIR Linux/macOS Launch Script
set -e

if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment not detected. Running ./setup.sh first..."
    ./setup.sh
fi

source .venv/bin/activate
python -m noir.main "$@"

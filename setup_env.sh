#!/usr/bin/env bash
# Setup script for OASIS standalone deconvolution module.
# Creates a Python virtual environment, installs dependencies, and compiles
# the Cython extension.
#
# Usage:
#   bash setup_env.sh          # uses default venv name 'venv'
#   bash setup_env.sh myenv    # uses custom venv name

set -euo pipefail

VENV_NAME="${1:-venv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== OASIS Standalone Setup ==="
echo "Working directory: ${SCRIPT_DIR}"
echo "Virtual environment: ${VENV_NAME}"

# Create virtual environment
if [ ! -d "${SCRIPT_DIR}/${VENV_NAME}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${SCRIPT_DIR}/${VENV_NAME}"
else
    echo "Virtual environment already exists."
fi

# Activate
source "${SCRIPT_DIR}/${VENV_NAME}/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

# Build Cython extension
echo "Building Cython extension..."
cd "${SCRIPT_DIR}"
python setup.py build_ext --inplace

echo ""
echo "=== Setup complete ==="
echo "Activate with: source ${VENV_NAME}/bin/activate"
echo "Run tests with: make test"
echo "Run demo with:  make notebook"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VENV_DIR=${VENV_DIR:-"$REPO_ROOT/.venv"}

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "Python >= 3.8 is required." >&2
    exit 1
fi

echo "Creating or updating virtual environment: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install --editable "$REPO_ROOT"

echo
echo "Environment ready. Activate it with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo
echo "The repository is installed in editable mode, including its command-line tools."

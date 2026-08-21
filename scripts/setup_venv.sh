#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VENV_DIR=${VENV_DIR:-"$REPO_ROOT/.venv"}

if [[ -z "${PYTHON_BIN:-}" ]]; then
    for candidate in python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN=$candidate
            break
        fi
    done
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Python $PYTHON_VERSION detected. Python >= 3.10 is required; install or load a newer Python and try again." >&2
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

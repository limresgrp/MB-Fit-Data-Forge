#!/usr/bin/env bash

# General entry point. The implementation retains the historical glycerol
# filename for compatibility with existing user commands.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/glycerol_qchem_workflow.sh" "$@"

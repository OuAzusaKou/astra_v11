#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ASTRA_PYTHON:-${HOME}/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON" -u -B -m astra.mac_bridge "$@"

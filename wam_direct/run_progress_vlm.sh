#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ " $* " != *" --help "* && " $* " != *" -h "* ]]; then
    set +u
    source "${LEO_ROS_SETUP:-${HOME}/Desktop/colcon_ws/install/setup.bash}"
    set -u
fi
exec python3 -m progress_vlm.observe "$@"

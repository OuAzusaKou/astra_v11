#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
# Control/help commands do not initialize ROS or require its environment.
needs_ros=false
if [[ "${1:-}" == run ]]; then needs_ros=true; fi
for arg in "$@"; do
  case "${arg}" in -h|--help) needs_ros=false ;; esac
done
if ${needs_ros}; then
    ROS_SETUP="${LEO_ROS_SETUP:-${HOME}/Desktop/colcon_ws/install/setup.bash}"
    if [[ ! -f "${ROS_SETUP}" ]]; then
      echo "ROS environment not found: ${ROS_SETUP}" >&2
      exit 1
    fi
    set +u
    source "${ROS_SETUP}"
    set -u
fi
exec python3 -m astra.run "$@"

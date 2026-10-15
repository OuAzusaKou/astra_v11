#!/usr/bin/env bash
# Display only: existing calibrated camera/real-hand publishers plus V6 ghost topic.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export QT_QPA_PLATFORM=xcb
export RVIZ_CONFIG="$ROOT/config/astra/version6_pointcloud.rviz"
export RVIZ_PROJECTION_SCRIPT="$ROOT/astra/live_projections.py"
exec bash "$ROOT/wam_direct/view_overlay_runtime.sh"

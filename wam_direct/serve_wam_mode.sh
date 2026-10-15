#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${IMAGEWAM_TRAIN_REPO:-/share-2/code/chenqi/teleop_wam}"
PORT="${IMAGEWAM_SERVER_PORT:-29571}"
MODE=""
EXTRA_ARGS=()

usage() {
  echo "Usage: $0 --mode {1|2|3|4} [--port PORT] [--no-speedup]"
}

while (($#)); do
  case "$1" in
    --mode)
      MODE="${2:?--mode requires 1, 2, 3, or 4}"
      shift 2
      ;;
    --port)
      PORT="${2:?--port requires a value}"
      shift 2
      ;;
    --no-speedup)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unsupported argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${MODE}" in
  1)
    RUN_NAME="iwam-leo-700h-h30-joint-bs1024-l2-canonical-7092"
    ;;
  2)
    RUN_NAME="iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-7114"
    ;;
  3)
    RUN_NAME="iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-progress-a1-7117"
    ;;
  4)
    RUN_NAME="iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-progress-a1-backprop-7489"
    ;;
  *)
    echo "ERROR: --mode must be 1, 2, 3, or 4" >&2
    usage >&2
    exit 2
    ;;
esac

CHECKPOINT="${REPO_DIR}/experiments/train/${RUN_NAME}/checkpoints/weights/step_100000.pt"
LAUNCHER="${REPO_DIR}/shortcuts/iwam/inference/real_inference.sh"
[[ -f "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -x "${LAUNCHER}" || -f "${LAUNCHER}" ]] || { echo "missing server launcher: ${LAUNCHER}" >&2; exit 1; }

echo "[mode] ${MODE}"
echo "[checkpoint] ${CHECKPOINT}"
echo "[listen] 0.0.0.0:${PORT}"

exec bash "${LAUNCHER}" \
  --checkpoint "${CHECKPOINT}" \
  --port "${PORT}" \
  "${EXTRA_ARGS[@]}"

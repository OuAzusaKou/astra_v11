#!/usr/bin/env bash
set -eo pipefail

REPO_DIR=/mnt/pfs/pfs-yc2F4O/oyx/chenqi/teleop_wam
RUN_DIR="$REPO_DIR/runs/real_robot_hierarchical"
CONDA_ENV=/mnt/pfs/pfs-yc2F4O/oyx/chenqi/conda-envs/imagewam-hierarchical
PORT=29561

VLM_MODEL_PATH=""
WAM_MODEL_PATH=""

usage() {
  cat <<EOF
Usage:
  $0 --vlm-model-path PATH --wam-model-path PATH

Arguments:
  --vlm-model-path PATH  High-level VLM checkpoint directory.
  --wam-model-path PATH  WAM checkpoint file (.../checkpoints/weights/step_*.pt).
EOF
}

while (($#)); do
  case "$1" in
    --vlm-model-path)
      VLM_MODEL_PATH="${2:?--vlm-model-path requires a path}"
      shift 2
      ;;
    --wam-model-path)
      WAM_MODEL_PATH="${2:?--wam-model-path requires a path}"
      shift 2
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

[[ -n "$VLM_MODEL_PATH" ]] || { echo "ERROR: --vlm-model-path is required" >&2; exit 2; }
[[ -n "$WAM_MODEL_PATH" ]] || { echo "ERROR: --wam-model-path is required" >&2; exit 2; }
[[ -d "$VLM_MODEL_PATH" ]] || { echo "ERROR: VLM checkpoint not found: $VLM_MODEL_PATH" >&2; exit 1; }
[[ -f "$WAM_MODEL_PATH" ]] || { echo "ERROR: WAM checkpoint not found: $WAM_MODEL_PATH" >&2; exit 1; }

VLM_MODEL_PATH="$(readlink -f "$VLM_MODEL_PATH")"
WAM_MODEL_PATH="$(readlink -f "$WAM_MODEL_PATH")"
WAM_RUN_DIR="$(dirname "$(dirname "$(dirname "$WAM_MODEL_PATH")")")"
WAM_CONFIG_PATH="$WAM_RUN_DIR/config.yaml"
[[ -f "$WAM_CONFIG_PATH" ]] || { echo "ERROR: WAM config not found: $WAM_CONFIG_PATH" >&2; exit 1; }

if [[ -t 0 ]]; then
  stty sane
  stty intr '^C'
fi

cd "$REPO_DIR"
if [[ -f "$CONDA_ENV/bin/activate" ]]; then
  source "$CONDA_ENV/bin/activate"
fi
export TOKENIZERS_PARALLELISM=false
export IMAGEWAM_CKPT_ROOT=/mnt/pfs/pfs-yc2F4O/oyx/chenqi/Liberai-Model-clean/checkpoints/imagewam
export FLUX2_SRC=/mnt/pfs/pfs-yc2F4O/oyx/chenqi/Liberai-Model-clean/third_party/flux2
export FLUX2_MODEL_PATH="$IMAGEWAM_CKPT_ROOT/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
export FLUX2_AE_MODEL_PATH="$IMAGEWAM_CKPT_ROOT/flux2/FLUX.2-dev/ae.safetensors"
export FLUX2_QWEN3_MODEL_SPEC="$IMAGEWAM_CKPT_ROOT/Qwen3-4B"
export PYTHONPATH="$REPO_DIR:$FLUX2_SRC:$FLUX2_SRC/src:${PYTHONPATH:-}"

mkdir -p "$RUN_DIR"
if python - "$PORT" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(0.25)
try:
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
finally:
    sock.close()
PY
then
  echo "ERROR: port $PORT is already in use" >&2
  exit 1
fi

echo "starting hierarchical policy server in foreground on port $PORT"
echo "VLM: $VLM_MODEL_PATH"
echo "WAM: $WAM_MODEL_PATH"
echo "press Ctrl+C to stop"

exec "$CONDA_ENV/bin/python" -u -m policy_server.imagewam.hierarchical_policy_server \
  --checkpoint "$WAM_MODEL_PATH" \
  --config "$WAM_CONFIG_PATH" \
  --vlm-model-path "$VLM_MODEL_PATH" \
  --device cuda:0 \
  --vlm-device cuda:3 \
  --vlm-interval-sec 1.0 \
  --vlm-max-new-tokens 64 \
  --progress-threshold 90 \
  --rollback-threshold 70 \
  --max-retries-per-l2 2 \
  --port "$PORT" \
  --debug-dir "$RUN_DIR" \
  --debug-keep-last 20

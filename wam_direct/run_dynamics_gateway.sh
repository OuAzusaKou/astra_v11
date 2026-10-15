#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
exec /home/pc/code/cairuwei/envs/imagewam-cu130-cudnn/bin/python -u -m astra.dynamics_gateway \
  --training-repo /home/pc/code/wangyi/dynamic-wm \
  --run /home/pc/liberai-checkpoint/wangyi/dynamic-wm/train/dynamicswam-h15-30-vector-array-resume10k-r01 \
  --bind 0.0.0.0 --port 29642 "$@"

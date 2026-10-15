#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_REPO="${IMAGEWAM_V22_DEPLOY_REPO:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
ROS_SETUP="${LEO_ROS_SETUP:-/home/liberai/Desktop/colcon_ws/install/setup.bash}"
CLIENT_CONFIG="${IMAGEWAM_CLIENT_CONFIG:-${SCRIPT_DIR}/client_wam_direct.yaml}"

if [[ ! -f "${DEPLOY_REPO}/utils/liber_teleop_v22_client.py" ]]; then
  echo "missing unified-v2.2 deploy repo: ${DEPLOY_REPO}" >&2
  exit 1
fi
if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "missing ROS setup: ${ROS_SETUP}" >&2
  exit 1
fi
if [[ ! -f "${CLIENT_CONFIG}" ]]; then
  echo "missing client config: ${CLIENT_CONFIG}" >&2
  exit 1
fi

set +u
source "${ROS_SETUP}"
set -u
cd "${DEPLOY_REPO}"
export PYTHONPATH="${DEPLOY_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

# Share ownership with the optional Astra runner in this independent copy.
lock_dir="/tmp/leo-astra-$(id -u)"
(umask 077; mkdir -p "${lock_dir}")
[[ -O "${lock_dir}" && "$(stat -c %a "${lock_dir}")" == 700 ]] || {
  echo "robot lock directory must be private: ${lock_dir}" >&2; exit 1;
}
exec 9>"${lock_dir}/robot.lock"
flock -n 9 || { echo "another client owns robot control" >&2; exit 1; }

exec python3 "${SCRIPT_DIR}/inference_imagewam_direct.py" \
  --config "${CLIENT_CONFIG}" \
  "$@"

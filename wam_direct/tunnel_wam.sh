#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${IMAGEWAM_SSH_HOST:-180.76.149.31}"
SSH_PORT="${IMAGEWAM_SSH_PORT:-5338}"
SSH_USER="${IMAGEWAM_SSH_USER:-root}"
SSH_TARGET="${SSH_USER}@${SSH_HOST}"
LOCAL_BIND="${IMAGEWAM_LOCAL_BIND:-127.0.0.1}"
LEGACY_LOCAL_PORT="${IMAGEWAM_LEGACY_LOCAL_PORT:-29670}"
LEGACY_REMOTE_PORT="${IMAGEWAM_LEGACY_REMOTE_PORT:-29575}"
LOCAL_PORT_BASE="${IMAGEWAM_LOCAL_PORT_BASE:-29671}"
REMOTE_PORT_BASE="${IMAGEWAM_REMOTE_PORT_BASE:-29571}"
HIER_LOCAL_PORT_BASE="${IMAGEWAM_HIER_LOCAL_PORT_BASE:-29675}"
HIER_REMOTE_PORT_BASE="${IMAGEWAM_HIER_REMOTE_PORT_BASE:-29561}"
MODE8_LOCAL_PORT="${IMAGEWAM_MODE8_LOCAL_PORT:-29678}"
MODE8_REMOTE_PORT="${IMAGEWAM_MODE8_REMOTE_PORT:-29567}"
MODE9_LOCAL_PORT="${IMAGEWAM_MODE9_LOCAL_PORT:-29679}"
MODE9_REMOTE_PORT="${IMAGEWAM_MODE9_REMOTE_PORT:-29568}"
MODE10_LOCAL_PORT="${IMAGEWAM_MODE10_LOCAL_PORT:-29680}"
MODE10_REMOTE_PORT="${IMAGEWAM_MODE10_REMOTE_PORT:-29569}"
RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}/wzn-wam-direct"
CONTROL_SOCKET="${RUNTIME_DIR}/ssh-control.sock"

usage() {
  echo "Usage: $0 {start|status|stop}"
}

is_running() {
  ssh -p "${SSH_PORT}" -S "${CONTROL_SOCKET}" -O check "${SSH_TARGET}" >/dev/null 2>&1
}

case "${1:-}" in
  start)
    mkdir -p "${RUNTIME_DIR}"
    if is_running; then
      if ! ss -ltn "sport = :${MODE9_LOCAL_PORT}" | grep -q LISTEN; then
        ssh -p "${SSH_PORT}" -S "${CONTROL_SOCKET}" -O forward \
          -L "${LOCAL_BIND}:${MODE9_LOCAL_PORT}:127.0.0.1:${MODE9_REMOTE_PORT}" "${SSH_TARGET}"
      fi
      if ! ss -ltn "sport = :${MODE10_LOCAL_PORT}" | grep -q LISTEN; then
        ssh -p "${SSH_PORT}" -S "${CONTROL_SOCKET}" -O forward \
          -L "${LOCAL_BIND}:${MODE10_LOCAL_PORT}:127.0.0.1:${MODE10_REMOTE_PORT}" "${SSH_TARGET}"
      fi
      echo "direct WAM tunnels already running: mode0 ${LOCAL_BIND}:${LEGACY_LOCAL_PORT}->127.0.0.1:${LEGACY_REMOTE_PORT}; modes1-4 ${LOCAL_BIND}:${LOCAL_PORT_BASE}-$((${LOCAL_PORT_BASE} + 3)) -> ${SSH_TARGET}:${SSH_PORT}:127.0.0.1:${REMOTE_PORT_BASE}-$((${REMOTE_PORT_BASE} + 3))"
      echo "hierarchical modes5-7 ${LOCAL_BIND}:${HIER_LOCAL_PORT_BASE}-$((HIER_LOCAL_PORT_BASE + 2)) -> 127.0.0.1:${HIER_REMOTE_PORT_BASE}-$((HIER_REMOTE_PORT_BASE + 2)); mode8 ${LOCAL_BIND}:${MODE8_LOCAL_PORT}->127.0.0.1:${MODE8_REMOTE_PORT}; mode9 ${LOCAL_BIND}:${MODE9_LOCAL_PORT}->127.0.0.1:${MODE9_REMOTE_PORT}; mode10 ${LOCAL_BIND}:${MODE10_LOCAL_PORT}->127.0.0.1:${MODE10_REMOTE_PORT}"
      exit 0
    fi
    forwards=(-L "${LOCAL_BIND}:${LEGACY_LOCAL_PORT}:127.0.0.1:${LEGACY_REMOTE_PORT}")
    for offset in 0 1 2 3; do
      forwards+=(
        -L
        "${LOCAL_BIND}:$((${LOCAL_PORT_BASE} + offset)):127.0.0.1:$((${REMOTE_PORT_BASE} + offset))"
      )
    done
    for offset in 0 1 2; do
      forwards+=(
        -L
        "${LOCAL_BIND}:$((HIER_LOCAL_PORT_BASE + offset)):127.0.0.1:$((HIER_REMOTE_PORT_BASE + offset))"
      )
    done
      forwards+=( -L "${LOCAL_BIND}:${MODE8_LOCAL_PORT}:127.0.0.1:${MODE8_REMOTE_PORT}" )
    forwards+=( -L "${LOCAL_BIND}:${MODE9_LOCAL_PORT}:127.0.0.1:${MODE9_REMOTE_PORT}" )
    forwards+=( -L "${LOCAL_BIND}:${MODE10_LOCAL_PORT}:127.0.0.1:${MODE10_REMOTE_PORT}" )
    ssh -p "${SSH_PORT}" -M -S "${CONTROL_SOCKET}" -fnNT \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      "${forwards[@]}" \
      "${SSH_TARGET}"
    echo "started: mode0 ${LOCAL_BIND}:${LEGACY_LOCAL_PORT}->127.0.0.1:${LEGACY_REMOTE_PORT}; modes1-4 ${LOCAL_BIND}:${LOCAL_PORT_BASE}-$((${LOCAL_PORT_BASE} + 3)) -> ${SSH_TARGET}:${SSH_PORT}:127.0.0.1:${REMOTE_PORT_BASE}-$((${REMOTE_PORT_BASE} + 3))"
      echo "hierarchical modes5-7 ${LOCAL_BIND}:${HIER_LOCAL_PORT_BASE}-$((HIER_LOCAL_PORT_BASE + 2)) -> 127.0.0.1:${HIER_REMOTE_PORT_BASE}-$((HIER_REMOTE_PORT_BASE + 2)); mode8 ${LOCAL_BIND}:${MODE8_LOCAL_PORT}->127.0.0.1:${MODE8_REMOTE_PORT}; mode9 ${LOCAL_BIND}:${MODE9_LOCAL_PORT}->127.0.0.1:${MODE9_REMOTE_PORT}; mode10 ${LOCAL_BIND}:${MODE10_LOCAL_PORT}->127.0.0.1:${MODE10_REMOTE_PORT}"
    ;;
  status)
    if is_running; then
      echo "running: mode0 ${LOCAL_BIND}:${LEGACY_LOCAL_PORT}->127.0.0.1:${LEGACY_REMOTE_PORT}; modes1-4 ${LOCAL_BIND}:${LOCAL_PORT_BASE}-$((${LOCAL_PORT_BASE} + 3)) -> ${SSH_TARGET}:${SSH_PORT}:127.0.0.1:${REMOTE_PORT_BASE}-$((${REMOTE_PORT_BASE} + 3))"
      echo "hierarchical modes5-7 ${LOCAL_BIND}:${HIER_LOCAL_PORT_BASE}-$((HIER_LOCAL_PORT_BASE + 2)) -> 127.0.0.1:${HIER_REMOTE_PORT_BASE}-$((HIER_REMOTE_PORT_BASE + 2)); mode8 ${LOCAL_BIND}:${MODE8_LOCAL_PORT}->127.0.0.1:${MODE8_REMOTE_PORT}; mode9 ${LOCAL_BIND}:${MODE9_LOCAL_PORT}->127.0.0.1:${MODE9_REMOTE_PORT}; mode10 ${LOCAL_BIND}:${MODE10_LOCAL_PORT}->127.0.0.1:${MODE10_REMOTE_PORT}"
    else
      echo "stopped"
      exit 1
    fi
    ;;
  stop)
    if is_running; then
      ssh -p "${SSH_PORT}" -S "${CONTROL_SOCKET}" -O exit "${SSH_TARGET}" >/dev/null
      echo "stopped"
    else
      echo "already stopped"
    fi
    ;;
  *)
    usage
    exit 2
    ;;
esac

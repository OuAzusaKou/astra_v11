#!/usr/bin/env bash
set -euo pipefail

# Stream the four Leo-700h WAM runs through this machine without writing model
# data to its filesystem. Incomplete destination files use a .part suffix and
# are resumed on the next invocation.

SOURCE_SSH_TARGET="${SOURCE_SSH_TARGET:-gpu03}"
SOURCE_ROOT="${SOURCE_ROOT:-/share-2/code/chenqi/teleop_wam/experiments/train}"
SSH_BIN="${SSH_BIN:-ssh}"

DEST_SSH_HOST="${DEST_SSH_HOST:-180.76.149.31}"
DEST_SSH_PORT="${DEST_SSH_PORT:-5338}"
DEST_SSH_USER="${DEST_SSH_USER:-root}"
DEST_SSH_TARGET="${DEST_SSH_USER}@${DEST_SSH_HOST}"
DEST_ROOT="${DEST_ROOT:-/mnt/pfs/pfs-yc2F4O/oyx/chenqi/checkpoints/wam/700h}"

RUNS=(
  iwam-leo-700h-h30-joint-bs1024-l2-canonical-7092
  iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-7114
  iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-progress-a1-7117
  iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-progress-a1-backprop-7489
)

SOURCE_SSH=("${SSH_BIN}" "${SOURCE_SSH_TARGET}")
DEST_SSH=("${SSH_BIN}" -p "${DEST_SSH_PORT}" "${DEST_SSH_TARGET}")

source_ssh() {
  "${SOURCE_SSH[@]}" "$@"
}

dest_ssh() {
  "${DEST_SSH[@]}" "$@"
}

sha256_source() {
  local output
  output="$(source_ssh "sha256sum '$1'")"
  printf '%s\n' "${output%% *}"
}

sha256_dest() {
  local output
  output="$(dest_ssh "sha256sum '$1'")"
  printf '%s\n' "${output%% *}"
}

stream_file() {
  local source_file="$1"
  local dest_file="$2"
  local dest_dir dest_part source_size final_size part_size remaining
  local source_hash dest_hash backup_suffix

  dest_dir="$(dirname -- "${dest_file}")"
  dest_part="${dest_file}.part"

  source_ssh "test -r '${source_file}'"
  dest_ssh "mkdir -p '${dest_dir}'"
  source_size="$(source_ssh "stat -c %s '${source_file}'")"

  final_size="$(dest_ssh "if [[ -f '${dest_file}' ]]; then stat -c %s '${dest_file}'; else echo -1; fi")"
  if [[ "${final_size}" == "${source_size}" ]]; then
    echo "[verify] existing ${dest_file}"
    source_hash="$(sha256_source "${source_file}")"
    dest_hash="$(sha256_dest "${dest_file}")"
    if [[ "${source_hash}" == "${dest_hash}" ]]; then
      echo "[skip] verified ${dest_file}"
      return
    fi
    backup_suffix="$(date +%Y%m%d-%H%M%S)"
    dest_ssh "mv '${dest_file}' '${dest_file}.checksum-mismatch-${backup_suffix}'"
    echo "[warn] preserved mismatched destination as ${dest_file}.checksum-mismatch-${backup_suffix}"
  elif [[ "${final_size}" != "-1" ]]; then
    backup_suffix="$(date +%Y%m%d-%H%M%S)"
    dest_ssh "mv '${dest_file}' '${dest_file}.size-mismatch-${backup_suffix}'"
    echo "[warn] preserved incomplete final file as ${dest_file}.size-mismatch-${backup_suffix}"
  fi

  part_size="$(dest_ssh "if [[ -f '${dest_part}' ]]; then stat -c %s '${dest_part}'; else echo 0; fi")"
  if ((part_size > source_size)); then
    backup_suffix="$(date +%Y%m%d-%H%M%S)"
    dest_ssh "mv '${dest_part}' '${dest_part}.oversized-${backup_suffix}'"
    echo "[warn] preserved oversized partial file as ${dest_part}.oversized-${backup_suffix}"
    part_size=0
  fi

  remaining=$((source_size - part_size))
  if ((remaining > 0)); then
    echo "[stream] ${source_file}"
    echo "         offset=${part_size} remaining=${remaining} bytes"
    if command -v pv >/dev/null 2>&1; then
      source_ssh "dd if='${source_file}' bs=4M iflag=skip_bytes skip=${part_size} status=none" \
        | pv -s "${remaining}" \
        | dest_ssh "dd of='${dest_part}' bs=4M oflag=append conv=notrunc status=none"
    else
      source_ssh "dd if='${source_file}' bs=4M iflag=skip_bytes skip=${part_size} status=none" \
        | dest_ssh "dd of='${dest_part}' bs=4M oflag=append conv=notrunc status=progress"
    fi
  fi

  part_size="$(dest_ssh "stat -c %s '${dest_part}'")"
  if [[ "${part_size}" != "${source_size}" ]]; then
    echo "ERROR: size mismatch after transfer: source=${source_size}, destination=${part_size}" >&2
    exit 1
  fi

  echo "[verify] SHA256 ${dest_part}"
  source_hash="$(sha256_source "${source_file}")"
  dest_hash="$(sha256_dest "${dest_part}")"
  if [[ "${source_hash}" != "${dest_hash}" ]]; then
    echo "ERROR: SHA256 mismatch for ${dest_part}" >&2
    exit 1
  fi

  dest_ssh "mv '${dest_part}' '${dest_file}'"
  echo "[done] ${dest_file}"
}

echo "[preflight] source=${SOURCE_SSH_TARGET}:${SOURCE_ROOT}"
echo "[preflight] destination=${DEST_SSH_TARGET}:${DEST_SSH_PORT}:${DEST_ROOT}"
source_ssh true
dest_ssh "mkdir -p '${DEST_ROOT}'"

for run_name in "${RUNS[@]}"; do
  echo
  echo "[run] ${run_name}"
  stream_file \
    "${SOURCE_ROOT}/${run_name}/config.yaml" \
    "${DEST_ROOT}/${run_name}/config.yaml"
  stream_file \
    "${SOURCE_ROOT}/${run_name}/checkpoints/weights/step_100000.pt" \
    "${DEST_ROOT}/${run_name}/checkpoints/weights/step_100000.pt"
done

echo
echo "All four Leo-700h WAM runs are present and SHA256-verified on Baidu Cloud."

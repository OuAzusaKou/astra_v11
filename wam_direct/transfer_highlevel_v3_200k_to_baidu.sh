#!/usr/bin/env bash
set -euo pipefail

# Stream two exported High-Level V3 checkpoints through the robot without
# storing model data on the robot. Incomplete files keep a .part suffix and are
# resumed on the next invocation.

SOURCE_SSH_TARGET="${SOURCE_SSH_TARGET:-gpu03}"
SOURCE_ROOT="${SOURCE_ROOT:-/share-2/code/wzn/vlm_pretrain/runs}"
DEST_SSH_TARGET="${DEST_SSH_TARGET:-root@180.76.149.31}"
DEST_SSH_PORT="${DEST_SSH_PORT:-5338}"
DEST_ROOT="${DEST_ROOT:-/mnt/pfs/pfs-yc2F4O/oyx/chenqi/checkpoints/highlevel/v3}"

RUNS=(
  highlevel_v3_mix60_from_v2_120k_300k
  highlevel_only_teleop_v3_from_v2_120k_300k
)

source_ssh() {
  ssh "${SOURCE_SSH_TARGET}" "$@"
}

dest_ssh() {
  ssh -p "${DEST_SSH_PORT}" "${DEST_SSH_TARGET}" "$@"
}

remote_sha256() {
  local side="$1"
  local path="$2"
  local output
  if [[ "${side}" == source ]]; then
    output="$(source_ssh "sha256sum '${path}'")"
  else
    output="$(dest_ssh "sha256sum '${path}'")"
  fi
  printf '%s\n' "${output%% *}"
}

stream_file() {
  local source_file="$1"
  local dest_file="$2"
  local dest_dir dest_part source_size final_size part_size remaining
  local source_hash dest_hash suffix

  dest_dir="$(dirname -- "${dest_file}")"
  dest_part="${dest_file}.part"
  source_size="$(source_ssh "stat -c %s '${source_file}'")"
  dest_ssh "mkdir -p '${dest_dir}'"

  final_size="$(dest_ssh "if [[ -f '${dest_file}' ]]; then stat -c %s '${dest_file}'; else echo -1; fi")"
  if [[ "${final_size}" == "${source_size}" ]]; then
    source_hash="$(remote_sha256 source "${source_file}")"
    dest_hash="$(remote_sha256 dest "${dest_file}")"
    if [[ "${source_hash}" == "${dest_hash}" ]]; then
      echo "[skip] verified ${dest_file}"
      return
    fi
  fi

  if [[ "${final_size}" != "-1" ]]; then
    suffix="$(date +%Y%m%d-%H%M%S)"
    dest_ssh "mv '${dest_file}' '${dest_file}.mismatch-${suffix}'"
  fi

  part_size="$(dest_ssh "if [[ -f '${dest_part}' ]]; then stat -c %s '${dest_part}'; else echo 0; fi")"
  if ((part_size > source_size)); then
    suffix="$(date +%Y%m%d-%H%M%S)"
    dest_ssh "mv '${dest_part}' '${dest_part}.oversized-${suffix}'"
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
    echo "ERROR: size mismatch: source=${source_size}, destination=${part_size}" >&2
    return 1
  fi

  echo "[verify] SHA256 ${dest_part}"
  source_hash="$(remote_sha256 source "${source_file}")"
  dest_hash="$(remote_sha256 dest "${dest_part}")"
  if [[ "${source_hash}" != "${dest_hash}" ]]; then
    echo "ERROR: SHA256 mismatch for ${dest_part}" >&2
    return 1
  fi

  dest_ssh "mv '${dest_part}' '${dest_file}'"
  echo "[done] ${dest_file}"
}

echo "[preflight] source=${SOURCE_SSH_TARGET}:${SOURCE_ROOT}"
echo "[preflight] destination=${DEST_SSH_TARGET}:${DEST_SSH_PORT}:${DEST_ROOT}"
source_ssh true
dest_ssh "mkdir -p '${DEST_ROOT}'"

for run_name in "${RUNS[@]}"; do
  source_dir="${SOURCE_ROOT}/${run_name}/export_hf/step_200000"
  dest_dir="${DEST_ROOT}/${run_name}/step_200000"
  mapfile -t files < <(source_ssh "cd '${source_dir}' && find . -maxdepth 1 -type f -printf '%P\\n' | sort")
  if ((${#files[@]} == 0)); then
    echo "ERROR: no exported files found in ${source_dir}" >&2
    exit 1
  fi

  echo
  echo "[run] ${run_name} (${#files[@]} files)"
  for relative_path in "${files[@]}"; do
    stream_file "${source_dir}/${relative_path}" "${dest_dir}/${relative_path}"
  done
done

echo
echo "Both High-Level V3 step_200000 exports are present and SHA256-verified on Baidu Cloud."

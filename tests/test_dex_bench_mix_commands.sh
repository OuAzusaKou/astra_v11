#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
test_root="$(mktemp -d)"
trap 'rm -rf -- "${test_root}"' EXIT
mix_test_python="${PYTHON:-python3}"
source "${repo_root}/ops/dex_bench_tasks_mano/alias.sh" >/dev/null
export DEPLOY_ROOT="${test_root}"
export DEX_BENCH_ROS_SETUP="${test_root}/setup.bash"
export MIX_TEST_LOG="${test_root}/tunnels"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
touch "${DEX_BENCH_ROS_SETUP}"
cat > "${test_root}/tunneltobaidu.sh" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "${CLIENT_CONFIG}" >> "${MIX_TEST_LOG}"
SH
chmod +x "${test_root}/tunneltobaidu.sh"

python3() {
  if [[ " $* " == *" --print-config "* ]]; then
    "${mix_test_python}" "$@"
  else
    printf '%s\0' "$@" > "${test_root}/inference.argv"
  fi
}

for task in 1 1-q20 2 3 4; do
  dex_bench_eval_imagewam_mix "${task}" --camera-latency 0.200
  mapfile -d '' -t actual < "${test_root}/inference.argv"
  expected=(-m scripts.inference_leo.inference_multiview_leo_mix --task "${task}" --camera-latency 0.200)
  [[ "${#actual[@]}" -eq "${#expected[@]}" ]]
  for index in "${!expected[@]}"; do
    [[ "${actual[index]}" == "${expected[index]}" ]]
  done
  [[ "$(tail -n 1 "${MIX_TEST_LOG}")" == "${repo_root}/config/client_dex_bench_mix_${task}.yaml" ]]
done

echo 'DEX_BENCH_MIX_COMMANDS_OK'

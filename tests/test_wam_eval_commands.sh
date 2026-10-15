#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
fixture_dir="${repo_root}/tests/fixtures/wam_eval"
commands_file="${repo_root}/ops/wam_direct/eval_commands.sh"
test_root="$(mktemp -d)"
trap 'rm -rf -- "${test_root}"' EXIT

cp "${fixture_dir}/run_wam.sh" "${test_root}/run_wam.sh"
mkdir "${test_root}/deploy"
cp "${fixture_dir}/empty_setup.sh" "${test_root}/base_ros_setup.sh"
cp "${fixture_dir}/empty_setup.sh" "${test_root}/robot_ros_setup.sh"
cp "${fixture_dir}/empty_setup.sh" "${test_root}/client.yaml"
export WAM_EVAL_DIR="${test_root}"
export IMAGEWAM_V22_DEPLOY_REPO="${test_root}/deploy"
export IMAGEWAM_CLIENT_CONFIG="${test_root}/client.yaml"
export WAM_EVAL_BASE_ROS_SETUP="${test_root}/base_ros_setup.sh"
export WAM_EVAL_ROS_SETUP="${test_root}/robot_ros_setup.sh"
export WAM_EVAL_TEST_LOG="${test_root}/argv.bin"
export WAM_EVAL_LOCK_FILE="${test_root}/client.lock"
source "${commands_file}" >/dev/null

# Keep process checks deterministic; this test covers their surrounding control
# flow while concurrency itself is exercised with the real flock below.
pgrep() { return 1; }

assert_argv() {
  local expected_mode="$1"
  local expected_port="$2"
  local expected_prompt="$3"
  local -a actual
  mapfile -d '' -t actual <"${WAM_EVAL_TEST_LOG}"
  local -a expected=(
    --mode "${expected_mode}"
    --port "${expected_port}"
    "--prompt=${expected_prompt}"
    --chunk_size 32
    --record
  )
  [[ "${#actual[@]}" -eq "${#expected[@]}" ]]
  local index
  for index in "${!expected[@]}"; do
    [[ "${actual[index]}" == "${expected[index]}" ]]
  done
}

mode0 pick up the A and place it to B
assert_argv 0 29670 'pick up the A and place it to B'

mode2 拿起 A and 放到 B
assert_argv 2 29672 '拿起 A and 放到 B'

IFS=:
mode7 instruction with multiple words
unset IFS
assert_argv 7 29677 'instruction with multiple words'

mode0 --leading-option is still instruction
assert_argv 0 29670 '--leading-option is still instruction'

set +e
mode0 >/dev/null 2>&1
empty_rc=$?
mode2 '   ' >/dev/null 2>&1
blank_rc=$?
WAM_EVAL_TEST_RC=37 mode7 child failure >/dev/null 2>&1
child_rc=$?
set -e

[[ "${empty_rc}" -eq 64 ]]
[[ "${blank_rc}" -eq 64 ]]
[[ "${child_rc}" -eq 37 ]]

unset IMAGEWAM_CLIENT_CONFIG
set +e
mode0 missing approved config >/dev/null 2>&1
config_rc=$?
set -e
export IMAGEWAM_CLIENT_CONFIG="${test_root}/client.yaml"
[[ "${config_rc}" -eq 1 ]]

ready_file="${test_root}/client.ready"
export WAM_EVAL_TEST_READY_FILE="${ready_file}"
export WAM_EVAL_TEST_SLEEP=0.5
mode0 first client holds lock >/dev/null 2>&1 &
first_client_pid=$!
for _ in {1..100}; do
  [[ -f "${ready_file}" ]] && break
  sleep 0.01
done
[[ -f "${ready_file}" ]]
unset WAM_EVAL_TEST_READY_FILE WAM_EVAL_TEST_SLEEP
set +e
mode0 second client must fail >/dev/null 2>&1
lock_rc=$?
set -e
wait "${first_client_pid}"
[[ "${lock_rc}" -eq 1 ]]

mode0 lock is released after child exit
assert_argv 0 29670 'lock is released after child exit'

echo 'WAM_EVAL_COMMANDS_OK'

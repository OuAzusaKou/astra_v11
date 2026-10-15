#!/usr/bin/env bash

printf '%s\0' "$@" >"${WAM_EVAL_TEST_LOG:?}"
if [[ -n "${WAM_EVAL_TEST_READY_FILE:-}" ]]; then
  : >"${WAM_EVAL_TEST_READY_FILE}"
fi
if [[ -n "${WAM_EVAL_TEST_SLEEP:-}" ]]; then
  sleep "${WAM_EVAL_TEST_SLEEP}"
fi
exit "${WAM_EVAL_TEST_RC:-0}"

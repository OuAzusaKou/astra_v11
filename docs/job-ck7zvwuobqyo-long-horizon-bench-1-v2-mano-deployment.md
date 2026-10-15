# Long Horizon Bench 1 v2 ImageWAM MANO 真机评测

模型：`job-ck7zvwuobqyo` / `imagewam-longhorizon1-v2-mano-h32-368x320-bs512-10k-1n-b30z-20260830-050614` / `step_010000.pt`

任务：`ClearTable1_Combination1_env213`、`ClearTable3_Combination1_env213`、`PerpareBreakfast1_combination1_env213`、`PerpareBreakfast4_combination1_env213`

## 1. 同步代码

```bash
SSH_CONFIG_PATH="${HOME}/.ssh/config"
mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"
touch "${SSH_CONFIG_PATH}"
chmod 600 "${SSH_CONFIG_PATH}"

if ! grep -Eq '^[[:space:]]*Host[[:space:]]+bce_dev01([[:space:]]|$)' "${SSH_CONFIG_PATH}"; then
  SSH_CONFIG_TMP="$(mktemp)"
  {
    printf '%s\n' \
      'Host bce_dev01' \
      '  HostName 180.76.149.31' \
      '  User root' \
      '  Port 5338' \
      ''
    sed -n '1,$p' "${SSH_CONFIG_PATH}"
  } > "${SSH_CONFIG_TMP}"
  install -m 600 "${SSH_CONFIG_TMP}" "${SSH_CONFIG_PATH}"
  rm -f "${SSH_CONFIG_TMP}"
fi

ssh \
  -o BatchMode=yes \
  -o ConnectTimeout=10 \
  -o StrictHostKeyChecking=accept-new \
  bce_dev01 'echo SERVER_SSH_OK'

DEPLOY_ROOT="${HOME}/bangguo/Liberai_Deploy"
mkdir -p "${DEPLOY_ROOT}"
rsync -av \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.venv/' \
  --exclude='outputs/' \
  --exclude='records/' \
  bce_dev01:/mnt/pfs/pfs-yc2F4O/bangguo/code/Liberai_Deploy/ \
  "${DEPLOY_ROOT}/"

cd "${DEPLOY_ROOT}"
source "${DEPLOY_ROOT}/ops/long_horizon_bench_1_v2_mano/alias.sh"
python3 scripts/build_native_mano_ik.py --force
python3 - <<'PY'
from mano.native.runtime import native_extension_path

path = native_extension_path()
if path is None:
    raise SystemExit("ERROR: MANO native extension is unavailable")
print(f"MANO native extension: {path}")
PY
```

## 2. 评测指令

### 清理桌面 1

```bash
long_horizon_bench_1_v2_eval_cleartable1_imagewam_mano
```

### 清理桌面 3

```bash
long_horizon_bench_1_v2_eval_cleartable3_imagewam_mano
```

### 准备早餐 1

```bash
long_horizon_bench_1_v2_eval_preparebreakfast1_imagewam_mano
```

### 准备早餐 4

```bash
long_horizon_bench_1_v2_eval_preparebreakfast4_imagewam_mano
```

# Long Horizon Bench 1 v2 L3 / L2 / L3+L2 ImageWAM MANO 真机评测

模型：

- L3：`job-a468p4c3uw5c` / `imagewam-longhorizon1-v2-mano-joint-h32-368x320-bs512-60k-1n-b30z-20260901-031831` / `step_060000.pt`
- L2：`job-qaljr0cp48ll` / `imagewam-longhorizon1-v2-mano-joint-gemini-l2-h32-368x320-bs512-60k-1n-b30z-20260902-121823` / `step_060000.pt`
- L3+L2：`job-su7tsg2oqi1f` / `imagewam-longhorizon1-v2-mano-joint-gemini-l3l2-h32-368x320-bs512-60k-1n-b30z-20260902-121825` / `step_060000.pt`

共 4 个任务布局 × 6 种操作顺序 = 24 个 setting。每个 setting 分别评测 L3、L2、L3+L2，共 72 次。

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
source "${DEPLOY_ROOT}/ops/long_horizon_bench_1_v2_language_mano/alias.sh"
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

L2 与 L3+L2 每完成当前子任务后按回车，Client 才会切换到下一条指令。L3 在当前 setting 的整个评测过程中保持不变。

### 清理桌面 1

#### Combination 1

L3：餐巾纸摆到餐桌左侧，把所有餐具叠放在右侧，剩余食物放在水果盘中，把餐余垃圾扔进垃圾桶，垃圾桶摆到餐桌右侧。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c1
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_cleartable1_c1
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_cleartable1_c1
```

#### Combination 2

L3：把所有餐具叠放在右侧，剩余食物放在水果盘中，把餐余垃圾扔进垃圾桶，垃圾桶摆到餐桌右侧，餐巾纸摆到餐桌左侧。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c2
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_cleartable1_c2
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_cleartable1_c2
```

#### Combination 3

L3：把所有餐具叠放在右侧，垃圾桶摆到餐桌右侧，餐巾纸摆到餐桌左侧，剩余食物放在水果盘中，把餐余垃圾扔进垃圾桶。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c3
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_cleartable1_c3
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_cleartable1_c3
```

#### Combination 4

L3：剩余食物放在水果盘中，把所有餐具叠放在右侧，垃圾桶摆到餐桌右侧，把餐余垃圾扔进垃圾桶，餐巾纸摆到餐桌左侧。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c4
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_cleartable1_c4
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_cleartable1_c4
```

#### Combination 5

L3：把餐余垃圾扔进垃圾桶，餐巾纸摆到餐桌左侧，把所有餐具叠放在右侧，剩余食物放在水果盘中，垃圾桶摆到餐桌右侧。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c5
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_cleartable1_c5
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_cleartable1_c5
```

#### Combination 6

L3：把餐余垃圾扔进垃圾桶，把所有餐具叠放在右侧，垃圾桶摆到餐桌右侧，剩余食物放在水果盘中，餐巾纸摆到餐桌左侧。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c6
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_cleartable1_c6
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_cleartable1_c6
```

### 清理桌面 3

#### Combination 1

L3：餐巾纸摆到餐垫上边，把所有餐具叠放在左侧，剩余食物放在水果盘中，把餐余垃圾扔进垃圾桶，垃圾桶摆到餐桌右侧。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c1
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_cleartable3_c1
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_cleartable3_c1
```

#### Combination 2

L3：把所有餐具叠放在左侧，把餐余垃圾扔进垃圾桶，剩余食物放在水果盘中，垃圾桶摆到餐桌右侧，餐巾纸摆到餐垫上边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c2
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_cleartable3_c2
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_cleartable3_c2
```

#### Combination 3

L3：把所有餐具叠放在左侧，垃圾桶摆到餐桌右侧，餐巾纸摆到餐垫上边，剩余食物放在水果盘中，把餐余垃圾扔进垃圾桶。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c3
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_cleartable3_c3
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_cleartable3_c3
```

#### Combination 4

L3：剩余食物放在水果盘中，把所有餐具叠放在左侧，垃圾桶摆到餐桌右侧，把餐余垃圾扔进垃圾桶，餐巾纸摆到餐垫上边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c4
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_cleartable3_c4
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_cleartable3_c4
```

#### Combination 5

L3：把餐余垃圾扔进垃圾桶，餐巾纸摆到餐垫上边，把所有餐具叠放在左侧，剩余食物放在水果盘中，垃圾桶摆到餐桌右侧。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c5
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_cleartable3_c5
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_cleartable3_c5
```

#### Combination 6

L3：把餐余垃圾扔进垃圾桶，把所有餐具叠放在左侧，垃圾桶摆到餐桌右侧，剩余食物放在水果盘中，餐巾纸摆到餐垫上边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c6
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_cleartable3_c6
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_cleartable3_c6
```

### 准备早餐 1

#### Combination 1

L3：把垃圾桶放在桌面左侧，清除餐桌上的垃圾，把餐盘摆放到餐垫中间，将食物放到餐盘上，把水杯摆放到餐垫左边，把餐巾纸摆到餐垫上边，把水果盘放到餐垫右边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c1
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_preparebreakfast1_c1
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_preparebreakfast1_c1
```

#### Combination 2

L3：把垃圾桶放在桌面左侧，清除餐桌上的垃圾，把餐盘摆放到餐垫中间，将食物放到餐盘上，把水杯摆放到餐垫左边，把水果盘放到餐垫右边，把餐巾纸摆到餐垫上边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c2
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_preparebreakfast1_c2
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_preparebreakfast1_c2
```

#### Combination 3

L3：把垃圾桶放在桌面左侧，清除餐桌上的垃圾，把水果盘放到餐垫右边，把餐盘摆放到餐垫中间，将食物放到餐盘上，把水杯摆放到餐垫左边，把餐巾纸摆到餐垫上边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c3
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_preparebreakfast1_c3
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_preparebreakfast1_c3
```

#### Combination 4

L3：清除餐桌上的垃圾，把水果盘放到餐垫右边，把餐盘摆放到餐垫中间，将食物放到餐盘上，把水杯摆放到餐垫左边，把餐巾纸摆到餐垫上边，把垃圾桶放在桌面左侧。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c4
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_preparebreakfast1_c4
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_preparebreakfast1_c4
```

#### Combination 5

L3：清除餐桌上的垃圾，把水果盘放到餐垫右边，把餐盘摆放到餐垫中间，把水杯摆放到餐垫左边，将食物放到餐盘上，把垃圾桶放在桌面左侧，把餐巾纸摆到餐垫上边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c5
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_preparebreakfast1_c5
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_preparebreakfast1_c5
```

#### Combination 6

L3：清除餐桌上的垃圾，把垃圾桶放在桌面左侧，把餐盘摆放到餐垫中间，把水果盘放到餐垫右边，把水杯摆放到餐垫左边，将食物放到餐盘上，把餐巾纸摆到餐垫上边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c6
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_preparebreakfast1_c6
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_preparebreakfast1_c6
```

### 准备早餐 4

#### Combination 1

L3：把垃圾桶放在桌面右侧，清除餐桌上的垃圾，把餐盘摆放到餐垫左边，将食物放到餐盘上，把水杯摆放到餐垫左上边，把餐巾纸摆到餐垫右上边，把水果盘放到餐垫右边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c1
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_preparebreakfast4_c1
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_preparebreakfast4_c1
```

#### Combination 2

L3：把垃圾桶放在桌面右侧，清除餐桌上的垃圾，把餐盘摆放到餐垫左边，将食物放到餐盘上，把水杯摆放到餐垫左上边，把水果盘放到餐垫右边，把餐巾纸摆到餐垫右上边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c2
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_preparebreakfast4_c2
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_preparebreakfast4_c2
```

#### Combination 3

L3：把垃圾桶放在桌面右侧，清除餐桌上的垃圾，把水果盘放到餐垫右边，把餐盘摆放到餐垫左边，将食物放到餐盘上，把水杯摆放到餐垫左上边，把餐巾纸摆到餐垫右上边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c3
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_preparebreakfast4_c3
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_preparebreakfast4_c3
```

#### Combination 4

L3：清除餐桌上的垃圾，把水果盘放到餐垫右边，把餐盘摆放到餐垫左边，将食物放到餐盘上，把水杯摆放到餐垫左上边，把餐巾纸摆到餐垫右上边，把垃圾桶放在桌面右侧。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c4
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_preparebreakfast4_c4
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_preparebreakfast4_c4
```

#### Combination 5

L3：清除餐桌上的垃圾，把水果盘放到餐垫右边，把餐盘摆放到餐垫左边，把水杯摆放到餐垫左上边，将食物放到餐盘上，把垃圾桶放在桌面右侧，把餐巾纸摆到餐垫右上边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c5
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_preparebreakfast4_c5
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_preparebreakfast4_c5
```

#### Combination 6

L3：清除餐桌上的垃圾，把垃圾桶放在桌面右侧，把餐盘摆放到餐垫左边，把水果盘放到餐垫右边，把水杯摆放到餐垫左上边，将食物放到餐盘上，把餐巾纸摆到餐垫右上边。

L3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c6
```

L2：

```bash
long_horizon_bench_1_v2_eval_l2_preparebreakfast4_c6
```

L3+L2：

```bash
long_horizon_bench_1_v2_eval_l3_l2_preparebreakfast4_c6
```

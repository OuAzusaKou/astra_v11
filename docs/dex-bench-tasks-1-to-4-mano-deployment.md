# DexBench Task 1-4 ImageWAM MANO / Q20 真机评测

## 0. 当前现场：新 openbottle 100 条 MANO（2026-09-10 训练）

训练 run：`imagewam-openbottle-100ep-mano-default-h32-368x320-lr1e4-10k-1n-b30z-20260910-062113`，`step_010000.pt`。现场服务切换至 `192.168.1.18:29571`，协议为 `liber_teleop_v22`，使用 `teleop_mano` 解码；BF16、H32、368×320、default mask、10 步推理。

```bash
cd /home/liberai/bangguo/Liberai_Deploy_5090_openbottle

# 同步
bash ops/openbottle_v22_5090_leo1/evaluate.sh --inference-mode sync --post-chunk-sleep 1.5

# 异步，默认 30 Hz、N=0、关闭融合
bash ops/openbottle_v22_5090_leo1/evaluate.sh --inference-mode sync-no-sleep

# 自定义 action 频率、等待与融合
bash ops/openbottle_v22_5090_leo1/evaluate.sh --inference-mode sync-no-sleep --action-fps 60 --async-sleep-steps 10 --temporal-ensemble --temporal-ensemble-max-chunks 3
```

两种模式都会先复位，再等待 Enter 开始推理。初始与复位位姿沿用任务 1 已使用的 `ops/dex_bench_tasks_mano/mix_openbottle_train_mean_initial_state.json`。现场相机、臂、手状态与轨迹执行节点需要保持运行。

`--action-fps` 调整相邻 action 的执行间隔；`--async-sleep-steps N` 对应 N/action_fps 秒；`--temporal-ensemble-max-chunks K` 含当前段，0 不限制、1 只用当前预测。异步融合和过期过滤均按相机延迟修正后的 deadline 进行，60 Hz 会压缩同一 chunk 的执行时间。

新模型详情见 [openbottle v2.2 现场说明](../ops/openbottle_v22_5090_leo1/README.md)。另一个新训练 run `imagewam-putglass-full-mano-default-h32-368x320-lr1e4-10k-1n-b30z-20260910-062115` 的配置位于 [putglass v2.2 现场说明](../ops/putglass_v22_5090_leo1/README.md)，预留 `29572`，当前未启动。

下方保留历史模型和通用部署入口；当前现场使用本节命令。

模型：

- 新 v2.2 Task 1：`imagewam-openbottle-100ep-mano-default-h32-368x320-lr1e4-10k-1n-b30z-20260910-062113` / `step_010000.pt`
- 新 v2.2 Task 2：`imagewam-putglass-full-mano-default-h32-368x320-lr1e4-10k-1n-b30z-20260910-062115` / `step_010000.pt`
- `job-3obbt9ffs83w` / `openbottle_ep100_mano6d_ee1_hand1_20260818_0129` / `step_010000.pt`
- `openbottle_full_q20_ee1_hand1_imagewam-openbottle-full-q20-ee1-hand1-v2-1n-b30z-20260818-151505` / `step_010000.pt`（Q20，全量 501 条）
- `job-pvbpr42vt1pg` / `putglassonstand_full_mano6d_ee1_hand05_20260817_1807` / `step_010000.pt`
- `job-ckgu8u644uug` / `imagewam-dexbench2-mano-h32-368x320-lr5e5-10k-1n-b30z-20260826-095327` / `step_010000.pt`
- `job-p70rofjkpwny` / `imagewam-dexbench-task4-mano-h32-368x320-bs512-10k-2n-b30z-20260829-085636` / `step_010000.pt`

## 1. 同步代码

仅使用第 2 节 mix 入口时，不依赖旧客户端，可省略下面向 `LEGACY_IMAGEWAM_CLIENT_ROOT` 同步代码的第二条 `rsync`。第 3 节原任务 1、2 命令仍依赖它。

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
LEGACY_IMAGEWAM_CLIENT_ROOT="${HOME}/bangguo/imagewam/Liberai-Model/leo_policy_clients"
mkdir -p "${DEPLOY_ROOT}"
mkdir -p "${LEGACY_IMAGEWAM_CLIENT_ROOT}"
rsync -av \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.venv/' \
  --exclude='outputs/' \
  --exclude='records/' \
  bce_dev01:/mnt/pfs/pfs-yc2F4O/bangguo/code/Liberai_Deploy/ \
  "${DEPLOY_ROOT}/"
rsync -av \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.venv/' \
  --exclude='outputs/' \
  --exclude='records/' \
  bce_dev01:/mnt/pfs/pfs-yc2F4O/bangguo/code/real-wam/Liberai-Model/leo_policy_clients/ \
  "${LEGACY_IMAGEWAM_CLIENT_ROOT}/"

cd "${DEPLOY_ROOT}"
source "${DEPLOY_ROOT}/ops/dex_bench_tasks_mano/alias.sh"
python3 scripts/build_native_mano_ik.py --force
python3 - <<'PY'
from mano.native.runtime import native_extension_path

path = native_extension_path()
if path is None:
    raise SystemExit("ERROR: MANO native extension is unavailable")
print(f"MANO native extension: {path}")
PY
```

## 2. 统一 mix client（推荐的新入口）

完成第 1 节代码同步后，在同一份 `Liberai_Deploy` 中启动以下命令。每次启动选择一个任务/模型：

```bash
source "${DEPLOY_ROOT}/ops/dex_bench_tasks_mano/alias.sh"
dex_bench_eval_imagewam_mix 1       # 任务 1，MANO 100 条
dex_bench_eval_imagewam_mix 1-q20   # 任务 1，Q20 全量
dex_bench_eval_imagewam_mix 2       # 任务 2，MANO
dex_bench_eval_imagewam_mix 3       # 任务 3，MANO
dex_bench_eval_imagewam_mix 4       # 任务 4，MANO
```

上面的命令每次选一条运行。入口会建立对应隧道、加载 ROS 环境，并使用统一推理程序；mix 路径仅依赖当前仓库。`mix` 表示兼容不同服务端协议，不是旧版 MIX 模型的坐标变换。

| task | 模型 | 本地 → 远端端口 | 请求 / action 协议 |
| --- | --- | --- | --- |
| `1` | OpenBottle MANO 100 条 | `19213 → 18213` | standardized，服务端 MANO → Q20，返回 58D |
| `1-q20` | OpenBottle Q20 全量 | `19225 → 18225` | standardized，返回 58D |
| `2` | PutGlass MANO | `19221 → 18221` | standardized，服务端 MANO → Q20，返回 58D |
| `3` | InsertStraw MANO | `29555 → 29566` | LiberTeleop v2.2，返回 326D，本地 MANO → Q20 |
| `4` | SweepTable MANO | `29557 → 29569` | LiberTeleop v2.2，返回 326D，本地 MANO → Q20 |

客户端是 `utils/mix_policy_client.py`，推理入口是 `scripts/inference_leo/inference_multiview_leo_mix.py`，实际循环复用 `scripts/inference_leo/inference_multiview_leo_sync.py`，过期 action 过滤在 `utils/action_timing.py`。任务配置位于 `config/client_dex_bench_mix_{1,1-q20,2,3,4}.yaml`。两种协议最终都转换为绝对控制 `[T,52] = arm12 + hand40`。任务 1、2 的初始状态从原评测使用的训练集均值文件原值导出为 JSON；任务 3、4 继续使用原评测的初始状态文件。

### 无 sleep、按观测时间过滤过期 action

新配置默认 `inference_mode: sync-no-sleep`。机器人执行当前 chunk 时可以推理下一段；TCP 推理请求仍然串行，没有独立的异步推理线程。每次发布后立即获取新观测并发起下一次推理，不等待 chunk 执行，也不额外等待相机延迟。启动复位等待和获取新相机帧的等待仍然存在。

时间轴使用已扣除相机延迟的 `obs["timestamp"]`：

```text
t_obs = camera_header_stamp - camera_latency
t_action[i] = t_obs + (i + action_row_offset) / fps
action_timestamps[i] = t_action[i] - t_publish
```

发布时同时删除 `action_timestamps <= 0` 对应的 arm/hand action（若设置执行器延迟，还会过滤无法及时执行的行）。`t_publish - t_obs` 已包含相机、观测处理、网络、推理和 MANO 解码耗时，不再重复减相机延迟。剩余 action 保留原始 deadline，arm 和 hand 消息使用同一个 ROS header 时间；新 chunk 使用 `cancel_previous=True` 替换前一条轨迹的未执行部分。整段过期时不发布空消息、不取消之前的轨迹，直接继续下一次推理。

`action_row_offset` 优先读取服务端声明；旧 v2.2 服务未返回该字段时，任务 3、4 的配置明确指定为 `0`。`discard_steps` 不会把剩余 action 的时间轴重新归零。默认 chunk 上限是 32；可用 `--chunk_size 30` 改为 30。

头部相机延迟默认 100 ms，可按实测值改成 200 ms：

```bash
dex_bench_eval_imagewam_mix 3 --camera-latency 0.200
```

该参数用于环境层的观测时间修正和状态对齐。新配置的腕部延迟沿用各自客户端：任务 1、2 为 94 ms，任务 3、4 为 40 ms；可用 `--wrist-camera-latency` 覆盖。日志输出 `observation_age`、`published` 和 `expired`，若持续 `published=0`，说明推理返回时整个可执行 horizon 已过期。

需要带等待的同步模式时：

```bash
dex_bench_eval_imagewam_mix 3 --inference-mode sync --post-chunk-sleep 1.5
```

已手动建立隧道并加载 ROS 环境时，也可直接运行：

```bash
python3 -m scripts.inference_leo.inference_multiview_leo_mix --task 3 --camera-latency 0.200
```

离线验证：22 项相关测试和五个任务/模型启动入口测试通过；新旧客户端协议对照结果一致。尚未进行真机验证。

## 3. 原评测指令

### 任务 1：拧开瓶盖

MANO 100 条：

```bash
dex_bench_task_1_eval_imagewam_mano
```

Q20 500 条：

```bash
dex_bench_task_1_eval_imagewam_q20_500
```

### 任务 2：把眼镜挂到眼镜架上

```bash
dex_bench_task_2_eval_imagewam_mano
```

### 任务 3：插吸管

```bash
dex_bench_task_3_eval_imagewam_mano
```

### 任务 4：清扫桌面

```bash
dex_bench_task_4_eval_imagewam_mano
```

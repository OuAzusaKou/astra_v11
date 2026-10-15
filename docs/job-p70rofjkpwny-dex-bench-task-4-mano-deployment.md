# DexBench Task 4 清扫桌面 ImageWAM MANO 真机评测

2026-09-10 当前现场已切回 [任务 3 插吸管](../ops/insertstraw_5090_leo1/README.md)，以下任务 4 配置供再次部署时参考。

## 0. 历史现场配置：5090 / leo1 任务 4（服务已停止）

当前现场模型为 Task 4 清扫桌面 MANO，服务地址 `192.168.1.18:29569`。在 leo1 上每次运行一种模式：

```bash
cd /home/liberai/bangguo/Liberai_Deploy_5090_openbottle

# 同步
bash ops/sweeptable_5090_leo1/evaluate.sh --inference-mode sync --post-chunk-sleep 1.5

# 异步执行：无 chunk sleep，按观测时间过滤过期 action
bash ops/sweeptable_5090_leo1/evaluate.sh --inference-mode sync-no-sleep
```

两种模式都会先复位机器人，再等待 Enter 开始推理。Task 4 专属初始/复位文件已经同步并通过加载验证。目录名保留 openbottle，但本入口明确使用 task 4 配置。

本次检查相机已就绪，但未收到机械臂、手部状态；现场测试前请启动或恢复状态与轨迹执行节点。

异步模式可加 `--async-sleep-steps N`：每次发布有效 action 后等待 `N / action_fps` 秒，再获取新观测推理；30 Hz 下 N=10 等待约 0.333 s。默认 N=0。N 的单位是单个 action 步，不是整段 32 步 chunk。相机延迟修正、执行延迟过滤和原始 action deadline 保持不变；整段过期而未发布时不等待。该参数仅作用于 `sync-no-sleep`，也可在 YAML 中设置 `runtime.async_sleep_steps`。

```bash
bash ops/sweeptable_5090_leo1/evaluate.sh --inference-mode sync-no-sleep --async-sleep-steps 10
```

这保证的是发布后留出 N 个 action 周期再开始下一次推理；若过滤后剩余 action 不足 N 步，不会补齐或延长轨迹。参数较大时，当前轨迹可能在下一段推理完成前执行完。

可同时开启 `--temporal-ensemble`，关闭用 `--no-temporal-ensemble`（默认关闭，YAML：`runtime.temporal_ensemble: false`）：

```bash
bash ops/sweeptable_5090_leo1/evaluate.sh --inference-mode sync-no-sleep --async-sleep-steps 10 --temporal-ensemble --temporal-ensemble-max-chunks 3
```

`--temporal-ensemble-max-chunks K` 控制最多融合最近 K 段（包含当前段），仅在开启 temporal ensemble 时生效。K=1 只使用当前预测，K=2 使用当前段与最近一段历史，K=3 最多融合三段；K=0 默认不限制。每个执行时刻仍只融合所选 chunk 中覆盖该时刻的预测，不足 K 段时按实际数量平均，不会补齐。YAML 对应 `runtime.temporal_ensemble_max_chunks`。

融合只用于异步模式：按修正后的观测时间加 action 偏移，对齐多次预测中相同执行时刻的绝对动作；历史轨迹只在重叠区间内插值。位置和手指关节等权平均，旋转通过 SLERP 插值和四元数平均处理，不直接平均旋转向量。融合后仍按相机、推理与执行延迟过滤过期 action，不改变 deadline；切换任务指令时清空历史。N=0 可单独对比融合效果。融合可能减轻预测跳变，但不填补轨迹断档，实际流畅程度以现场测试为准。

可用 `--action-fps 60` 将发布轨迹的 action 间隔改为 1/60 s；同步、异步均生效。默认继承 `--fps`（现场为 30），可在 YAML 设置 `runtime.action_fps`，`null` 表示继承。观测历史和录像仍使用原来的 fps。

```bash
bash ops/sweeptable_5090_leo1/evaluate.sh --inference-mode sync-no-sleep --action-fps 60 --async-sleep-steps 10 --temporal-ensemble --temporal-ensemble-max-chunks 3
```

异步 deadline 为修正后的观测时间 + 原始 action 行偏移 / action_fps；相机和执行延迟仍按秒处理，融合按新 deadline 对齐。N 步等待也变为 N / action_fps：60 Hz、N=10 等待约 0.167 s。同步的 `--post-chunk-sleep` 仍为独立的固定秒数。相同 32 行预测在 60 Hz 下覆盖 0 到 31/60 ≈ 0.517 s，较 30 Hz 压缩一半，异步过期行可能增多；启动日志显示 action_fps 和 action_dt。

完整部署说明和最新延迟见 [现场任务 4 说明](../ops/sweeptable_5090_leo1/README.md)。下方保留通用部署和原有任务入口；当前现场优先使用本节命令。

模型：`job-p70rofjkpwny` / `imagewam-dexbench-task4-mano-h32-368x320-bs512-10k-2n-b30z-20260829-085636` / `step_010000.pt`

## 1. 同步代码

```bash
SSH_CONFIG_PATH="${HOME}/.ssh/config"
mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"
touch "${SSH_CONFIG_PATH}"
chmod 600 "${SSH_CONFIG_PATH}"

if ! grep -Eq '^[[:space:]]*Host[[:space:]]+baiduyun32([[:space:]]|$)' "${SSH_CONFIG_PATH}"; then
  SSH_CONFIG_TMP="$(mktemp)"
  {
    printf '%s\n' \
      'Host baiduyun32' \
      '  HostName 180.76.149.31' \
      '  User root' \
      '  Port 8608' \
      ''
    sed -n '1,$p' "${SSH_CONFIG_PATH}"
  } > "${SSH_CONFIG_TMP}"
  install -m 600 "${SSH_CONFIG_TMP}" "${SSH_CONFIG_PATH}"
  rm -f "${SSH_CONFIG_TMP}"
fi

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
  baiduyun32 'echo SSH_OK'

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
  baiduyun32:/mnt/pfs/pfs-yc2F4O/bangguo/code/Liberai_Deploy/ \
  "${DEPLOY_ROOT}/"

cd "${DEPLOY_ROOT}"
source "${DEPLOY_ROOT}/ops/dex_bench_task_4_mano/alias.sh"
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

```bash
dex_bench_task_4_eval_imagewam_mano
```

# PickPlace Bench v22 ImageWAM Q20 真机评测

- Job ID：`job-jcqr9fmtvpng`
- 任务名：`imagewam-pickplace-v22-q20-ee1-hand02-15k-1n-b30z-20260825-091114`
- checkpoint：`step_015000.pt`
- 动作：relative EE、absolute q20 hand
- 模型输入分辨率：368×320

## 0. 推理变体与服务端

三种评测使用同一个 checkpoint，但每个 server 的推理配置固定，避免客户端误改配置：

| 变体 | image 更新 | action 更新 | 远程端口 | 机器人本地端口 |
|---|---:|---:|---:|---:|
| 基线 | 10 | 10 | 29565 | 29551 |
| IS5 | 5 | 5 | 29563 | 29552 |
| image5/action10 | 前 5 步更新，随后冻结 | 10 | 29564 | 29553 |

`image5/action10` 使用 10-step schedule：前五步联合更新 image/action；后五步固定第 5 步 image forward 的 K/V，只继续 action 分支。它不是把 image 单独改成一条完整的 5-step schedule。

截至 2026-08-27，三种配置分别运行在 tmux 会话 `imagewam-pickplace-base`（GPU 2）、
`imagewam-pickplace-is5`（GPU 3）和 `imagewam-pickplace-split`（GPU 1）中。可用
`tmux list-sessions` 查看；不要重复启动同端口服务。

服务端管理员在 `wam-clean-unified/Liberai-Model` 下执行以下命令。两个变体需要两张可用 GPU，可以同时启动：

```bash
cd /mnt/pfs/b6j7ah/bangguo/code/wam-clean-unified/Liberai-Model

export IMAGEWAM_SITE_ENV=/dev/null
export CONDA_ENV=/mnt/pfs/b6j7ah/bangguo/conda-envs/imagewam-torch212-cu130
export CONDA_SH=/mnt/pfs/b6j7ah/bangguo/miniforge3/etc/profile.d/conda.sh
export IMAGEWAM_CKPT_ROOT=/mnt/pfs/b6j7ah/bangguo/aihc_assets/imagewam
export FLUX2_SRC="$PWD/third_party/flux2"
export FLUX2_MODEL_PATH="$IMAGEWAM_CKPT_ROOT/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
export FLUX2_AE_MODEL_PATH="$IMAGEWAM_CKPT_ROOT/flux2/FLUX.2-dev/ae.safetensors"
export FLUX2_QWEN3_MODEL_SPEC="$IMAGEWAM_CKPT_ROOT/Qwen3-4B-official"

checkpoint="$PWD/experiments/train/imagewam-pickplace-v22-q20-ee1-hand02-15k-1n-b30z-20260825-091114/checkpoints/weights/step_015000.pt"
serve_dir=/mnt/pfs/pfs-yc2F4O/bangguo/imagewam-experiments/serve/pickplace-v22-q20
mkdir -p "$serve_dir"

# 将 0、1、2 换成三张实际空闲 GPU。
nohup env CUDA_VISIBLE_DEVICES=0 \
  bash shortcuts/iwam/inference/serve.sh \
  --checkpoint "$checkpoint" \
  --port 29565 \
  --num-inference-steps 10 \
  --image-num-inference-steps 10 \
  > "$serve_dir/image10-action10.log" 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 \
  bash shortcuts/iwam/inference/serve.sh \
  --checkpoint "$checkpoint" \
  --port 29563 \
  --num-inference-steps 5 \
  --image-num-inference-steps 5 \
  > "$serve_dir/image5-action5.log" 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 \
  bash shortcuts/iwam/inference/serve.sh \
  --checkpoint "$checkpoint" \
  --port 29564 \
  --num-inference-steps 10 \
  --image-num-inference-steps 5 \
  > "$serve_dir/image5-frozen-action10.log" 2>&1 &
```

不要让三个 server 绑定同一张 GPU。启动后确认：

```bash
python3 - <<'PY'
import socket
for port in (29565, 29563, 29564):
    with socket.create_connection(("127.0.0.1", port), timeout=2):
        print(f"port {port}: ready")
PY
tail -n 20 "$serve_dir/image10-action10.log"
tail -n 20 "$serve_dir/image5-action5.log"
tail -n 20 "$serve_dir/image5-frozen-action10.log"
```

日志必须分别显示 `num_inference_steps=5 image_num_inference_steps=5` 和 `num_inference_steps=10 image_num_inference_steps=5`。

### 0.1 Client/server 离线 MSE（2026-08-27）

从训练配置定义的原始 split 中，以 seed 42 在有效 frame anchor 内无放回抽样：train 100 帧、
validation（代码中名为 `eval`）100 帧。三种配置使用同一个 `step_015000.pt`、相同的 200 个
frame、GT 和 server seed 42，通过正式 `liber_teleop_v22` client/server 协议推理。

| 推理配置 | train 加权 MSE | validation 加权 MSE | train 未加权 MSE | validation 未加权 MSE |
|---|---:|---:|---:|---:|
| image10/action10 | 0.000176798 | 0.004655334 | 0.000466336 | 0.011431858 |
| image5/action5 | 0.000172366 | 0.004650263 | 0.000453710 | 0.011442924 |
| image5/action10 | **0.000169419** | **0.004605475** | **0.000449923** | **0.011373275** |

这里的加权 MSE 是归一化 action 上与训练配方一致的 EE=1.0、Q20 hand=0.2 macro MSE；
未加权 MSE 同样只统计有效 action 值。它们比较最终去噪预测与 canonical GT，不是 diffusion
flow-matching 训练 loss。相对 image10/action10，image5/action10 的加权 MSE 在 train/validation
分别降低 4.17%/1.07%；image5/action5 分别降低 2.51%/0.11%。

完整样本 manifest、逐样本预测和指标位于：
`/mnt/pfs/pfs-yc2F4O/bangguo/imagewam-experiments/eval/pickplace-v22-q20-step015000-inference-variants/`。

58 个有效动作维度（左右 EE 各 9D、左右 Q20 hand 各 20D）的曲线对比位于同目录的
`plots_58d/`：validation 固定抽取 5 个样本，每种推理配置各有 5 张 pred-vs-GT 30-step
horizon 曲线图，并附 `summary_58d_mse.png` 和可复现选择信息 `manifest.json`。

## 1. 同步代码

以下命令均在机器人本地评测机执行。完整复制下面整段命令并粘贴到终端中：

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

ssh \
  -o BatchMode=yes \
  -o ConnectTimeout=10 \
  -o StrictHostKeyChecking=accept-new \
  baiduyun32 'echo SSH_OK'

mkdir -p "${HOME}/bangguo/Liberai_Deploy"
rsync -av \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.venv/' \
  --exclude='records/' \
  baiduyun32:/mnt/pfs/pfs-yc2F4O/bangguo/code/Liberai_Deploy/ \
  "${HOME}/bangguo/Liberai_Deploy/"

source "${HOME}/bangguo/Liberai_Deploy/ops/pickplace_v22_q20/alias.sh"
```

命令执行完成后，应看到：

```text
SSH_OK
已加载一键评测命令：
  pickplace_eval_imagewam_q20                 # image10 / action10
  pickplace_eval_imagewam_q20_is5             # image5 / action5
  pickplace_eval_imagewam_q20_image5_action10 # image5 frozen / action10
```

如果没有看到 `SSH_OK`，不要继续评测，让服务器管理员检查 SSH key 或端口。
每次新开一个机器人终端，都需要重新执行：

```bash
source "${HOME}/bangguo/Liberai_Deploy/ops/pickplace_v22_q20/alias.sh"
```

## 2. 运行评测

确认远程模型服务已经由管理员启动。同一时刻只运行一个真机 client。

在机器人本地终端执行三个命令之一：

```bash
# 基线：image10 / action10
pickplace_eval_imagewam_q20

# 变体 1：image5 / action5
pickplace_eval_imagewam_q20_is5

# 变体 2：image 前5步后冻结 / action10
pickplace_eval_imagewam_q20_image5_action10
```

命令会自动完成以下操作：

1. 显示 8 个 PickPlace 摆台任务；
2. 根据输入的数字选择对应英文指令和训练集 mean init-state；
3. 检查是否已经有另一个 client 在运行；
4. 检查远程服务并建立 SSH tunnel；
5. 加载 ROS 环境并启动 Q20 真机推理；
6. 基线录像沿用原目录；两个变体分别保存到
   `~/bangguo/recordings/policies/pickplace-v22-q20/image5-action5/` 和
   `~/bangguo/recordings/policies/pickplace-v22-q20/image5-frozen-action10/`。

客户端会检查 server 返回的 total/image step 配置；如果 tunnel 连到了错误端口，会在机器人动作执行前直接报错退出。

看到任务菜单后，输入与当前摆台一致的数字并按回车：

```text
1. 香蕉放进碗里
2. 香蕉放到盘子上
3. 香蕉放到红辣椒上
4. 杯子放到盘子上
5. 绿苹果放到杯子上
6. 绿苹果放进碗里
7. 绿苹果放到盘子上
8. 红辣椒放进碗里
```

机器人完成 reset 后，确认场地、机器人姿态和急停状态安全，再按一次回车正式开始。
使用 `Ctrl-C` 结束本次评测。

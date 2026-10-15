# V5：Astra 先想象，再执行

V5 继承 V4 的真实 Progress VLM 自动切 L2、可选停滞接管、Astra 自动交回 WAM。只在 Astra 候选动作下发前增加 Dynamics 想象。WAM 不等待 Dynamics；生成图像绝不写入真实 VLM 进度日志。

## 当前实现与部署

- PC 5090 的 WAM 29572 和 Progress VLM 29573 保持原样。
- PRO6000 `10.128.39.115:29640` 仍是现有 40k canonical Dynamics 服务。
- 新增 CPU 适配入口 `10.128.39.115:29642`：接收 Leo3 已转换的 MANO 字段和原始 RGB，调用训练仓库的 `AdapterTensorAssembler`、`LerobotSampleCore._normalize`、`build_mosaic`，再请求 29640。不加载第二份模型、不改原 canonical 接口。
- Leo3 使用仓库内 `TeleopManoFieldConverter` 转换真实状态与候选动作，复用已有平移线性／旋转 SLERP 插值工具；不将 52D 直接补零到 326D。机器人观测是 RGB、已按现有相机方向约定处理，不走 WAM 的额外图像预处理。
- 默认 **preview**：WAM 仍会正常执行！进入 Astra 后只生成／展示预测，暂停，不下发 Astra 候选。R 可用新观测重新规划。它不是整个机器人无动作模式。
- **review**：Astra 审阅预测，批准精确候选后才下发。这条路径已用假机器人测试，尚未完成 V5 真机执行验收。

## 服务准备

PRO6000 上先确认 29640 已 READY。新增适配服务不占用它的监听端口。当前代码副本位于 `/home/pc/liberai-tmp/wangyi/v5-native-adapter`；日志 `gateway.log`。未运行 29642 时才执行：

```bash
cd /home/pc/liberai-tmp/wangyi/v5-native-adapter
nohup bash wam_direct/run_dynamics_gateway.sh > gateway.log 2>&1 < /dev/null &
```

脚本使用已有训练 Python 环境和训练仓库，加载固定 adapter/norm 文件并核验 SHA256。新机器需调整脚本路径。29642 只供受信任内网客户端使用；没有公网认证层。服务串行处理，客户端只允许一条在途请求，不排队累积候选。

## 启动 V5

在 Mac 的仓库根目录运行，保留原有“Mac 先启动、复制整段命令到 Leo3 桌面终端”的模式：

```bash
RUN_ID="apple_pepper_v5_$(date +%Y%m%d_%H%M%S)"
python3 wam_direct/start_mac_vlm.py \
  --run-id "$RUN_ID" --control-version 5 \
  --vlm-auto-takeover --dynamics-mode preview \
  --prompt "Task: Pick up the green apple and place it in the bowl, then pick up the red pepper and place it on the plate. Subtask: pick up the green apple" \
  --next-subtask "place the green apple in the bowl" \
  --next-subtask "pick up the red pepper" \
  --next-subtask "place the red pepper on the plate"
```

- 配置：`config/astra/version5_dynamics.json`。
- Leo3 tmux 名称：本次 RUN_ID；session：`/home/liberai/Liberai_Deploy_astra/outputs/astra/<RUN_ID>`。
- 第一次仍人工 Enter 开始。T/P/B/R 沿用 V4。
- 自动打开只读 `Dynamics-preview` 窗口，对比真实 anchor 与想象未来；Q/Esc 只关闭窗口，不停止机器人。
- 明确准备执行候选时，把 `--dynamics-mode preview` 换成 `review`。不能用模型生成画面当成实际成功的证据。
- 只打印命令加 `--dry-run`。生成的 Leo3 命令可用 `--dynamics-host`、`--dynamics-port`、`--dynamics-timeout` 调整适配服务地址和等待上限。

## 时序、覆盖范围与失效条件

第一版只覆盖 0.5–1 秒、终点对齐 30Hz 网格的前缀。当前 20fps、0.1s lead 下最多保留 18 个目标点，对应 1 秒。较长的 Astra 提案会截取该前缀；批准／执行的就是这 18 点，之后重新观察、规划，不能批准未预测的剩余轨迹。短于 0.5 秒且无法构成合法前缀时暂停，不补造动作。

候选按真实目标时间表，以 anchor 为起点重采样到 30Hz，再经 MANO 转换与训练相对 anchor 变换。H 后在归一化后置零并关闭 mask。第一版使用理想的目标轨迹插值；实际 ROS 规划器的历史插值起点、IK 与跟踪偏差需在真机验收中测量，不能声称想象等于真实执行。

状态链路：`WAITING` 接收 takeover → `IMAGINING` → `WAITING / REVIEW_IMAGINATION` → 批准后 `EXECUTING`。preview 模式在保存预测后进入暂停。每条候选绑定 session、condition、phase、UUID、动作 SHA256、fps 和 lead。T/B、退出、重新观察、新候选使原候选失效。批准必须复制候选 ID/hash、指定完整前缀长度且不包含新动作；修改动作必须重新想象。

批准前用新真实观测检查漂移：末端位置 2cm、旋转 0.15rad、手关节 0.15rad、三视角缩小图平均像素差 8%，候选最长 600 秒。这些是保守的陈旧结果过滤初值，不是碰撞检测或成功保证；真实场景变化可能要求重新观察。原轨迹检查与硬件故障检查继续生效。错误、超时或无合法候选不会自动放行 Astra 动作。

预测时刻是 anchor 加候选时长，实际下发发生在审阅之后。`imagination_execution` 记录批准候选和真实发布 receipt；发布时钟不能被当成原 anchor 时钟。执行后 `VERIFY_RECOVERY` 获取真实反馈，其时间戳必须单独核对，不能自动视为精确目标时刻的 GT。

## 记录与离线回放

session 的 `imagination/<candidate_id>/` 每次成功预测自动保存 `current.png`（真实 anchor 三视角拼图）、`future.png`（预测三视角拼图）、`panel.png`（左右对比）、带有效 mask 的归一化 `prediction.npz`，以及 `metadata.json`（L3/L2、phase、候选动作、时刻与文件路径）。preview/review 两种模式均保存，每次候选独立目录，不覆盖旧图；无需打开窗口。`imagination/index.jsonl` 按完成顺序索引各次预测。metadata 中目标时刻为假设从 anchor 开始执行的预测时刻，`status_at_save` 仅表示保存时尚未执行，不代表后续是否批准执行。`events.jsonl` 保存 `imagination_started/completed/execution` 与候选动作；`imagination/latest.json` 供只读窗口读取。原 VLM 窗口录像不包含这个新窗口，想象图会独立保存。

不用启动机器人即可回放已保存的 Astra 候选：

```bash
cd /home/liberai/Liberai_Deploy_astra
export PYTHONPATH=/opt/openrobots/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=/opt/openrobots/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python3 -m astra.dynamics_replay \
  --session outputs/astra/apple_pepper_v4_20260930_165443 \
  --output outputs/dynamics_v5/replay_001
```

回放只读历史文件，不导入 ROS、不创建发布器。该命令使用此历史部署的 20fps/0.1s lead，不能用来猜测其他 session 的时序。未来 state 是归一化 canonical 数据及 mask，不用于机器人控制；当前界面不展示未经反归一化的物理量。

## 验证范围

2026-10-02：真实录制观测与 Astra 目标在 Leo3 完成 MANO 转换，经新入口调用 40k 服务成功；首次端到端约 1.74s，模型约 0.58s。该结果证明传输与原生字段适配可运行，不证明预测准确性。原 8 样本 smoke（direct parity、H15 padding、mask、错误指纹）继续通过。

控制回归测试在假传感器／发布器下完成；训练环境另测拼图顺序、归一化 mask、padding 和错误输入。尚需人工启动 V5 preview，检查当前机器相机／动作语义和预测质量，再验收 review 模式真实执行。V4 配置与默认启动版本仍保留，不会自动切换到 V5。

图像颜色约定：Leo3 `env.get_obs()` 返回 BGR，`native_request` 在发送前转为 RGB；gateway、模型返回值与 PNG 保存均使用 RGB。离线回放先将 RGB PNG 恢复为 BGR 观测，再走同一入口，避免重复转换。

V5 审阅期间手指关节偏差容许值为 0.15 rad（此前 0.05 rad），与想象 anchor 的手指检查一致。手臂及图像检查不变；V4 默认值不变。日志会显示触发通道和 limit。

# Astra-For-Policies

让 VLM、Astra 与 WAM 协同控制机器人：WAM 负责常规动作，VLM 判断当前 L2 进度，Astra 接管恢复后可自动交还 WAM。当前推荐使用 V4，支持一个 L3 总任务和多条有序 L2 子任务。

## 先让 Codex 了解项目

在仓库目录打开 Codex，告诉它：

> 请先读取 `astra/SUPERVISOR.md` 和 `docs/astra_mode2.md`，遵守 `AGENTS.md`。解释 V1/V2/V3/V4 的区别，并根据我的机器环境给出启动命令；先不要启动机器人。若使用 Mac 监督，再读取 `docs/mac_supervisor.md`；V4 同时读取 `docs/progress_vlm.md`。

- **V1**：Astra 审阅每批模型动作，可决定接管和交还。
- **V2**：模型直接执行；你按 **T** 让 Astra 接管，再按 **T** 交还模型。
- **V3**：你按 **T** 让 Astra 接管；Astra 判断恢复后可自动交还模型，也可按 **T** 人工交还。
- **V4**：VLM 自动切下一 L2；可显式启用 VLM 停滞接管，Astra 恢复后自动交还 WAM。

## V5 实验版：Dynamics 想象

在 V4 上增加 Astra 候选的执行前想象：真实观测和候选动作 → PRO6000 Dynamics → Astra 审阅 → 执行同一前缀。使用 `--control-version 5`，默认 `--dynamics-mode preview` 只预览 Astra 候选；WAM 仍正常运行。`review` 模式才允许审阅后执行。V4 仍是默认版本。部署、完整命令、时间对齐和验证范围见 [V5 使用说明](docs/dynamics_v5.md)。

## 当前推荐：V4 · VLM + Astra + WAM

Windows 笔记本现可原生运行 Astra 监督桥接，无需 WSL/Bash；仍使用 `python wam_direct/start_mac_vlm.py`，自动选择平台分支。Linux/macOS 原有入口保留。环境检查、PowerShell 命令及兼容边界见 [Windows 监督说明](docs/windows_supervisor.md)。

| 位置 | 运行内容 | 当前部署 |
| --- | --- | --- |
| PC | WAM 动作服务、独立 40k Progress VLM 服务 | `10.128.40.203:29572` / `:29573` |
| Leo3 | 相机、机器人 runner、VLM 观察客户端、窗口录像 | `/home/liberai/Liberai_Deploy_astra` |
| Mac | Astra 监督会话，通过 SSH 邮箱与 Leo3 通信 | SSH 别名 `leo3` |

VLM 观察客户端发送五图输入（历史主视角、L2 首帧、当前三视角），记录进度和 replay 标签；它本身不发布机器人动作。runner 在当前动作批次结束后检查绑定的 VLM 记录，决定切换 L2 或请求 Astra 接管。Astra 根据新观测返回恢复动作或 `hand_back`；交还时清除旧动作，保留当前 L2 和剩余预算，由 WAM 重新推理。

### 1. 准备现有服务

- PC 的 WAM 和 40k VLM 服务须已就绪；VLM 独立使用 `29573`，不占用 WAM socket。当前 VLM 导出目录为 `/data/trans/progress_40k_20260930`，使用既有 `dp` 环境。模型权重、匹配的模型运行代码、ROS 工作区与机器认证不包含在本仓库，服务启动方式见 [Progress VLM 文档](docs/progress_vlm.md#pc-服务管理)。
- Leo3 按既有流程启动机器人和三路相机，当前 ROS domain 为 `103`。桌面需可打开终端、tmux 和 OpenCV 窗口。
- Mac 的 Codex/Astra 与 `ssh leo3` 须可用。两端使用匹配的源码版本；辅助脚本中的路径、IP、ROS 和 OpenRobots 环境是当前部署配置，其他机器需相应调整。

### 2. 在 Mac 的仓库根目录启动

以下是已测试的“绿苹果入碗，再将红椒放盘”四条 L2 示例：

```bash
RUN_ID="apple_pepper_v4_$(date +%Y%m%d_%H%M%S)"
python3 wam_direct/start_mac_vlm.py \
  --run-id "$RUN_ID" \
  --control-version 4 \
  --vlm-auto-takeover \
  --prompt "Task: Pick up the green apple and place it in the bowl, then pick up the red pepper and place it on the plate. Subtask: pick up the green apple" \
  --next-subtask "place the green apple in the bowl" \
  --next-subtask "pick up the red pepper" \
  --next-subtask "place the red pepper on the plate"
```

保留 Mac 终端。脚本创建本次 RUN_ID 对应的 Mac 桥接，打印 Leo3 的完整启动命令，并保存到 `outputs/mac_supervisor/launch_<RUN_ID>/leo3_command.sh`。它不会远程启动机器人或代按按键。只查看命令可加 `--dry-run`。

### 3. 复制打印的整段命令到 Leo3 桌面终端

不要在 Leo3 重新生成 RUN_ID。生成的命令已包含 ROS/OpenRobots 环境、`--vlm-progress`、V4 配置和本次 `predictions.jsonl` 路径。

- 版本：**V4**，配置 `config/astra/version4_vlm_subtasks.json`。
- tmux 名称：Mac 打印的 **RUN_ID**。
- runner session：`/home/liberai/Liberai_Deploy_astra/outputs/astra/<RUN_ID>`。
- 等待 Mac 桥接和客户端就绪后，在 Leo3 tmux 主控制窗口按 **Enter** 开始。每次新测试使用新 RUN_ID。

第一条 L2 来自 prompt 的 `Subtask:`；后续使用重复的 `--next-subtask`，不使用 `--plan`。`--original-step-limit 400` 在当前设置中对应总预算 1200 步，跨所有 L2 共用，不在切换或复位时清零。

### V4 自动切换规则

| 行为 | 默认条件 |
| --- | --- |
| 下一条 L2 | 当前阶段至少 3 秒；连续 2 次新鲜、不同图像的有效预测达到 100%，且不是 anomalous replay；通过时间间隔与会话身份校验 |
| VLM 请求 Astra 接管 | 显式加 `--vlm-auto-takeover`；当前 L2/阶段至少 10 秒；12 秒无进度新高；最近 8 次有效预测覆盖至少 6 秒；replay 仅记录，不作为门槛；当前进度低于 90% |
| Astra 交回 WAM | Astra 根据最新图像、实测状态和恢复证据返回 `hand_back`，不是固定 progress 阈值；交还后自动接管冷却 15 秒 |

最新接管图像必须仍新鲜；过期、重复、断流或错误结果不能直接授权接管。同一阶段已验证的进度最高值会保留，避免批次延迟读取清空停滞历史。完整时序条件见 [Progress VLM 文档](docs/progress_vlm.md)。当前没有独立的“持续回退即可接管”规则。

不加 `--vlm-auto-takeover` 时，V4 仍自动切 L2，但 Astra 接管由 T 触发。`--control-version 3` 回到只观察进度的 V3。首轮、手动 P、B 后仍需 Enter；最后一条 L2 不自动循环或宣布完成。`progress` 是当前 L2 的预测档位，不是 L3 成功率；replay 也不等于物理失败。

### 查看进度、监督和录像

- Leo3 的 VLM 图形窗口显示 L3/L2、progress、replay、Auto L2 / Auto Astra 和录像状态；tmux `VLM-progress` 窗口保留观察日志。
- tmux `Astra-supervisor` 窗口查看监督状态；runner session 内的 `events.jsonl`、`status.json`、`supervisor.json`、`supervisor.log` 用于对齐控制切换与决策。
- VLM 记录位于 `/home/liberai/Liberai_Deploy_astra/outputs/progress_vlm/<RUN_ID>_follow/`，包含 `predictions.jsonl` 与实际输入图。
- `window_video/` 自动录制 1200×660、10 fps、每段 60 秒的 `window_0001.mp4` 等文件；`segments.jsonl` 记录分段是否完成。结束时在观察窗口按 Q/Esc 正常封存最后一段，再按顺序拼接。窗口录像是连续主视角与最近推理输入的合成画面，腕部不是独立连续录像。
- Astra 暂停可按 R 获取新观测并重试；T 可人工交还。不要向同一个后台监督会话并行发送消息或启动第二个桥接。

可选的深度判断与参考视频输入分别见 [深度说明](docs/depth_judgment.md)、[参考视频说明](docs/reference_video.md)。默认 V4 示例没有开启这些选项。

## 旧版 V1 / V2 / V3：选择启动方式

- **机器人电脑已配置好 Codex/Astra**：用方式 A，只需机器人端启动，不需要 Mac、SSH 桥接或复制两端 RUN_ID。
- **Astra 跑在 Mac**：用方式 B，Mac 与机器人分别启动。

下面两种方式都是现有 **leo3 + 5090** 部署的 V2 示例，选择一种即可。先准备好 ROS 与 WAM 服务；其他机器需替换仓库路径和模型服务地址。模型权重与 ROS 工作区不在本仓库。

## 方式 A：机器人本机运行 Astra

机器人电脑上的 Codex/Astra 需已能正常调用模型。在机器人桌面终端执行：

```bash
cd /home/liberai/Liberai_Deploy_astra

RUN_ID="pepper_v2_local_$(date +%Y%m%d_%H%M%S)"
echo "$RUN_ID"

python3 wam_direct/launch_tmux.py \
  --name "$RUN_ID" \
  --client astra -- run \
  --supervisor-backend local \
  --control-config config/astra/version2_keyboard.json \
  --host 10.128.40.203 \
  --port 29572 \
  --fps 20 \
  --prompt "Task: Pick up the red pepper and place it in the bowl. Subtask: pick up the red pepper" \
  --next-subtask "place the red pepper in the bowl" \
  --session "/home/liberai/Liberai_Deploy_astra/outputs/astra/$RUN_ID" \
  --owner "$RUN_ID" \
  --original-step-limit 400
```

runner 会自动建立本机专属 Astra 监督会话，无需另开 Codex 对话接收请求。等待就绪，在弹出的 tmux 主窗口按 **Enter** 开始；监督日志在 `Astra-supervisor` 窗口查看。

这里的 `local` 指 **Astra 在机器人电脑运行**；WAM 仍连接 `--host` / `--port` 指定的模型服务。

## 方式 B：Mac Astra + 机器人客户端

Mac 需已配置好 Codex，以及到机器人的 SSH（当前默认主机名为 `leo3`）。

### 1. Mac 终端

进入本仓库目录，执行：

```bash
RUN_ID="pepper_v2_mac_$(date +%Y%m%d_%H%M%S)"
echo "$RUN_ID"

bash wam_direct/run_mac_supervisor.sh \
  --session "/home/liberai/Liberai_Deploy_astra/outputs/astra/$RUN_ID"
```

复制打印的 RUN_ID，等待 `MAC READY`。保留这个终端。

### 2. 机器人桌面终端

两端必须使用同一个 RUN_ID；每次新测试使用新名称。

```bash
RUN_ID="这里替换为Mac打印的RUN_ID"

cd /home/liberai/Liberai_Deploy_astra

python3 wam_direct/launch_tmux.py \
  --name "$RUN_ID" \
  --client astra -- run \
  --supervisor-backend mac \
  --control-config config/astra/version2_keyboard.json \
  --host 10.128.40.203 \
  --port 29572 \
  --fps 20 \
  --prompt "Task: Pick up the red pepper and place it in the bowl. Subtask: pick up the red pepper" \
  --next-subtask "place the red pepper in the bowl" \
  --session "/home/liberai/Liberai_Deploy_astra/outputs/astra/$RUN_ID" \
  --owner "$RUN_ID" \
  --original-step-limit 400
```

启动器会弹出 tmux 窗口，准备好后在主窗口按 **Enter**。

## 两种方式通用的参数

- 换 **V3**：将配置改为 `config/astra/version3_auto_handback.json`。
- 换 **V1**：使用 `config/astra/version1_review.json`，删除 `--next-subtask`；具体审阅与初始子任务要求让 Codex 按上述文档确认。
- `Task` 是整局任务 L3，`Subtask` 是你指定的初始 L2；`--next-subtask` 是后续子任务，可重复指定。

## 按键

在机器人 tmux **主控制窗口**输入。

| 操作 | 功能 |
| --- | --- |
| Enter | 首次开始；选择子任务或复位后确认继续 |
| T | V2/V3/V4 切换 WAM 与 Astra，等待当前推理/动作批次边界 |
| P → Enter | V2/V3/V4 在 WAM 阶段选择并确认下一个子任务 |
| R | V2/V3/V4 的 Astra 暂停后，用新观测重新评估；不会直接发布动作 |
| B | 回到任务初始姿态与第一个子任务，再按 Enter 开始；预算不清零 |
| Ctrl+C | 退出当前 runner；重新测试需要新 session |
| ROS 踏板 | 沿用机器人原有急停功能；B 和 Ctrl+C 不是急停 |

## 暂停后恢复

**Astra 暂停**优先按 R。**WAM 因模型断连等问题暂停**时，先处理原因，再使用下面的 `resume`。

保留机器人 runner/tmux；方式 B 还需保留 Mac 监督终端。在机器人上另开终端，查看当前状态（两种方式相同）：

```bash
cd /home/liberai/Liberai_Deploy_astra
SESSION="/home/liberai/Liberai_Deploy_astra/outputs/astra/这里替换为本次RUN_ID"

python3 -m astra.run status --session "$SESSION"
```

从输出复制本次的 `session_id`、`pause_id`。确认仍为 `PAUSED_FOR_OPERATOR`，并已准备好继续：

```bash
python3 -m astra.run resume \
  --session "$SESSION" \
  --session-id "这里替换为当前session_id" \
  --pause-id "这里替换为当前pause_id" \
  --reason "Operator ready to reconnect WAM and continue"
```

**WAM 恢复成功会直接重新推理并继续动作，不再等待 Enter。** 不要复制别人的 ID 或上一次暂停的 ID；已退出的 runner 无法用 resume 复活。若再次暂停，让 Codex 读取本次 session 日志定位原因。

### V6 点云几何想象

V4 加 Astra 候选动作的 URDF 手部点云预演，无需 Dynamics 服务。先启动 `bash wam_direct/view_pointcloud_v6.sh`，再用 Mac helper 的 `--control-version 6 --pointcloud-mode review` 生成本轮 Leo3 命令。使用、颜色与局限见 [V6 文档](docs/pointcloud_v6.md)。

### V7 整段预演与前缀执行

使用 `--control-version 7 --pointcloud-mode review`。Astra 最多规划120点，RViz预演整段后审批有限前缀；执行后重新观察、重规划或交还WAM。复用V6点云窗口和录制。启动、阶段格式、时长/距离限制与能力边界见 [V7 文档](docs/pointcloud_v7.md)。

### V8 六方向与 policy 候选比较

基于 V5，Astra 先选左/右手；Leo3 从同一真实观测生成该手 `chest_link` 六方向平移（默认10cm）和一条新 policy 前缀，逐条经 Dynamics 预测，由 Astra 选一条执行，再观察真实反馈。Windows、Linux/macOS 监督方式均保留。使用 `--control-version 8 --dynamics-mode review`；单 L2 任务加 `--single-subtask`。默认 preview 仅阻止 Astra 候选下发，WAM 仍运行。详见 [V8 文档](docs/dynamics_candidates_v8.md)。尚未进行 V8 真机验收。

V4 / V8 可在启动时用 `--astra-model astra|opus5.5|gemini|glm5.3-flash` 固定接管模型；也可编辑 `wam_direct/start_drawer_test.py` 的 `MODEL` / `CONTROL_VERSION` 后启动。后三者走 QuickRouter，默认仍为原 Astra。凭据、两端配置与验证范围见 [接管模型配置](docs/supervisor_models.md)。

### V4–V7 版本边界

四个版本通过 control-version 和独立配置选择功能，但共用 runner、VLM 接管、通信与动作校验，并非完全冻结的独立实现。

| 版本 | Astra 候选处理 | 外部想象服务 |
| --- | --- | --- |
| V4 | 原接管动作执行与真实观测反馈 | 无 |
| V5 | Dynamics 预测；preview 或绑定候选的 review | Dynamics gateway/model |
| V6 | URDF 点云预演；最多1秒候选后审阅 | Leo3 场景点云/TF |
| V7 | 最多120点整段预演，执行有界前缀后丢弃尾段 | 与V6相同，无Dynamics |

当前启用 `--vlm-auto-takeover` 的 V4–V7 均使用无 replay 门槛的停滞规则，replay仅记录。V5–V7手指审阅漂移容差0.15rad，V4仍为0.05rad。V6/V7共用RViz配置及实时投影显示。120点和阶段前缀契约仅V7生效；默认helper仍为V4。

点云启动编排已纳入 `wam_direct/view_overlay_runtime.sh`，依赖Leo3现有camera_pose发布器和标定；仓库入口无需手工修改外部view_overlay.sh。当前尚未实现“规划前自动附点云、Astra请求换观察角度”的讨论方案；现有点云审阅仍在候选动作生成后。

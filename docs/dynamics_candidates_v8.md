# V8：Astra 选手，六方向 + policy，Dynamics 比较后执行

V8 在 V5 的 Dynamics 链路上增量实现。Windows 笔记本运行 Astra 监督，Leo3 运行机器人客户端，WAM / VLM / Dynamics 复用已有服务器。Linux / macOS 监督入口继续保留，默认 helper 版本仍为 V4。

## 每轮接管流程

1. WAM 正常运行；T 或显式启用的 VLM 停滞规则将控制权交给 Astra。
2. Astra 根据真实图像、当前测量状态和 L2，先返回 `propose_candidates`，选择 `active_hand: left` 或 `right`。
3. Leo3 取得新观测并核对选手时的场景，在这份观测上请求一次当前任务的 WAM policy。不会重用先前丢弃的 policy 队列。
4. 从同一状态生成选定手在 `chest_link` 下的 `+X/-X/+Y/-Y/+Z/-Z` 六条平移动作，加入一条新 policy 前缀，共七条候选。平移用余弦曲线平滑到达端点，默认 10 cm，可用 `--candidate-distance-m` 设置，范围 `(0, 0.10]` 米。
5. 六条方向动作保持手腕旋转、两手手指和另一只手臂不变。**Policy 候选保留原始 52D 输出，可能改变两臂和手指；Astra 的选手只约束六方向候选。** 抓取、松手及姿态调整主要由 policy 候选提供。
6. 所有候选共用时钟和长度，先进行现有轨迹/状态检查，再逐条通过 MANO 原生适配请求 Dynamics。每次仅一条请求在途。校验拒绝或预测失败的候选会显示原因，不可选中；全部失败则暂停。
7. `REVIEW_CANDIDATES` 将带候选 ID 和方向标签的预测图传到 Windows。Astra 比较真实观测、各候选预测和任务目标，只批准其中一条。
8. 复核 ID、动作哈希、时钟、控制权、预算、场景漂移和硬件状态后执行完整预测前缀。其余候选全部失效。获得真实反馈后重新选手、生成候选，或按原规则交还 WAM。

这仅改变 **Astra 接管阶段**；WAM 常规运行阶段沿用 V4/V5，不增加 Dynamics 审阅。预测图不写入真实 Progress VLM 进度日志。

## 时序与决策契约

沿用 V5 的 0.5–1 秒预测范围（含 lead），终点对齐 Dynamics 30 Hz 网格。在 20 fps、0.1 秒 lead 下通常是 18 个目标点、总时长 1 秒。方向候选在最后一个点到达完整指定距离；policy 取相同长度的原始前缀。剩余预算或 policy 长度不能构成合法前缀时不补造动作。

每个决策仍须复制当前请求的 `session_id`、`owner`、`request_id` 并包含非空字符串 `reason`、`evidence`。

选手决策额外字段：

```json
{"operation":"propose_candidates","active_hand":"right"}
```

审批额外字段（ID/hash 必须从当前 `imagination` 复制）：

```json
{"operation":"approve","candidate_set_id":"当前集合ID","candidate_id":"选中候选ID","candidate_sha256":"选中动作哈希","execute_steps":18}
```

V8 禁止直接提交 `takeover`、`correct` 或 `actions`。不得拼接多个候选、编辑坐标、缩短已预测前缀。可用 `observe` 获取新观测，或再次 `propose_candidates` 更换操作手/重建整组。T、B、重新观察和新集合使旧候选失效；不能重放已批准的集合。取消后的在途 RPC 有超时界限，不允许新集合与其并发。

沿用 V5 的 anchor 校验：位置 2 cm、旋转 0.15 rad、手指 0.15 rad、三视角缩小图平均差 8%，最长 600 秒；审批与真正下发之间再做原有状态与动作检查。它们是陈旧结果过滤，不是完整碰撞检测或 IK 证明。硬件故障不解除、不自动交还。

`--astra-pause-handback` 继续按已有操作者配置处理模型主动暂停，包括 `planning`、`collision_risk`、`observation_uncertainty`；交还记录 `recovery_confirmed=false`。通信错误、预测失败、硬件故障等技术暂停不会冒充模型规划暂停放行。正常 `hand_back` 和 `complete` 必须基于真实反馈。

## Windows / Leo3 启动

以下是准备启动的命令，开发验证不会自动执行真机启动。先确认 ROS 和已有服务准备好。每轮新建 RUN_ID，不复用历史 session。默认 `preview` 只阻止 Astra 候选下发，**WAM 仍会动作**；正式审批执行须显式使用 `review`。

Windows PowerShell，使用“只开抽屉”任务：

```powershell
Set-Location C:\dynamics_eval\Liberai_Deploy_astra
$runId = 'drawer_open_v8_' + (Get-Date -Format yyyyMMdd_HHmmss)
python wam_direct/start_mac_vlm.py --run-id $runId --control-version 8 --single-subtask --dynamics-mode review --candidate-distance-m 0.10 --vlm-auto-takeover --astra-pause-handback --ssh-timeout 60 --ssh-connect-timeout 15 --ssh-retries 3 --prompt "Task: Pull the storage rack drawer outward by a moderate distance. Subtask: pull out the drawer"
```

仅查看命令不启动桥接时，加 `--dry-run`。`--single-subtask` 明确不追加默认的红椒第二子任务。保留 Windows 监督终端，将打印的整段 shell 命令复制到 **Leo3 桌面终端**。它会包含完整环境变量、同一 RUN_ID 的 VLM 日志和 session 路径，并通过以下入口启动：

```bash
python3 wam_direct/launch_tmux.py --name "$RUN_ID" --vlm-progress --dynamics-view --client astra -- run \
  --supervisor-backend mac --control-config config/astra/version8_dynamics_candidates.json \
  --host 10.128.40.203 --port 29572 --fps 20 \
  --prompt 'Task: Pull the storage rack drawer outward by a moderate distance. Subtask: pull out the drawer' \
  --vlm-progress-log "/home/liberai/Liberai_Deploy_astra/outputs/progress_vlm/${RUN_ID}_follow/predictions.jsonl" \
  --dynamics-mode review --candidate-distance-m 0.10 --vlm-auto-takeover --astra-pause-handback \
  --session "/home/liberai/Liberai_Deploy_astra/outputs/astra/${RUN_ID}" --owner "$RUN_ID" --original-step-limit 400
```

上面是核心命令；优先使用 helper 生成的完整环境块。Leo3 tmux 名为本轮 RUN_ID，runner session 为 `/home/liberai/Liberai_Deploy_astra/outputs/astra/<RUN_ID>`。首轮仍由操作者 Enter 启动，T/P/B/R 沿用原含义。版本配置独立，V5～V7 的启动方式保留。

Linux/macOS 笔记本仍可在相应仓库根目录用 `python3 wam_direct/start_mac_vlm.py` 和同样参数；脚本沿用 POSIX Bash 桥接。`--supervisor-backend mac` 是历史协议名称，也用于 Windows，不代表要求 macOS。

WAM/VLM 使用 `10.128.40.203:29572/29573`；Dynamics 使用 `10.128.39.115:29642` 原生 gateway 和后端 29640。runner 启动核验原有 adapter/norm 指纹，不创建第二份模型。可在生成的 Leo3 命令中添加 `--dynamics-host`、`--dynamics-port`、`--dynamics-timeout`，默认单候选 RPC 超时 30 秒。

## 记录和验证边界

每组保存至 `imagination/<candidate_set_id>/`：

- `request.json`：共同 anchor、操作手、距离、七条动作、policy 请求 receipt。
- 每个 `<candidate_id>/metadata.json`：来源、标签、哈希、动作、预测/失败状态；成功时另有真实 `current.png`、合成 `future.png`、带有效 mask 的 `prediction.npz`。
- `panel.png`：七条候选与共同真实 anchor 的对比图；`metadata.json`、`index.jsonl`、`latest.json` 供回放和只读窗口使用。
- `events.jsonl`：`candidate_set_started`、`candidate_set_predicted`、`candidate_selected` 和 `imagination_execution` / `batch_published` 可关联选中动作与真实发布 receipt。

合成图表示“假设执行这条候选后的预测”，不是执行证据。normalized state 不直接转为机器人命令。实际跟踪、接触与物体变化必须由下一轮真实观测确认。原 VLM 视频和这些静态预测文件分别保存。

离线验证命令（假机器人、假 policy 和假 Dynamics，不连接 ROS）：

```bash
python3 -m unittest discover -s tests -p 'test_dynamics_candidates_v8.py' -q
```

V8 尚未完成真实 Dynamics 七候选场景质量验证或机器人执行验收；离线通过不代表真机任务成功。

2026-10-05：Leo3 的 V8 / V5～V7 / 控制 / VLM / SSH 与终端回归共运行 252 项，251 项通过、1 项平台相关跳过；Windows 桥接平台测试 8 项通过。V8 专项覆盖两手六方向、不变关节、同 anchor 七候选、选中精确前缀、ID/hash/场景/硬件拦截、预测失败、取消、preview、七图跨机传输和监督提示。无真机发布。

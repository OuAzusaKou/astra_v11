## 恢复推进策略更新

Astra结合上批目标、最新实测状态与图像评估进展；有依据时继续推进。
短批位移小或缺少执行确认标志本身不触发暂停。批次选择可观察的小目标，最多15点，
不固定每次3点；可据反馈改用后续片段、调整批次或选择更合适的参考。
仍保留具体障碍/抓持/观测/硬件问题及无可用动作依据时的pause，动作校验和预算不变。

## 接管轨迹参考

T 接管请求的 `wam_context` 保留最近3次WAM推理：完整52维动作、对应三视角/状态、
L2、条件版本、时间与新鲜度，以及逐行动作的发布状态和发布前状态。
执行队列清空不删除参考；参考不能绕过重新审阅和原有轨迹校验自动重放。
当前三视角后附各次推理三视角，顺序明确标注。已发布不等于实测执行成功；部分发送失败单独标记。
B复位清空参考；切换L2会将旧条件参考标为不匹配。Astra优先基于这些参考选择/截短/修正动作，
不再以缺少标定文件本身作为暂停理由。推理强度仍为medium，监督超时仍为600秒。

## V2 多 Subtask 顺序运行

在初始 --prompt 后增加 `--next-subtask "place the red pepper in the bowl"`。
该参数可重复，按所给顺序排列。运行 WAM 时按 P，等当前短批结束后选择下一条；
清空旧动作并等待 Enter 确认。L3 不变，不自动判断切换。
Astra 接管期间先用 T 交还 WAM，再按 P；B 复位返回第一条 Subtask。
最后一条再按 P 不切换也不循环。所有 Subtask 共用本轮预算。

# 2026-09-27 更新：B 复位与自动 Astra 接管

最新实现以本节为准。部署目录为 leo3 的 `/home/liberai/Liberai_Deploy_astra`。

## 启动与窗口

```bash
cd /home/liberai/Liberai_Deploy_astra
python3 wam_direct/launch_tmux.py --name task1-v2 --client astra -- run \
  --control-config config/astra/version2_keyboard.json \
  --prompt "Task: Pick up the red pepper and place it in the bowl. Subtask: pick up the red pepper" \
  --session outputs/astra/task1_new_run --owner astra-task1 \
  --original-step-limit 400
```

必须使用新 session 路径、未占用的 tmux 名称。启动后先建立专属 Astra 会话；
握手失败退出，不进入 READY。终端主窗口运行 runner；tmux 的 `Astra-supervisor`
窗口显示会话 ID 与监督日志。用 tmux 的 Ctrl+B 然后 W 选择窗口；单独 B 由 runner 接收。
单独 T/B 必须在 runner 主窗口输入，不能在只读日志窗口输入。

- Enter：READY 后开始；V1 先审阅，V2 直接 WAM。
- T：V2 下在批次边界切到 Astra，自动发送三张图片和状态；再按 T 取消当前监督请求并交还 WAM。
- B：放弃待执行动作与监督请求，回到 startup.reset_state_path 中的任务姿态。
  使用原客户端相同的手部1秒、手臂5秒目标时序，保持现有硬件互锁；不调用会改变硬件模式的 reset 服务。
  用新鲜状态确认到位（位置3厘米、旋转0.15弧度、手关节0.15弧度容差），失败保持暂停。
  成功后再次等待 Enter，恢复初始 L2，总预算保留。结束并退出的 runner 不再接收 B，重新启动新 session 后可在 Enter 前按 B。
- Ctrl+C：退出。紧急停止仍用 ROS 踏板；T/B 不能取消已经下发到 ROS 的当前短轨迹。

`supervisor.json` 保存真实 Codex thread_id，`supervisor.log` 显示状态，`supervisor/`
保存每次调用的事件、错误和 JSON 结果。采用 codex exec 创建、exec resume 恢复同一个会话；
这是专属 CLI 会话，不保证 App 自动切到该对话。运行中不要手动同时 resume 这个会话。
结束后可用记录的 `codex resume <thread_id>` 查看历史。无需人工发消息触发 T 请求。

默认 gpt-6-astra，沿用本机 Codex provider 配置。仅监督子进程使用 Clash 7897 代理；
`--supervisor-proxy ""` 可取消额外指定，`--supervisor-timeout` 默认600秒（10分钟）；监督会话显式使用 medium 推理强度，不继承全局 ultra。
V2 WAM 阶段除启动握手外不调用 Astra。无动作生成依据时返回 pause 并说明，不能凭空猜52维目标。
监督失败会暂停下发；T 可从 Astra 暂停交还健康的 WAM，但不能清除硬件互锁。

以下为原有协议说明（涉及手工监督的段落仅作协议参考，实际由自动桥接提交）：

# Astra + WAM mode 2（leo1 副本）

## 已实现的链路

本目录基于 leo1 实际客户端，不使用先前另一份代码中的 bbox/条件回执协议。
用户先启动已有 ROS 与 mode-2 模型服务/隧道，再由 Astra 启动
`wam_direct/run_astra.sh run`，自动监督桥接读取 session，调用绑定会话审阅并提交决策。
入口已直接装配 ROS 环境、模型连接和控制层，不再需要占位的 factory。

```text
Astra：读三视角、状态、示范、待执行动作
  ↓ condition / approve / replan / correct / takeover / hand_back / complete
本地 session 文件 → 单一 Runner
  ↔ WAM mode 2：infer_action_chunk（Task: L3 Subtask: L2）
  → 既有 326→52 解码（MANO→q20）
  → 既有 publish_arm_chunk / publish_hand_chunk
  → 已运行的 ROS 规划器与踏板急停
```

自动建立 Codex/Astra 会话；不自动启动 ROS、CAN 服务、模型或 SSH 隧道。不会自动复位、
移动至启动姿态、解除硬件急停，也不使用原 `base.main()` 的启动运动流程。
人工提前将机器人准备到任务起始姿态。原 `run_wam.sh` 的复位行为仍保留，
它不是协作入口；仅增加与 Astra 共用的进程锁。

## 在机器人上启动

先将本副本部署到机器人上的独立目录；本次本地开发没有进行部署。
在副本根目录，用机器人原有 Python/ROS 环境，额外确保 Pillow 可用。

```bash
bash wam_direct/run_astra.sh run \
  --mode 2 \
  --control-config config/astra/version1_review.json \
  --prompt "Task: Pick up the red pepper and place it in the bowl. Subtask: pick up the red pepper" \
  --chunk_size 32 \
  --session outputs/astra/task1_run001 \
  --owner astra-task1 \
  --original-step-limit 400
```

`--prompt` 使用原版 mode 2 的 `Task: <L3> Subtask: <L2>` 格式，两项由用户指定。
Runner 获取初始观测后等待 Astra 审阅；首次 condition 的 L2 必须与用户传入值一致，
不能自行替换初始 L2。后续换子目标使用显式 replan/hand_back，并记录依据。

第一次推理之前不会下发动作；推理返回后也必须单独审阅才放行。
`--original-step-limit` 必须按本次任务选择；这里的 400 只是示例，总预算默认乘3。
`--session` 必须是新目录，拒绝覆盖或自动续跑旧会话。

当前两个控制配置都直连 5090：`10.128.40.203:29572`。29572 是约定的服务端口，
需要在 5090 显式启动对应 server 并等待编译/预热完成；本次没有启动常驻服务。
不再默认使用 leo1 的 SSH 隧道。`--host`/`--port` 可覆盖连接地址；模型身份要求仍由
control config 的 expected_checkpoint 指定。当前匹配迁移后的 mode2-7114 路径。
服务端 checkpoint/adapter/protocol/action_dim 在每次推理响应后校验。
模型配置实际 action_horizon=30，`--chunk_size 32` 只设置截取上限，不补成32步。

## Astra 如何读、写

session 内包含：

- `manifest.json`：配置、源码哈希、来源快照指纹、预算语义。
- `request.json`：当前 request_id/session_id/owner、控制模式、当前 L3/L2、
  条件版本、最多15步待执行的52维动作、停滞提示，以及图像和状态引用。
- `obs_*`：主相机/左右腕部 PNG（RGB），可直接查看；state.json 可直接读取。
- `events.jsonl`：推理、批准、干预前后轨迹、发布时序、异常、恢复与完成记录。
- `status.json`：当前等待/推理/执行/暂停/结束状态。
- `summary.json`：成功来源、下发步数、预算计数、接管次数及恢复尝试次数。

监督者必须实际查看示范和当前图片；填入 review_evidence 不等于完成视觉审阅。
目前不自动检索示范，也不提供自动规划器或无人审阅的批准循环。

初次 decision.json 的格式如下；ID 和示范内容必须替换为真实值：

```json
{
  "session_id": "复制本次request里的session_id",
  "request_id": "复制当前request里的request_id",
  "owner": "astra-task1",
  "operation": "condition",
  "l2_instruction": "pick up the red pepper",
  "reason": "开始第一个子目标",
  "evidence": "实际查看当前三视角后填写观察依据",
  "demo_context": {
    "video_path": "真实示范路径",
    "viewed_frames": [0, 15, 30],
    "review_evidence": "实际查看示范后填写",
    "scene_differences": "当前布局相对示范的差异"
  }
}
```

```bash
bash wam_direct/run_astra.sh respond \
  --session outputs/astra/task1_run001 --file decision.json
```

每次响应显式绑定请求ID和owner，原子提交，不覆盖已有响应；旧响应和串会话响应拒绝。
正常决策都要求 reason/evidence。支持的操作：

| 操作 | 额外字段与行为 |
| --- | --- |
| condition | 用户指定的初次 l2_instruction + demo_context；调用 WAM 后等待动作审阅 |
| approve | 执行当前已提出动作，默认最多15步；无剩余动作时只重新推理，再次等待审阅 |
| replan | 新 l2_instruction + demo_context，或明确 conditioning_version；丢弃旧动作并重新推理 |
| correct | actions（最多15行52维）；只替换当前队列 |
| takeover | actions（最多30行52维）、new_attempt 布尔；新尝试还需 change_reason |
| hand_back | recovery_evidence + 新子目标或 conditioning_version；清旧队列并重新推理 |
| observe | 不动作，重新获取当前图像；不清停滞计数 |
| pause | 停止后续下发，保留现场，等待主操作员处理 |
| complete | 记录为 astra_visual 完成，停止下发；不是硬件权威成功信号 |

提交少于默认行数的 correct/takeover，或者用 execute_steps 缩短当前执行前缀，
必须填写 short_batch_reason。30步接管提交也在15步后再次审阅，剩余15步不会自动执行。
连续接管用 new_attempt=false 延续一次恢复；改变策略重试用 true。次数不设上限。
progress_evidence 表示实际观察到进展；重命名/重新推理/交还不会自动清零停滞。
128步无进展阻止直接批准，要求观察到进展或诊断/调整，不自动判失败或复位。

动作统一使用机器人原生绝对目标：

```python
actions.shape == (T, 52)
left_ee_pose6 = actions[:, 0:6]
right_ee_pose6 = actions[:, 6:12]
left_hand_q20 = actions[:, 12:32]
right_hand_q20 = actions[:, 32:52]
```

手臂是 chest_link 下 xyz（米）和旋转向量（弧度）；双手 q20 为关节目标。
Astra 不输出326维，不得把像素直接当机械位移。发布前检查有限性、形状、
本副本 URDF 手关节限位，以及相邻末端目标的跳变。默认平移跳变0.10米、旋转
跳变0.75弧度，可通过 --max-step-m / --max-step-rad 配置并记录；这只是额外的
目标跳变检查，不是碰撞/可达性证明，不替代 ROS 的已有运动约束。

## 人工完成与暂停恢复

人工完成可在等待或暂停时提交；提供当前 status.json 中的 session_id。
处于暂停时还必须匹配 pause_id：

```bash
bash wam_direct/run_astra.sh complete --session outputs/astra/task1_run001 \
  --session-id ACTUAL_SESSION_ID --reason "人工确认辣椒已放入碗中"

bash wam_direct/run_astra.sh resume --session outputs/astra/task1_run001 \
  --session-id ACTUAL_SESSION_ID --pause-id ACTUAL_PAUSE_ID \
  --reason "已处理故障并人工恢复硬件状态"
```

complete 来源记为 human；Astra 自己判断完成应通过普通 complete 决策，不能冒充人工。
这些身份字段用于防串请求，不是同一系统用户之间的安全权限隔离。
resume 不调用 ROS reset，不解除急停，不直接执行动作；恢复后重新观察与决策。

## 时序、急停和记录的准确含义

Astra 是同步监督：审阅期间不下发新动作。新入口每批默认最多15个轨迹点，
用相同 ROS 起点发布双臂/双手，保持30Hz点间隔，第一点使用正时间偏移；
等该批计划时段及0.3秒余量过去后才进入下一次审阅。因为审阅时间不固定，
不能复用旧观测时刻计算出的截止时间；本入口不采用原 sync-no-sleep 或跨批融合。

现有 ROS publish 没有逐点执行完成回执。步骤预算按待下发的轨迹点保守扣除，
避免部分发布失败后重新获得预算；另外统计双臂双手发布调用均完成的点数。
两者都不宣称硬件实际执行确认，人工踏板中断也不伪造已完成动作数。
因此不再移植 RoboDojo 的“1秒有效仿真推进时间内必须触发官方成功”规则，
真机以人工或 Astra 视觉完成判定收尾。

硬件 /leo/<hardware>/mode 需要五个组件都为 RUNNING 且消息新鲜。
进入非RUNNING会锁存事件，即使在模型推理时急停后又恢复，也必须人工 resume。
发生异常不再发布后续动作；**已发布的短轨迹可能继续完成，软件暂停不是急停**。
实时急停仍由现有 ROS/踏板负责。没有发送零目标或空轨迹来冒充安全保持。
接管/交还只在批次边界进行，旧客户端动作队列被丢弃。

本副本两个启动入口共享 /tmp/leo-astra-UID/robot.lock，不杀旧进程抢锁；
还检查 ROS 三个动作话题是否只有一个发布者。原远端目录未改，若另起旧客户端，
它不参与这把锁，不要与新 Runner 并行运行。

Ctrl+C/SIGTERM 停止后续下发并退出，无自动复位。Runner 崩溃后没有自动动作重放或
自动恢复；保留 session 供排查，新运行需重新确认当前场景。Astra 会话退出不代表
tmux里的 Runner 退出，它会等待当前请求，超过默认900秒进入暂停。

## 离线验证与上线边界

```bash
python -m unittest discover -s tests -p 'test_astra_mode2.py' -v
```

这些测试不连接机器人和模型服务。实际部署还需核验安装依赖、硬件 mode 消息、
相机/TF时间戳与运动时序，以及一次真实短批接管/交还。本次开发不启动上述服务。

## 终端与原版键盘操作

两个启动脚本都在当前终端运行，不创建 tmux 或新终端。原版 `run_wam.sh --mode 2`
在当前交互终端按 P，等待 chunk 结束，再按 Enter 切换预设的下一条 L2。
若在 tmux 内启动，就在对应窗格输入；无 TTY 时此键盘功能不可用。
新 `run_astra.sh` 第一版使用 session 审阅；第二版使用 T 切换 WAM/ASTRA 控制权，
不需要 Enter。P 保留给子任务切换，当前新版 runner 尚未接入原版 P/Enter 逻辑。
`--session` 指运行数据目录，不是 tmux 会话；如需 tmux，应单独创建。

## 在 leo1 本机弹出 tmux 终端

Astra 与客户端都运行在 leo1。在 leo1 的图形桌面会话中，让 Astra 调用
`wam_direct/launch_tmux.py`：创建本机 tmux，再打开本机终端窗口并接入。
不使用 Mac、SSH 或远程部署路径参数。启动器以自身所在副本为客户端目录。

在已部署副本根目录执行（以下仅为启动示例，本次没有执行）：

```bash
python3 wam_direct/launch_tmux.py --name wam-task1 --client astra -- run --mode 2 --control-config config/astra/version1_review.json --prompt "Task: Pick up the red pepper and place it in the bowl. Subtask: pick up the red pepper" --chunk_size 32 --session outputs/astra/task1_run001 --owner astra-task1 --original-step-limit 400
```

需要原版 P/Enter 时使用原版入口，以 `--` 分隔启动器和客户端参数：

```bash
python3 wam_direct/launch_tmux.py --name wam-manual-task1 --client wam -- --mode 2 --port 29672 --prompt "Task: Pick up the red pepper and place it in the bowl. Subtask: pick up the red pepper" --chunk_size 32 --record --video-name task1_mode2_seed4
```

新版 Astra runner 第二版支持 T 切换控制权，未接入原版 P/Enter 子任务切换和
视频录制参数。此启动器只改变启动与查看方式。`--session` 是运行数据目录，`--name` 才是 tmux 名称。

需要安装 tmux，以及 gnome-terminal、konsole、xfce4-terminal、xterm 中至少一个。
启动进程需继承 leo1 图形桌面的 DISPLAY/WAYLAND_DISPLAY 与访问权限；缺少显示环境
或终端程序时在启动客户端前报错。仅有 SSH/无桌面环境无法弹出本机窗口。

同名会话已存在时拒绝启动，不替换或重启。客户端退出后保留输出，按 Enter 关闭窗格。
如果弹窗失败，tmux 中客户端可能已经运行，使用 `tmux attach-session -t =wam-task1`
接入检查，不要重复启动。关闭终端窗口或 Ctrl+B 再 D 都不代表停止机器人。

在 `--` 前加 `--dry-run` 可只打印命令，不启动任何进程。启动器不启动 ROS、CAN
服务或 server 隧道；它们需提前准备。本次只修改本地副本，尚未部署到 leo1。

## 两个 Astra/WAM 控制版本

通过 `--control-config` 选择，独立于 ROS 的 `--config`。两个版本都是 WAM mode 2，
初始 Task/Subtask 都由用户指定，保留已有326→52转换、动作校验和ROS踏板急停。

| 配置 | 行为 |
|---|---|
| config/astra/version1_review.json | 首次 Astra 审阅用户 L2，此后每批最多15步均等待 Astra 审阅；可纠正、接管、交还 |
| config/astra/version2_keyboard.json | 初始用户 L2 直接用于模型推理；WAM 连续执行，不生成逐批 Astra 审阅请求；T 切换到 Astra，再按 T 交还 WAM |

第二版从 leo1 图形桌面启动并弹出 tmux：

```bash
python3 wam_direct/launch_tmux.py --name wam-keyboard-task1 --client astra -- run --control-config config/astra/version2_keyboard.json --mode 2 --prompt "Task: Pick up the red pepper and place it in the bowl. Subtask: pick up the red pepper" --chunk_size 32 --session outputs/astra/task1_keyboard001 --owner astra-task1 --original-step-limit 400
```

第一版将配置换为 version1_review.json。默认也是第一版。server应在5090上监听
29572，使用迁移后的mode2-7114权重。server当前仍需另行启动；客户端配置不会启动它。

第二版按键行为：

1. 按 T 后打印 SWITCH REQUESTED。若推理正在进行，等它返回但不执行结果；若已
   下发动作，等当前短批次执行等待结束后切换。批次最多15个轨迹点，30Hz下轨迹长度约
   0.5秒，另有发布提前量、等待余量及观测处理，因此不能承诺0.5秒内完成切换。
2. WAM→ASTRA：清空WAM队列，生成 ASTRA_TAKEOVER 请求和最新观测。Astra通过
   session提交 takeover 的52维绝对目标，接管期间不会调用WAM。没有Astra响应时保持等待。
   Astra仍需逐批提交/审阅自己的动作；按键本身不会生成动作，也不会创建新的Astra会话。
3. ASTRA→WAM：清空Astra剩余队列并使旧响应失效，保留当前L2，以最新观测重新请求
   WAM后自动执行。第二版不允许Astra通过hand_back自行交还；控制权由用户T键决定。
4. 初始L2不会按预设计划自动切换。第二版当前切换键仅改变控制权，不改变L2；如需更换
   子任务，应另行明确任务指令流程。第一版仍可由Astra通过replan改变L2。
5. 硬件/模型异常会暂停；T不能清除故障或代替主操作员resume。已下发批次不能通过此键
   可靠撤销，实时急停继续使用ROS踏板。无交互终端时第二版拒绝启动；终端输入断开则退出。

阶段切换同时写入终端、events.jsonl和status.json，包含时间、阶段编号、来源/目标控制方
及步骤数，例如 `[PHASE 002] WAM -> ASTRA`。按键请求尚未处理时的重复按键会合并，
请按一下并等待阶段日志后再切换。模型执行阶段不等待Astra审阅；步数预算与动作校验仍生效。

当前仅完成本地代码及离线测试，未将新版client部署到leo1，未进行真机切换验证。

V1/V2统一启动确认：runner准备完成后进入WAITING_FOR_START，终端显示READY。
用户按Enter后，V1进入Astra初始审阅流程，批准前不执行动作；V2开始WAM推理和执行。
两版都会丢弃初始化期间的提前按键；开始前T无效，开始后仅V2支持T切换控制权。
Enter不会重复启动。Ctrl+C可在等待期间退出，不发布动作。两版均需要交互终端。

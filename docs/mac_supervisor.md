# 临时把 Astra 放到 Mac

Windows 原生增量支持见 [Windows 监督说明](windows_supervisor.md)。保留 `mac` 作为 SSH 后端协议名称；Windows 不需要改 Leo3 的后端参数。本文原有 Linux/macOS 调用方式继续有效。

默认行为不变。leo3 启动时增加 `--supervisor-backend mac` 才切到 Mac；删除这个参数即恢复 leo3 本地 Astra。
ROS、WAM server、按键与执行校验仍在原位置，medium / 600秒不变。

两端必须指定同一个新 session。先开 Mac 桥接（可以等待目录出现），再用 leo3 图形终端启动 runner。

Mac，在本副本根目录：

```bash
bash wam_direct/run_mac_supervisor.sh \
  --session /home/liberai/Liberai_Deploy_astra/outputs/astra/pepper_mac_001
```

leo3：

```bash
cd /home/liberai/Liberai_Deploy_astra
python3 wam_direct/launch_tmux.py --name pepper_mac_001 --client astra -- run \
  --supervisor-backend mac \
  --control-config config/astra/version2_keyboard.json \
  --prompt "Task: Pick up the red pepper and place it in the bowl. Subtask: pick up the red pepper" \
  --next-subtask "place the red pepper in the bowl" \
  --session outputs/astra/pepper_mac_001 --owner pepper_mac_001 \
  --original-step-limit 400
```

每次换一个未使用的名称，两端一起改。Mac 输出 MAC READY、leo3 输出 READY 后，按原流程 Enter/P/T/B。
Mac启动握手会真实调用Astra，但无机器人动作；runner在Mac就绪前不进入开始等待。
Mac退出、休眠、网络断开时，Astra等待阶段会暂停。WAM阶段不调用监督模型。
远端心跳允许30秒；每次SSH请求最长20秒，网络失败时Mac取消模型调用并退出，需人工重新连接。

Mac上的 `outputs/mac_supervisor/<session_id>_<connection_id>/` 保存 thread_id、图片与调用日志。
leo3的 `supervisor.json` 显示 backend=mac 和 Mac 会话ID；App不会自动跳转到该会话。
所有模型调用使用Mac本机Codex账号/配置，不复制leo3密钥。Mac可用 `--proxy ""` 取消额外代理设置。
不要并行运行两个桥接进程，不要手动resume正在被桥接管理的会话。
迟到或身份不符的响应在SSH提交层和runner接收层分别拒绝。图片传输校验SHA256。
这不是热切换：已运行的旧runner继续用原后端，须在下一次新session启动时选择mac。

V2接管：T后Astra独立生成takeover；WAM历史仅用于学习控制与反馈关系，不是待执行轨迹。observe会清空旧片段并自动获取新观测继续规划，日志显示ASTRA保留控制权；具体故障/风险仍可pause。T交还模型，B复位，P/Enter切换子任务的流程不变。更新后两端启动新的session，旧进程不热更新。

客户端手部URDF目标拒绝检查默认关闭（--no-client-hand-limits），可用--client-hand-limits恢复。只影响客户端整批拒绝，不裁剪目标，不修改ROS驱动限位/踏板。ROS可能将超界手部目标限制到边界，实测姿态不保证保持不变。新runner生效，旧进程不热切换。

审阅期间状态变化检查默认只记录变化最大的维度、旧值、新值和差值，不再因超过0.05暂停；--review-state-guard可恢复拒绝行为。执行轨迹仍相对最新实测状态检查手臂跳变，硬件/观测有效性检查保留。修改不会让旧观测自动变新，执行前场景可能已经变化。

Execution defaults to 20 fps for both WAM and Astra; --fps overrides it. Each supervisor request includes the actual fps. Astra increases total travel when clear space and prior measured progress support it, and shortens travel near contact. No fixed displacement is guaranteed. Start a new runner and Mac bridge session for a consistent updated run; --fps 20 can also be explicit.

Supervisor calls explicitly request gpt-6-astra, medium effort, features.fast_mode=true and service_tier="fast". This overrides the default tier for these calls without modifying global Codex settings. supervisor.json records requested_service_tier, not a server-confirmed tier. CALL_ERROR lines surface CLI transport retries/fallbacks during a turn; Fast does not fix network failures. See https://learn.chatgpt.com/docs/agent-configuration/speed for availability and credit usage.

Endpoint-first planning: Astra chooses a grounded batch endpoint before producing up to 30 coordinated waypoints that cover the path to it. It can combine translation axes and orientation instead of requesting separate lift/x/y turns. Exceptions require a concrete obstacle, contact, uncertainty or timing reason. Endpoint, expected displacement and duration are reported in reason/evidence. This is a prompt update only: custom trajectory duration, 200-point resampling and long-queue cancellation are not implemented; current scheduled timing still applies.

Astra takeover now submits and executes up to 30 waypoints per review (1.5s at 20fps, plus lead/settle time). WAM review batches remain 15. T/B still wait for an already-published batch to drain; no long-queue cancellation or custom-duration resampling was added.

Mac transport workaround (2026-09-28): the Darwin supervisor uses a per-call astra_https provider with requires_openai_auth=true, wire_api=responses and supports_websockets=false. It reuses the existing OpenAI login; global config and leo3 local provider are not changed. CLI 0.154 rejects overrides of reserved model_providers.openai, so a distinct per-call provider ID is required. Startup prints OpenAI transport=HTTPS. Restart the Mac bridge and create a fresh runner session to apply consistently. This bypasses WebSocket reconnects, but does not guarantee HTTPS/network reliability. Reference: https://learn.chatgpt.com/docs/config-file/config-reference

V2 pause recovery: press R in the leo3 main control terminal when state=PAUSED_FOR_OPERATOR and control_mode=ASTRA. Runner acquires fresh observations and issues ASTRA_RETRY to the same bound supervisor; no action is published by R, no WAM inference is triggered, and budget/L2 are preserved. Previous pause reason is included in retry_context. Hardware checks remain latched. R is ignored outside paused V2 Astra; it cannot restart an EXITED runner or a disconnected Mac bridge. T hands back to WAM; B resets. The planning prompt first assesses supported intermediate/view-improving poses when final placement geometry is uncertain, while preserving explicit pause for unresolved concrete hazards.

Read-only supervision tools (2026-09-28): Codex uses the astra_observe permission profile extending :read-only with network.enabled=true, replacing the old -s read-only network-off preset. Local reads and outbound SSH work; local command file writes remain sandbox-blocked. Remote SSH commands use the existing account, so remote read-only behavior is an instruction boundary, not a forced-command/read-only SSH account. ROS state/TF/parameter queries are allowed; control publishing, mutating services, file edits/deletes and process changes remain disallowed. Model decisions are still delivered only through the runner. No global Codex config is modified. Exact target coordinates are not mandatory for every supported intermediate motion. Permission-profile support requires the deployed Codex CLI; legacy sandbox_mode in a user config can override profile selection and should be diagnosed rather than bypassed.

## V3 可选自主交还

Mac桥接启动命令不变。leo3启动参数改为 `--control-config config/astra/version3_auto_handback.json`。Enter启动WAM，T人工接管；Astra确认恢复后可自主hand_back，日志显示 `ASTRA -> WAM | Astra automatic hand-back` 及原因。T仍可人工切回。保留当前L2、预算，丢弃旧动作后重新推理；P/Enter、B、R不变。V1/V2行为不变。更新后用新session启动，不热重启真机。

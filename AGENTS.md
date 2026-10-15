# Astra 真机客户端会话约定

启动前读取 `astra/SUPERVISOR.md` 和 `docs/astra_mode2.md`。

- 用户授权启动后，统一使用 `python3 wam_direct/launch_tmux.py --name <唯一名称> --client astra -- run ...`。不要在工具隐藏 PTY 直接启动，不代按 Enter/T/B。
- 启动说明必须包含版本、tmux 名称、runner session 绝对路径。每次使用新 session 目录。
- runner 自动创建专属 Codex/Astra 监督会话，启动握手成功后写入 `supervisor.json` 的真实 thread_id；`--owner` 仍是 runner 标签。不要声称发起启动的 App 对话就是后台监督会话。
- 启动器开终端窗口并确认 tmux 已连接。主窗口操作 Enter/T/B，`Astra-supervisor` 窗口显示监督状态；可以用 tmux 窗口列表选择它。
- V2 在 WAM 阶段不调用监督模型审阅动作。T 在当前短批结束后自动唤醒绑定会话；再次 T 取消待处理响应并用当前观测恢复 WAM。不需要人工向监督对话发送消息。
- B 是人工回到配置任务初始姿态的动作，不是急停；保持 ROS 踏板急停。复位后必须再次 Enter，预算不清零。B 不解除硬件故障。
- 不并行向专属监督会话发消息或交互 resume，以免抢占请求。会话 ID、结果与错误在 supervisor.json、supervisor.log 和 supervisor/ 目录中。
- supervisor 默认通过 leo3 已有 http://127.0.0.1:7897 代理访问配置的提供商；可用 --supervisor-proxy 指定，空字符串表示不额外指定代理。不要输出密钥。
- V2 T后独立生成takeover，WAM历史用于理解控制与反馈，不是待执行计划。规划未就绪用observe获取新观测并继续，具体故障或无法解决的动作依据缺口才pause；不捏造52维动作。未通过真机验证前不可宣称任务成功。
- 未经用户授权不启动 ROS/CAN 或执行真机测试。离线测试/纯通信测试不发布动作。

- V2 支持重复 --next-subtask 参数；P 选择下一条、Enter 确认。Astra 接管中先 T 交还再 P。B 复位回到第一条，预算不重置。

- 临时 Mac 监督模式见 docs/mac_supervisor.md。只有 --supervisor-backend mac 才使用SSH邮箱；默认local不变。禁止为迁移而自动重启正在运行的真机runner。

- V2 Astra暂停后，主终端R重新观察并重试同一监督会话，不交还WAM、不直接发动作、不清预算；不能解除硬件故障。T仍为交还WAM，B仍为复位。退出的runner需新session启动，R不能复活已退出进程。

- 监督会话允许本地/SSH/ROS只读查询，使用astra_observe只读文件系统＋网络权限；远端只读是指令约束，不能声称SSH账号已强制只读。最终仍只返回decision，由runner下发；禁止直接发布控制、修改删除文件、重启进程。相对方向与已核实控制映射可支持中间动作，不把精确目标坐标作为所有运动的先决条件。

- V3使用config/astra/version3_auto_handback.json；复用V2键盘/子任务流程，但允许Astra依据最新恢复证据hand_back。仅T发起接管，T仍可人工交还；自主交还保留L2和预算，清队列后重新推理。启动说明必须区分V2与V3。

- V4 使用 config/astra/version4_vlm_subtasks.json：VLM 持续高进度仅在 WAM 批次边界自动 next_subtask/start；首轮、手动 P、B 后仍需人工 Enter。保留 V3 T 接管/自主交还，最后 L2 不自动完成；加 --vlm-auto-takeover 可显式启用停滞自动接管：10秒L2保护、12秒无新高、最近8次有效预测覆盖6秒，replay仅记录、不作为门槛、当前进度低于90%；交还后15秒冷却。Mac start_mac_vlm.py 默认 V4，可显式 --control-version 3 回到旁路。

- V5 使用 config/astra/version5_dynamics.json，继承 V4，仅 Astra 候选先做 Dynamics 想象。默认 --dynamics-mode preview：WAM 仍正常运行，Astra 候选只展示、不下发；review 才允许请求绑定的 approve 执行已想象前缀。不要把“预览”说成整个机器人不动。运行与原生数据适配见 docs/dynamics_v5.md。模型 29640 保持原 canonical 协议，新 CPU 适配入口 29642；不得直接用 52D/58D 拼零填充 canonical action。

- V6 使用 config/astra/version6_pointcloud.json：基于 V4，Astra 候选通过独立 PointCloud2 想象再 review，无 Dynamics 服务。见 docs/pointcloud_v6.md。场景冻结，不是物体动力学，不可把几何图当作执行证据；不得向真实 TF/joint_states 发布候选。

- V7 使用 config/astra/version7_prefix.json，见 docs/pointcloud_v7.md。最多120点整段预演，审批有界前缀，执行后丢弃尾段并重新观察。不得宣称点云提供完整碰撞/IK证明；仍需真实恢复证据才能hand_back。复用V6 RViz与录制，不自动启动硬件测试。

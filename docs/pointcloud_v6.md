# V6：场景点云中的手部几何想象

V6 基于 V4：WAM 和 VLM 的子任务/接管流程保持原样；只有 Astra 的候选动作先通过 URDF 手部点云想象。无需 PRO6000、Dynamics 模型或 29640/29642。默认 review，允许 Astra 审阅后执行。

## 使用

在 Leo3 图形桌面终端先启动点云显示（已有 overlay 时先正常关闭旧 overlay 窗口，避免两个程序争用相机）：

```bash
cd /home/liberai/Liberai_Deploy_astra
bash wam_direct/view_pointcloud_v6.sh
```

该脚本使用现有 camera_pose 标定、深度相机和真实手部点云发布器，选择包含 V6 想象话题的 RViz 配置。不会启动 ROS/CAN 控制器。相机标定要求头部保持原脚本配置角度；请遵循原 view_overlay.sh 的提示。

在监督笔记本的仓库目录启动（目前部署在 Mac）：

```bash
python3 wam_direct/start_mac_vlm.py \
  --control-version 6 --vlm-auto-takeover --pointcloud-mode review \
  --prompt "Task: Pick up the green apple and place it in the bowl, then pick up the red pepper and place it on the plate. Subtask: pick up the green apple" \
  --next-subtask "place the green apple in the bowl" \
  --next-subtask "pick up the red pepper" \
  --next-subtask "place the red pepper on the plate"
```

复制生成的同 RUN_ID 命令到 Leo3，初始化后主窗口 Enter。T 人工切换，R 对暂停的 Astra 重新观察规划，B 是实体复位。旧 client 必须正常退出才能开启新会话。`--pointcloud-mode preview` 可只看几何想象；WAM 仍会运动。

## 链路和颜色

Astra 候选 → 新鲜 RGB/state anchor → 读取深度点云和 URDF → 采样 visual mesh 表面 → 根据候选末端姿态及 40 个手指关节做 FK → RViz 动画 → Astra 审阅 → exact candidate approve → 执行 → 真实观察验证/hand_back。

动作是 chest_link 下左右末端 xyz+rotvec、左右各20关节（与当前 runner 一致）。URDF 的固定连接、joint origin/axis、mesh scale/visual origin 均保留。使用 camera_pose 的现有 sample_hand_visuals，以较低密度120000稳定采样；开始时用新鲜 TF 检查 FK，一致性超过15mm则拒绝。client 持续缓存场景点云；取得机器人 anchor 时即冻结相邻点云和该 anchor 时刻的历史 TF，随后才保存图像和启动渲染进程。不会用渲染启动后的新点云与旧 anchor 比较。深度时间戳仍不是硬件同步测量。

RViz：原真实手部仍是左青右紫；橙色 `/astra/imagination/hand_points` 是候选手部，按实际 fps 播放一次，审阅时保持终点。取消、替换、批准后清理。只发布 PointCloud2，绝不发布 joint_states、TF 或控制话题。每段最多约1秒（20fps、lead 0.1时18点）；整个被批准前缀与显示动作一致。

Astra 得到与 RViz 相同几何数据的俯视XY、正视YZ、侧视XZ过程图（初段、中段、终点），不是 RViz 截屏。图中青色为 anchor 双手，橙色为想象双手，场景颜色调暗。三视图保留坐标范围，并结合真实相机图判断。批准必须匹配候选ID、动作哈希和整段长度；修改动作重新想象。

## 数据与限制

`outputs/astra/<RUN_ID>/imagination/<candidate_id>/` 自动保存 request.json、frozen_scene.npz（冻结点云、TF及采集时间）、robot.urdf、geometry.npz、三个投影视图PNG、metadata.json 和 worker.log。geometry.npz 包括冻结场景、anchor 手部、每一步候选手部点与动作。latest.json、index.jsonl、events.jsonl 记录索引和批准执行事件。VLM 视频仍单独保存。V6 helper 自动加入 `--rviz-record`，实际 RViz 窗口按10 FPS、每60秒一段录制到 `outputs/rviz/<RUN_ID>/window_video/`；`segments.jsonl` 记录片段，外层 `frames.jsonl` 记录绝对时间、运行阶段及L2。`RViz-recording` tmux 窗口和 `status.json` 显示录制状态。client 正常退出后录制自动收尾；未找到 V6 窗口时录制明确的等待提示画面，不冒充有效 RViz 画面。请保持 RViz 窗口可见、不要最小化或遮挡；窗口缩放时保持视频固定尺寸并留黑边。录屏使用 X11/XWayland，V6 显示脚本自动选择 xcb。旧版已运行 client 不会自动新增录屏，需要下次用新生成命令启动。

这是运动几何预演，不能预测接触力、物体移动、滑落或抓取结果。场景深度中可能包含真实机器人点，未自动去除；重叠可能是自身点，不能直接当作碰撞。无自动 collision-free 认证、无逆运动学可达性认证。Astra 必须以执行后的真实观测判断成功或交还。

场景必须为 chest_link、至少100个有限点；冻结时年龄<=2秒、与anchor差<=2秒。深度发布器时间戳是主机接收时间，不是传感器同步。TF/FK、图像完整性失败或缺少相机点云都会暂停，不会绕过审阅下发。审阅后保留 V5 的 anchor 漂移检查和0.15rad手指容差。

## 已完成验证

230 项相关回归测试通过；实际 URDF 的 52 个 mesh link、14723 个采样点通过平移/手指运动检查，FK 与 Pinocchio 独立结果一致。隔离 ROS_DOMAIN_ID=201 的合成场景通过真实 worker 收发、三视图保存、动画与取消测试。尚未用本版本执行真机任务或确认真实深度场景下的 Astra 判断质量。

RViz 启动脚本现自动打开实时 XY/YZ/XZ 三投影 Image 面板，显示场景、青色实测手和橙色预演手；与 Astra 审阅共享正交投影计算。详见 docs/pointcloud_v7.md 的“RViz 实时三投影”。

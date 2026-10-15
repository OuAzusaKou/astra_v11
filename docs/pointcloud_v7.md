# V7：整段点云预演、有限前缀执行

基于 V4/V6。每个 L2 仍由 WAM 开始，VLM 或 T 触发 Astra 接管。Astra 用当前真实图像和状态规划到下一个明确目标，例如已抓稳后的碗上方；不能假设接管时已经抓稳。没有 Dynamics 服务依赖。

## 启动

Leo3 桌面先运行 `bash wam_direct/view_pointcloud_v6.sh`（V7 复用这个 RViz 配置和录制窗口）。Mac 仓库根目录运行：

```bash
python3 wam_direct/start_mac_vlm.py \
  --control-version 7 --vlm-auto-takeover --pointcloud-mode review \
  --prompt "Task: Pick up the red pepper and place it in the bowl, then pick up the green apple and place it on the plate. Subtask: pick up the red pepper" \
  --next-subtask "place the red pepper in the bowl" \
  --next-subtask "pick up the green apple" \
  --next-subtask "place the green apple on the plate"
```

helper 创建唯一 RUN_ID，打印同名 Leo3 tmux 启动命令及 `/home/liberai/Liberai_Deploy_astra/outputs/astra/<RUN_ID>`。复制到 Leo3 桌面，准备好后人工 Enter。Mac 保持运行。自动录制 VLM 和 RViz；V4 默认入口不变。preview 只在 Astra 预演后暂停，review 才允许完整循环。T 交还，R 重试 Astra 暂停，B 复位；不改变现有硬件急停行为。

## 规划与审批契约

- takeover 的 actions 最多 120 行原生 52D，按运行 fps 连续采样；20 fps 下最多 6 秒加发布提前量。不是更快执行同样点数。
- `execute_steps` 在 takeover 中等于完整提交行数，只触发预演。少于120行也允许，保留 `short_batch_reason`。
- 必须携带 `plan_stages`，每项含 kind、end_step、evidence。kind 为 transport/approach/contact/release；end_step 为从1开始计数的阶段末尾点数（Python切片上界），严格增加并覆盖全部行。下降、接触、松手分段。
- 全计划通过现有有限值、手关节限位与目标跳变校验，RViz 演示全部动作；保存完整轨迹及前三视角全计划图、后三视角前缀图。
- 首段搬运前缀上限1.5秒/每臂累计路径0.18米；其余阶段0.5秒/0.04米。时间含lead；20fps、lead=0.1时分别最多28点和8点。同时受阶段末尾和剩余预算限制。
- 前缀最大平移速度0.15米/秒，旋转向量差速度0.8弧度/秒；transport 手指相对anchor不超过0.05弧度，不能夹带释放。未找到至少1个合规点会暂停并要求重新观察/规划。
- REVIEW_IMAGINATION 必须绑定原 candidate_id/hash，批准1至 max_execute_steps 个原始连续点，不允许替换动作。审批时再次核对新状态、时钟、控制权、完整队列与预算。
- 执行后清掉剩余计划，VERIFY_RECOVERY 重新观察。Astra 重新规划或依据真实恢复证据 hand_back 给 WAM，不能自动重放旧尾段。VLM L2 切换规则继续沿用 V4。

`imagination/<candidate>/` 保存 request/result/metadata、geometry.npz、完整和 prefix_ 三视角图。events.jsonl 新增 v7_prefix_approved 与 v7_tail_discarded，记录计划点数、批准点数和限制。

## 能力边界

这是有边界、经 Astra 视觉审阅的执行前缀，不是自动“安全前缀”证明。当前没有完整双臂 IK 可达性求解、全身/自碰撞检测、被抓物体扫掠体积检测。冻结的场景不会随被抓物体运动，不能据此证明避障或任务完成。保留现有控制器约束和操作员踏板。

已进行隔离离线测试；新增长轨迹尚需操作员真机验证。

## RViz 实时三投影

`view_pointcloud_v6.sh` 与 Leo3 原 `~/camera_pose/rviz_overlay/view_overlay.sh` 现在一起启动显示节点，并在 RViz 添加 Projection top_XY / front_YZ / side_XZ 三个 Image 面板。XY横X纵Y，YZ横Y纵Z，XZ横X纵Z，均为chest_link正交投影；横轴正向向右，纵轴正向向上。窗口可拖动停靠排列。

节点只订阅场景、实际手部和想象点云，以5Hz发布 `/astra/projections/{top_XY,front_YZ,side_XZ}` RGB图像。共用Astra审阅的投影像素计算；这里显示当前实时帧，取景随手部范围调整，不是审阅图的首/中/末三帧拼图。青色是当前实际手，橙色是当前预演手。2秒无数据则清除对应点云，并标出场景/实测手断流。仅发布Image，不发布机器人控制或TF。关闭启动脚本时同步清理显示节点。

Leo3 实时投影另支持环境变量 `RVIZ_PROJECTION_CROP_FACTOR`：默认1（不裁剪），大于1时居中裁剪并放大到原面板尺寸。此项仅影响实时显示，不改变Astra审阅图、轨迹或执行阈值；可能裁掉边缘对象。

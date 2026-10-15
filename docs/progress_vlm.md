## 可选自动 Astra 接管（2026-09-30）

Mac 使用 `python3 wam_direct/start_mac_vlm.py --control-version 4 --vlm-auto-takeover`。不加此参数仍仅自动切 L2。两边新会话生效，旧进程不热更新。

接管条件全部满足：当前 L2/控制阶段运行至少10秒；持续12秒没有刷新进度最高值；最近8次有效独立相机预测覆盖至少6秒，replay仅记录、不作为门槛；当前进度低于90%。无效、断流、重复、旧条件/阶段和过期证据清空短期确认，保留同一控制阶段已经验证的历史最高进度。到达时有效的记录允许延迟读取以补充历史；实际接管仍要求最新图像不超过3秒，并满足完整的近期8次确认窗口。L2/控制阶段改变时才重新建立进度基线。最终L2同样可触发恢复；L2高进度切换优先。只在当前批次 drain 完成后执行现有受检查的 takeover 路径，保留当前L2和预算。Astra交还后重置统计、冷却15秒。T/P/B和退出优先；首轮仍人工Enter。窗口增加 Auto Astra 指标，runner记录 vlm_auto_takeover，Astra请求包含 takeover_trigger。

3/8 离线回放：此前停滞录像在裁剪后41.1秒首次满足条件，成功V4录像无候选。此为检测回放，不证明接管后一定改善结果。

## V4 初版：VLM 自动下一 L2

Mac 运行 `python3 wam_direct/start_mac_vlm.py --control-version 4`，复制打印出的命令到 Leo3 桌面终端。仍需手动 Enter 启动第一条 L2；后续满足进度条件时自动选择下一条并继续。运行使用 `config/astra/version4_vlm_subtasks.json`，独立于 V3 配置。

默认条件：同一 runner/session、L2 条件版本和控制阶段，连续两次不同相机时间戳的有效 40k 预测达到 100%，两次至少间隔 0.8 秒、至多间隔 2.5 秒，最近相机帧不超过 3 秒，预测到达时不超过 2 秒，当前 L2/控制阶段至少已运行 3 秒，anomalous_replay=false。单次高值、重复帧、无效/过期/断流结果会清空确认。默认 1 Hz；阈值由 `--vlm-progress-threshold` 配置（90 或 100），确认次数由 `--vlm-progress-count` 配置（2–10）。当前录像仅支持以 100% 作为初始候选，90% 尚未验证。

runner 在当前发布批次 drain 返回后、下次推理或发布前读取自身绑定的 predictions.jsonl，清除旧动作，复用 next_subtask/start 校验，保留 L3、步数和预算，用新观测推理新 L2。观察客户端只提供带身份的证据，不能发布动作。窗口显示 V4 和 Auto L2 确认状态；runner 记录 vlm_auto_subtask（含 request_ids 和阈值）。

人工 T/P/B 或退出优先；Astra 接管、暂停、等待 Enter 时不自动切换。首轮 Enter、B 后 Enter、手动 P 后 Enter 不变。最终 L2 不自动循环或完成。VLM 断开或证据不足时不自动切换，WAM 保持原有流程和预算，可手动 P/Enter。T 接管和 Astra 自主 hand_back 保留 V3 行为；停滞自动接管由上文的 `--vlm-auto-takeover` 显式启用。

已做真实日志离线回放、假传感器/发布器测试，并在人工启动的苹果/红椒任务中验证 WAM → Astra → WAM 链路；不代表其他任务的成功率。

## Mac 先启动（自动生成 RUN_ID）与窗口录像

在 Mac 仓库根目录运行 `python3 wam_direct/start_mac_vlm.py`。脚本自动生成全新 RUN_ID，打印对应的 Leo3 完整启动命令并启动 Mac 桥接等待；将打印的命令复制到 Leo3 桌面终端即可。不要在 Leo3 重新生成 RUN_ID。命令也保存到 Mac 的 `outputs/mac_supervisor/launch_<RUN_ID>/leo3_command.sh`。需要指定名字时加 `--run-id NEW_NAME`；`--prompt` 和可重复的 `--next-subtask` 可改任务。`--dry-run` 仅打印，不创建目录或启动桥接。

保留原有 `run_mac_supervisor.sh --session ...` 用法。新辅助脚本默认 V4（可加 `--control-version 3` 回到仅观察）、Mac Astra、PC WAM 29572、VLM 29573、20 fps、红椒两个 L2；它不会通过 SSH 启动机器人，也不会代按 Enter/T/B。

每次 VLM 图形窗口打开即自动录制合成画布：实时主相机、五图输入、L3/L2、进度、诊断提示及时间。录像 1200×660、10 fps，每 60 秒封存一段 MP4，按 `window_0001.mp4`、`window_0002.mp4` 排序。跟随模式路径为 Leo3 的 `outputs/progress_vlm/<RUN_ID>_follow/window_video/`，同目录 `segments.jsonl` 记录每段时间偏移和完成情况。窗口显示绿色 REC；编码器启动或写入抛错时显示红色 RECORDING FAILED 并记录错误，进度观察继续。

录制覆盖等待、运行和子任务切换，直到关闭观察窗口。结束后在 VLM 窗口按 Q/Esc，或正常关闭窗口，封存最后一段；Ctrl+C、SIGTERM、SIGHUP 也会尝试封存。强杀/断电可能损失正在写的最后一段，已经完成的分段保留。无界面相机 probe 不录像。录像由 Leo3 写盘，不增加 PC 模型请求频率。

# 独立 Progress VLM 观察链路

本节描述独立观察器与 V3 旁路用法。观察器没有机器人动作发布器、复位接口或 Astra 接管调用；V4 由 runner 消费其记录进行控制切换。

## 推荐：跟随同一个 Astra/WAM 会话的 L3 + 多 L2

在原有 `launch_tmux.py` 命令的 `--client astra -- run` **之前**添加 `--vlm-progress`：

```bash
export ROS_DOMAIN_ID=103
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
RUN_ID="pepper_v3_vlm_$(date +%Y%m%d_%H%M%S)"
cd /home/liberai/Liberai_Deploy_astra
python3 wam_direct/launch_tmux.py \
  --name "$RUN_ID" --vlm-progress \
  --client astra -- run \
  --supervisor-backend mac \
  --control-config config/astra/version3_auto_handback.json \
  --host 10.128.40.203 --port 29572 --fps 20 \
  --prompt "Task: Pick up the red pepper and place it in the bowl. Subtask: pick up the red pepper" \
  --next-subtask "place the red pepper in the bowl" \
  --session "/home/liberai/Liberai_Deploy_astra/outputs/astra/$RUN_ID" \
  --owner "$RUN_ID" --original-step-limit 400
```

这仍是 **V3**，tmux 名称为 RUN_ID，runner session 为上面指定的绝对路径。Mac supervisor 桥接仍按原方式绑定同一个 session；VLM 不负责替代或启动 Mac 桥接。

当前 Leo3 的已运行相机使用 ROS domain 103，上面的 export 与该机器一致；不要用默认 domain 0。启动器把这些 ROS 环境变量显式传给 VLM 窗口。2026-09-30 在 domain 103 的接收检查已收到三路图像，未发布任何动作。

- 主终端 Enter 开始后，VLM 自动读取该 session 的 L3/当前 L2，无需另填 prompt。
- 不按 P：一直用当前 L2，首帧保持不变，即使 WAM 再推理、T 接管或交还也不重设首帧。
- 主终端 P：按原流程选择下一条 L2；VLM 显示新 L2、清除旧统计，等待 Enter。
- 主终端 Enter 确认：使用新 L2 的第一次 WAM 推理保存的主视角作为首帧，自动开始新 L2 的进度判断。这张首帧来自 WAM 真正的输入观测，不是看到进度返回时随便截取的一帧。
- B 复位：跟随 runner 返回第一条 L2，等待 Enter，以新观测建立新首帧。
- VLM 窗口只有查看和 Q 退出；Enter/P/R 不会覆盖 runner 的任务选择。
- runner 暂停、复位、结束、状态陈旧时停止采用进度结果；旧 L2 在途返回被丢弃。

观察器只读 `manifest.json`、`status.json`、增量 `events.jsonl` 和已保存的头部图，不写 runner 文件。记录保存到 `outputs/progress_vlm/${RUN_ID}_follow/`，tmux 的 `VLM-progress` 窗口保留日志，图像在单独窗口显示。默认连接 PC 的 `29573`，可在 launcher 参数中用 `--vlm-host` / `--vlm-port` 覆盖，不影响 WAM 的 `--host` / `--port`。

如果 runner 已经运行，也可在 Leo3 图形终端单独附加；不要重启机器人 runner：

```bash
bash wam_direct/run_progress_vlm.sh \
  --follow-session "/home/liberai/Liberai_Deploy_astra/outputs/astra/$RUN_ID" \
  --output "/home/liberai/Liberai_Deploy_astra/outputs/progress_vlm/${RUN_ID}_follow"
```

使用新的观察器输出目录。即使中途附加，也会从该 L2 最早保存的 WAM 输入恢复首帧。

## 当前部署

- PC `10.128.40.203:29573`：40k 进度服务，独立于 WAM 的 `29572`。
- PC 模型和运行文件：`/data/trans/progress_40k_20260930`。
- PC 常驻 tmux：`progress-vlm40-20260930`；日志 `results/server_console.log` 和 `results/server_requests.jsonl`。
- Leo3 客户端：`/home/liberai/Liberai_Deploy_astra/progress_vlm`。
- Mac 源码位于用户提供的 `Liberai_Deploy_astra` 副本，新增模块与现有修改隔离。
- 使用既有 TCP 四字节长度前缀 + MessagePack。VLM 有独立协议、连接和端口，不复用 WAM 动作 socket。
- 服务是手动启动的用户级 tmux 进程，没有设置开机自启。

2026-09-30 从 Leo3 发送四组真实五图样本到 PC 全部成功，预测与 PC 本地测试一致。模型推理约 0.49 秒，往返约 0.65–0.82 秒；这些是录制样本的链路测试，不是实机准确率，也不是与 WAM 并发负载下的性能保证。

## 实时观察时启动

当前不需要启动机器人。待准备验证时，先按原有流程启动三路相机发布。在 Leo3 图形桌面的终端运行下面命令，替换真实的英文 L3/L2，并使用新的 output 目录：

```bash
cd /home/liberai/Liberai_Deploy_astra
bash wam_direct/run_progress_vlm.sh \
  --l3 "YOUR FULL TASK" \
  --l2 "YOUR CURRENT SUBTASK" \
  --output outputs/progress_vlm/live_001
```

窗口显示实时头部图和上一次有效推理的完整五图、进度、异常回放、停滞候选提示与延迟。启动时不推理，先等相机图像出现，在**该观察窗口**按 Enter，将此刻头部图设为 L2 首帧并开始推理。机器人应处于该 L2 起始状态；中途随意重设首帧会改变模型语义。

- Enter / R：重新记录当前 L2 首帧并清空统计，不执行机器人复位。
- `--next-l2 "next subtask"` 可重复配置；窗口 P 切到下一条并记录首帧，不执行动作。
- Q / Esc：退出观察客户端；PC 服务继续运行。
- 默认 1 Hz，一次只处理一个最新请求，不积压图像；后台采集不被模型推理阻塞。
- 每次有效预测保存五张实际输入图和 `predictions.jsonl`，用于检查误报、漏报及任务切换时的表现。日志会持续占用磁盘，结束观察时退出客户端。

## 相机与输入约定

订阅既有相机话题，不创建额外硬件采集器，不实例化带动作发布器的 LeoWujiInferEnv：

| 视角 | 话题 | 处理 |
|---|---|---|
| 头部/主视角 | `/cam/chest/image_raw/compressed` | 保持方向，时间戳减 0.100 秒 |
| 左腕 | `/cam/hand_left/image_raw/compressed` | 旋转 180 度，时间戳减 0.040 秒 |
| 右腕 | `/cam/hand_right/image_raw/compressed` | 旋转 180 度，时间戳减 0.040 秒 |

方向与延迟补偿和当前 WAM 环境一致。输入顺序为：约一秒前主视角、L2 首帧主视角、当前主视角、当前左腕、当前右腕。历史按时间戳取，不按固定帧数推测。新 L2 可以沿用前一秒历史，但首帧和趋势必须重新记录。

当前图像默认最多 1 秒旧，三视角及历史对齐误差容限 0.15 秒。返回结果图像年龄默认最多 2 秒；超时、断流、非法输出、旧子任务返回均不计入趋势。每条响应校验请求、会话、子任务及 40k 权重 SHA256。

## 如何理解输出

- `progress`：当前指定 L2 的进度档位，10 到 100；不是整个 L3 的完成概率。
- `anomalous_replay`：训练中倒放/交换时序样本的异常标签，不能直接解释为机器人 retry 或抓取失败。
- `plateau_candidate`：默认至少 5 个有效不同时间戳样本覆盖 8 秒，且窗口内进度变化不足 10 个百分点；是观察提示。正常动作阶段、量化平台、100% 完成和模型错误也可能产生该提示。
- `replay_streak`：连续有效 VLM 观测的异常标签次数，不是 WAM 动作回合数。
- 无效输出保持无效，不补成 0%，不触发任何控制切换。原始文本始终保留。

V4 已支持显式启用停滞自动接管，规则见本文开头。接管在批次边界清空旧动作后走既有 Astra 流程；阈值仍需结合任务记录评估误报、漏报。

## 不启动机器人也能复测通信

```bash
cd /home/liberai/Liberai_Deploy_astra
python3 -m progress_vlm.replay \
  --samples outputs/progress_vlm/fixtures_40k/samples.json \
  --output outputs/progress_vlm/replay_new/results.json
```

已有结果：`outputs/progress_vlm/link_test_40k_20260930/results.json`。相机探测记录在 `outputs/progress_vlm/camera_probe_20260930/camera_probe.json`，当次三路均为 0 帧。这是早期未启动相机时的探测记录；后续 domain 103 已收到三路图像，现场表现应以对应 session 的记录为准。

## PC 服务管理

```bash
# 查看现有服务日志
tail -f /data/trans/progress_40k_20260930/results/server_console.log

# 仅在原服务已退出、端口空闲时重新启动，使用新的 tmux 名称
tmux new-session -d -s progress-vlm40-new \
  'bash /data/trans/progress_40k_20260930/serve_progress.sh > /data/trans/progress_40k_20260930/results/server_console.log 2>&1'
```

首次启动会加载模型和预热，READY 后才监听服务。服务正在处理请求时拒绝另一并发推理请求，客户端下一次发送新图；没有排队执行陈旧请求。dp 环境和原 WAM 服务均保留。

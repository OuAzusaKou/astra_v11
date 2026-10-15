# V10：纯数据子会话与固定 Action Chunk

V10 从 V4 增量扩展，保留 WAM 常规执行、T 切入/切回、P/Enter、VLM 可选触发、代码处理停止兜底和硬件互锁。已有 V9 是点云候选选择版本，不占用或替换它。

## 数据边界

子会话只获得固定数据语义、当前 L3/L2、三路观测图像、实测机器人状态、按时间抽帧的示范视频、带发布状态的 WAM 历史以及本次纯数据会话的历史。旧 V4/V8 对话不会导入，避免把曾读取的源码带入新会话。状态第 0 行为当前值；历史和示范都不能当作当前实测。

输入由 `astra/action_chunk.py` 的字段白名单生成。不发送工程路径、SSH 信息、ROS 话题/服务、源码、控制函数、执行命令、旧监督提示或原始事件日志。子会话需要知道动作数据的含义，但不获得执行接口或实现源码。

- Astra：复用本机 Codex 登录。私有 loopback relay 在发送模型请求前，用白名单数据与完整纯数据历史替换 CLI 自动组装的 input/instructions，明确 `tools=[]`、`tool_choice=none`。模型响应完整接收后，拒绝所有工具输出类型，随后才交给 CLI。登录头只在内存中转发，不写日志。
- QuickRouter：直接发送相同白名单上下文和 JSON Schema，工具为空；收到工具调用、截断回复或非法格式直接拒绝。既有网络错误重试仍保留；没有“查代码—再调用模型”循环，也不做无限格式重试。
- 执行代码留在主程序。模型不能通过生成 Python、ROS 命令、函数名称等改变执行逻辑。

此边界适用于模型收到的上下文和可执行输出，不宣称把运行 CLI 的操作系统账号隔离成了独立容器。relay 仅绑定 loopback，使用每次调用的随机路径；不暴露机器人控制入口。

## 固定输出

机器可读定义：`config/astra/action_chunk_v2.schema.json`。主程序记录格式版本 `astra.action_chunk.v2`；子会话不回传版本或身份字段。旧 v1 schema 仅保留为历史记录。

所有回复（包括初始化握手）只能包含一个字段：`action_chunk`，不能附加 reason、evidence、status、stop_kind、request_id、说明或 Markdown。

- 非空 chunk：1～30 行，每行恰好 52 个有限数值。全部是绝对目标，顺序为左末端 xyz 米+旋转向量弧度（6）、右末端同格式（6）、左手关节弧度（20）、右手关节弧度（20）。末端坐标系为 chest_link，手指顺序与实测状态一致。
- 空 chunk：`{"action_chunk": []}`。初始化时只表示握手；执行请求中表示没有可执行动作，由固定代码进入 planning 暂停及既有交还兜底。硬件异常仍由主程序检测和锁存，不交还。空数组不表示任务成功。
- 身份由桥接绑定原始请求信封并校验；手动 T/B 取消、过期请求、数值/轨迹校验保持有效。模型不再通过文本声明完成或选择交还。
- 日志中的内部 reason/evidence 由代码生成并标注 Host，仅用于控制器协议兼容，不冒充模型解释或执行证据。子会话历史只记录原始 action_chunk。

桥接先校验格式；Leo3 收到绑定会话的 `v10_action_chunk` 信封后再次校验，固定适配器转为主控制器动作。主控制器仍执行硬件检查、轨迹/跳变检查、最新状态核对、预算、T/B 取消和过期响应拒绝；没有 eval/exec 模型代码，不填零、不修剪或修复非法动作。新一轮必须重新观察。模型输出不是硬件执行成功凭据。

## 启动

Windows 保留现有入口；Linux/macOS 使用 python3 和同样参数。先运行笔记本桥接，然后在 Leo3 桌面终端执行打印的完整 tmux 命令。首次执行仍需人工 Enter，不自动启动 ROS。

```powershell
cd C:\dynamics_eval\Liberai_Deploy_astra
C:\Python314\python.exe -u wam_direct\start_v10.py `
  --astra-model astra `
  --proxy http://127.0.0.1:7897 `
  --takeover-mode manual --l2-switch-mode manual --no-step-limit `
  --reference-video /home/liberai/Liberai_Deploy_astra/data/reference_videos/4ed8a596b026_V12_collect_1780699002.mp4 `
  --reference-video-frames 30
```

任务默认沿用笔记本与笔的四条 L2；其他任务用 `start_mac_vlm.py --control-version 10 --prompt ... --next-subtask ...`。模式可分别选 manual/vlm；no-step-limit 只取消总预算，单批最多 30 点保持不变。此示例使用 Astra 原有代理，不走 QuickRouter。替代模型仍通过 --astra-model 指定，直连使用 --proxy=；是否支持 strict JSON Schema 需对应提供商通信验证，失败不会退回有工具的模式。

## 日志与验证

桥接目录 supervisor/ 中：`<request>.public.json` 是纯文本输入，`<request>.model_input.json` 是 Astra 实际发送的完整数据（含图像），`<request>.reply.json` 是原始结构化回复，`data_history.json` 是纯数据会话历史。它们不包含认证头。Leo3 events.jsonl 记录 v10_chunk_received、既有 decision/batch_published 与暂停/交还事件。

离线测试：`python -m unittest discover -s tests -p test_action_chunk_v10.py -q`。测试使用假机器人和本地模拟 HTTP，不发布 ROS 动作。实际通信验证使用独立会话，不连接机器人；通信成功不代表真机任务成功。

Codex 配置参考：https://learn.chatgpt.com/docs/config-file/config-reference 。仅依赖工具开关不足以保证该模型完全不带 agent 工具，所以 V10 另外在实际出站请求和入站响应处强制执行数据边界。

2026-10-08：已在 Windows 和 Leo3 完成假机器人及已有版本回归；实际 Astra 纯通信验证通过初始化握手、合成图片识别、两轮历史记忆和固定格式输出，所有测试回复均为 observe/空动作。没有进行机器人启动或动作测试。QuickRouter 替代模型走相同数据边界的离线用例已通过，本次未进行这些模型的实际通信或真机验收。

2026-10-08 输出精简：新会话使用 v2 单字段输出协议；需两端同时更新并重启新会话，旧会话不热更新。

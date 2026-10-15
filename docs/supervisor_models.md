# V4 / V8 启动时选择接管模型

模型由最开始的启动程序指定，整场固定。没有自动模型路由、失败后换模型或运行中热切换。原 Astra（Codex `gpt-6-astra`）保持默认；QuickRouter 替代入口仅用于 V4 / V8。机器人控制、WAM、VLM、Dynamics 和动作审核规则沿用原实现。

## 直接在代码里选择

编辑 `wam_direct/start_drawer_test.py` 顶部：

```python
MODEL = 'gemini'        # astra / opus5.5 / gemini / glm5.3-flash
CONTROL_VERSION = 8    # 4 或 8
DIRECTION_DISTANCE_M = 0.10
```

Windows 在仓库根目录先检查生成命令：

```powershell
python wam_direct/start_drawer_test.py --dry-run
```

准备新一轮测试后去掉 `--dry-run`，启动笔记本桥接，再将生成的完整命令复制到 Leo3 图形终端。首轮仍由操作者按 Enter。这个入口固定为“只开抽屉”，开启 VLM 停滞接管；接管模型优先自主尝试和反馈调整，确实无法继续而主动暂停时才使用此前配置的交还兜底；V8 使用 Dynamics review 模式。每次生成新 RUN_ID，修改启动配置不会改变已启动的测试。

## 接管后优先自主处理（2026-10-05）

Astra、Opus、Gemini、GLM 复用同一接管策略：保留控制权，自主提出有依据的策略、执行版本允许的有界动作、检查真实反馈，策略无效后作有针对性的调整。V4 用独立 takeover；V8 继续通过选手、方向加policy候选、Dynamics预测和候选审核执行。一般性不确定、缺少精确坐标、WAM停滞或一批结束，都不足以直接交还。

正常 hand_back 要有真实恢复证据和交给 WAM 的具体理由；启用 `--astra-pause-handback` 时，非硬件主动暂停交还仍是最后兜底。暂停前说明已做的尝试与反馈，或没有合理动作可尝试的具体依据；不强制尝试危险动作。硬件故障、急停和技术故障检查保持原有行为。这是共同提示策略调整，不是强制运动或禁止交还的控制器规则，也不保证模型总能规划成功。

完整策略需要新启动的监督桥接加载；已提交的 API 请求不会被热改。更新代码不自动重启机器人或恢复暂停的测试。

## 通用启动参数

也可直接指定 `--astra-model`：

- `astra`：原 `gpt-6-astra`，原 Codex 登录、会话和传输方式。
- `opus5.5`：QuickRouter `claude-opus-5-5`。
- `gemini`：QuickRouter `gemini-3.8-flash`。
- `glm5.3-flash`：QuickRouter `glm-5.3-flash`。
- `qwen3.8-max`：QuickRouter 同名型号，自动使用 `preloaded_no_tools` 无工具模式；见下节。

### Qwen 3.8 无工具适配

V4/V8 在启动时选择 `--astra-model qwen3.8-max` 即启用此适配，不另建控制版本。Qwen请求不发送 tools/tool_choice；桥接先分页完整读取 Leo3 的控制接口 `envs/leo_wuji_env_infer.py`、`astra/robot.py`、姿态转换 `utils/pose_utils.py`、V4/V8控制配置和 ROS `_leo.urdf.xacro`，保存到 `supervisor/control_context.json`，包含源路径、带行号的原文、读取时间和内容哈希。读取失败、截断或页数变化均拒绝启动。Windows远程模式使用原只读SSH读取器；Linux本地监督使用同一读取白名单。

启动握手改为随机图片识别＋确认已注入资料的哈希、52维绝对动作和chest_link坐标系；不再要求Qwen自行调用工具。握手消息与资料保留在同一API会话历史中。动态图像、状态、WAM历史及V8候选图仍逐轮完整提供。不存在可供Qwen调用的shell/SSH/ROS工具；意外返回tool_calls直接拒绝，不解释执行。静态源码不是实时TF、完整标定、接触或碰撞证明，未包含的xacro引用也不能当作已读取。缺少必要证据时明确observe或pause，不捏造查询结果。

保留请求身份、图像校验、动作范围、硬件检查、V8候选审核、HTTPS重试与取消机制；其他模型仍使用原工具模式。该适配需要重启监督桥接，并通过带 `inspection_mode=preloaded_no_tools` 的两端握手；不自动启动机器人或恢复旧测试。

Opus、Gemini、GLM 也接受对应的完整模型 ID；这些替代模型统一调用 `https://api.quickrouter.ai/v1/chat/completions`。模型 ID 通过该账号 `/v1/models` 核实；协议说明见 [QuickRouter 文档](https://doc.quickrouter.ai/docs/model-catalog.html)。

```powershell
python wam_direct/start_mac_vlm.py --control-version 8 --astra-model opus5.5 --single-subtask --dynamics-mode review --vlm-auto-takeover --astra-pause-handback --prompt "Task: Pull the storage rack drawer outward by a moderate distance. Subtask: pull out the drawer"
```

V4 将 `--control-version 8` 换为 `4`。helper 同时给 Leo3 命令和本地桥接传入相同模型；握手时再次检查两端身份，配置不一致拒绝启动。V5～V7 保留原模型行为。

Linux/macOS 笔记本使用同一个 Python 入口；在 Leo3 本地监督时可给 `astra.run run --supervisor-backend local` 添加相同 `--astra-model`。环境变量和凭据必须配置在**实际运行监督模型的一端**。

## 凭据与网络

代码不接受 `--api-key` 明文命令行参数，key 不写入源码、runner manifest 或模型请求日志。优先读取监督主机上的 `QUICKROUTER_API_KEY` 环境变量。

当前 Windows 笔记本另支持已配置的 DPAPI 文件 `C:\dynamics_eval\.secrets\quickrouter.dpapi`，位于工程目录之外，绑定当前 Windows 用户。可用 `QUICKROUTER_KEY_FILE` 指向另一个 DPAPI 文件。此文件不复制到 Leo3；Linux/macOS 使用环境变量。更换 key 可在 PowerShell 通过隐藏输入更新：

```powershell
$secret = Read-Host 'QuickRouter API key' -AsSecureString
$secret | ConvertFrom-SecureString | Set-Content -LiteralPath C:\dynamics_eval\.secrets\quickrouter.dpapi -Encoding ascii
```

桥接默认使用已有 `http://127.0.0.1:7897` 代理；本机实测 QuickRouter 直连失败，经该代理成功。`--proxy` 配置笔记本监督代理；Leo3 本地监督用 `--supervisor-proxy`。HTTP 重定向不携带密钥跟随，失败不自动更换模型。

## 图像、工具、会话和取消

- 复用 V4/V8 的完整监督提示、当前相机图片、WAM 参考、示范与 V8 候选预测图片。图片按原顺序编码为 data URL，依据真实文件头识别 PNG/JPEG，不因缓存扩展名误判格式。
- 工具模式提供 `read_file`、`list_directory`、`search_text`、`ros_query`，复用已有项目/ROS 路径白名单和只读 SSH 查询；没有任意 shell 或机器人发布工具。Qwen 使用上文的桥接预加载模式，不向模型提供工具。
- 每次启动用随机验证码图片与实际控制源码读取做握手。工具模式必须确实识图并成功使用只读工具，才报告 READY；Qwen 必须识图并确认桥接预加载的控制资料。失败不退化成纯文本机器人监督。
- API 请求运行于可终止子进程。T/B、请求变化或退出取消旧调用；完成后仍验证 session/owner/request_id。JSON 不完整、截断、错误身份、超时和网络错误不会放行动作。
- QuickRouter 使用本地持久消息历史，`conversation_kind=local_api_history`。`thread_id` 字段保留 UUID 兼容邮箱协议，它**不是 Codex 会话 ID**，不能用 `codex resume` 打开。原 Astra 仍使用真实 Codex thread。
- 历史完整保存在监督端 `supervisor/api_history.json`，不会静默裁剪旧图或工具记录；长测试可能遇到供应商上下文上限，届时暂停报错，不自动切模型。
- `supervisor/*.input.json` 保存本轮提示与图片路径/哈希，`*.tools.jsonl` 保存只读工具结果，`*.json` 保存决策。日志含实际 provider/model，便于区分实验结果。

## 验证范围

2026-10-06，Qwen 无工具适配在 Windows 完成真实启动握手：桥接从 Leo3 读取控制资料，Qwen 正确识别随机图片并确认资料哈希与动作协议，全程约28.27秒。跨平台回归72项，71项通过、1项按平台跳过。

完整 V4 冻结请求另做了隔离回放（12张图片，请求10884133字节）：三次尝试均在等待HTTP响应头阶段约60秒发生 RemoteDisconnected，未取得决策。只能确认当前网络/供应商链路仍有完整请求断线问题，不能仅凭这些日志判定是大小限制或哪一侧超时。错误日志在 Windows `C:\dynamics_eval\quickrouter_diagnostics\qwen_full_20261006_114443_e91334\supervisor\api_transport.jsonl`。回放未连接机器人动作提交路径，未启动真机；启动握手成功不代表完整任务执行验收通过。

### API 断线重试与错误日志（2026-10-06）

QuickRouter 的每次 HTTP 请求最多尝试3次（首次加2次重试），默认退避2秒、4秒；数字 Retry-After 可延长等待至最多60秒。RemoteDisconnected、TLS意外EOF、连接重置、网络超时、HTTP408/429/500/502/503/504可重试；认证/权限、证书验证、无效JSON和不完整模型决策不自动重试。单次连接/读等待最多120秒，总监督轮次仍受600秒期限约束，重试不重置总预算。

重试同一模型和同一HTTP消息内容，只在完整有效响应到达后处理工具或提交决策。取消/交还可终止请求及退避等待；不重发已执行的工具、不重放机器人动作、不自动恢复已暂停的旧测试。上游可能已经处理了断线请求，因此重试可能增加模型计费；没有服务器端幂等保证。

监督端 `supervisor/api_transport.jsonl` 记录请求ID、调用ID、尝试次数、阶段、耗时、请求/响应字节数、HTTP状态和异常类型，不记录key、完整请求/响应正文或任意上游错误文本。`open_or_wait_response_headers` 表示尚未取得HTTP响应头，urllib无法进一步区分连接、代理隧道与等待响应头；`read_response_body` 表示已收到响应头后读取中断。日志不能单独确定代理与上游哪一方主动断开。

`supervisor.log` 显示 `API WAITING`、`API RETRY`、`API FAILED` 及 `TOOL` 进度，工具结果文件增加时间戳。现有运行中的监督进程需要重新启动桥接才能加载重试循环；同步文件不会自动恢复机器人。

2026-10-05，三个指定型号均通过 QuickRouter 的随机图片识别与 Leo3 `read_file` 工具完整往返：Opus 约7.6秒、Gemini 约10.9秒、GLM 约4.9秒。这是小请求连通性/能力验证，不能当作真实机器人规划质量、完整候选选择质量或端到端延迟保证。

离线测试覆盖七图输入、PNG/JPEG识别、工具白名单、完整历史、错误身份/截断/取消拦截、实际子进程取消、模型启动绑定，以及原 Windows 桥接回归。替代模型尚未进行 V4/V8 真机执行验收。开发过程中没有给正在运行的机器人会话更换模型。

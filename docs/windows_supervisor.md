# Windows 原生 Astra 监督（增量支持）

Windows 笔记本运行 Astra/Codex 与 SSH 桥接；Leo3 继续运行 Linux runner、ROS 和相机；WAM/VLM 继续使用已有服务器。无需 WSL 或 Git Bash。既有 Linux 本地监督和 Linux/macOS SSH 监督入口保留。

## 前置检查

在本机用户的 PowerShell 中运行（不要使用其他服务账号）：

```powershell
cd C:\dynamics_eval\Liberai_Deploy_astra
python --version
python -c "import numpy; print(numpy.__version__)"
codex --version
codex login status
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no leo3 "test -d /home/liberai/Liberai_Deploy_astra && echo REPO_OK"
```

桥接不需要本机 ROS、CUDA、PyTorch 或 MessagePack。使用本机已有 Codex 登录，不复制 Leo3 的凭据。Python 3.10+；本机验证环境为 Python 3.14 / Codex CLI 0.160.0。

Windows 自动将 npm 的 `codex.cmd` 定位到同一安装目录中的原生 `codex.exe`，不通过 cmd.exe/Bash 执行模型参数。若安装布局不同，传 `--codex-bin "C:\实际路径\codex.exe"`。Windows 使用每次调用独立的 Job Object；取消、超时或调用结束时清理所属子进程。无法建立该隔离时，在发送模型提示之前报错。Linux 保持 Bash + POSIX 进程组；macOS 保持直接 CLI + HTTPS 配置。三端沿用相同的请求身份校验和只读监督权限，不更改全局 Codex 配置。

### 模型内部的 Leo3 查询

Windows 沙箱账号不能继承用户 SSH 配置和私钥；因此外层桥接连接成功，不代表模型执行 `ssh leo3` 也成功。Windows 远端监督现按调用注册 `astra_readonly` MCP：由正常用户身份使用既有 SSH 认证，仅提供 `read_file`、`list_directory`、`search_text` 和 `ros_query`。源码/URDF 读取限制在工程、ROS 工作区、camera_pose 和 ROS share 目录；阻止越界和私有路径；查询有输出上限与超时。不提供任意 shell、发布话题、写文件或改变状态的服务调用，不复制私钥或修改沙箱 ACL。

启动握手要求 Astra 通过该工具实际读取 Leo3 的控制定义后才返回 READY。监督提示明确要求 Windows 使用该工具，不能用沙箱内的 `ssh leo3` 失败来代替查询。原 Linux/macOS 直接 SSH 行为保持不变。新增模块必须同步到 Leo3；旧桥接进程不会热加载，更新后使用新会话。Windows 的模型初始化、源码查询和真实动作规划是三个不同验证环节，通信测试通过不代表抓持/路径已有足够依据。

## 先只看命令

```powershell
python wam_direct/start_mac_vlm.py --control-version 4 --dry-run
```

脚本名称保留以兼容旧入口；会自动选择 Windows 原生桥接。`--dry-run` 只打印，不创建目录、不启动模型、桥接或机器人。

## 以后启动：先笔记本，再 Leo3

下面以红椒入碗为示例；运行前替换实际 L3/L2。每次自动生成新的 RUN_ID。

```powershell
cd C:\dynamics_eval\Liberai_Deploy_astra
python wam_direct/start_mac_vlm.py `
  --control-version 4 `
  --prompt "Task: Pick up the red pepper and place it in the bowl. Subtask: pick up the red pepper" `
  --next-subtask "place the red pepper in the bowl"
```

保留 PowerShell 终端，将输出的整段 Bash 命令复制到 **Leo3 图形桌面终端**。两端共用打印的 RUN_ID，不在 Leo3 重新生成。命令还保存在 `outputs/mac_supervisor/launch_<RUN_ID>/leo3_command.sh`，编码 UTF-8、LF 换行。

- 版本：V4，`config/astra/version4_vlm_subtasks.json`。
- tmux 名称：打印的 RUN_ID。
- runner session：`/home/liberai/Liberai_Deploy_astra/outputs/astra/<RUN_ID>`。
- WAM：`10.128.40.203:29572`；VLM：`10.128.40.203:29573`；ROS domain：103。
- 协议后端仍名为 `--supervisor-backend mac`，表示 SSH 邮箱；Windows 也使用它。`local` 仍表示在 Leo3 运行监督模型。

笔记本先等待 session 出现；Leo3 启动后才执行 Astra 握手。双方出现 `MAC READY` / `READY` 后，由操作者在 Leo3 主控制窗口按 Enter。桥接不会启动远端 runner、ROS/CAN 或代按 Enter/T/B；Ctrl+C 退出笔记本桥接不是硬件急停。

默认 V4 自动切 L2，T 人工触发 Astra，Astra 可自主交还 WAM。需要停滞自动接管时，在上述笔记本命令添加 `--vlm-auto-takeover`，脚本会将其写进 Leo3 命令。最后一条 L2 不自动宣布整局完成。

默认代理为 **笔记本本机** `http://127.0.0.1:7897`。修改时传 `--proxy http://实际地址:端口`；取消额外代理覆盖可用 `--proxy=`（兼容旧 PowerShell 的空参数处理；继承已有环境变量）。SSH 别名可用 `--host` 修改；这不会修改生成命令中的 Leo3 工程绝对路径。

## 单独运行桥接与回归测试

已有精确 session 时可直接运行：

```powershell
python -B -m astra.mac_bridge --session /home/liberai/Liberai_Deploy_astra/outputs/astra/本次RUN_ID
python -B -m unittest discover -s tests -p test_supervisor_platform.py -v
```

平台测试不访问模型/网络/ROS，只用模拟 CLI 和真实本机测试子进程验证中文、路径、续接、取消与超时清理。Linux 可运行同一文件，并继续跑原有 `test_astra_remote.py`、`test_astra_bridge_reset.py`、`test_version_routes.py`、`test_vlm_auto_l2.py`、`test_vlm_takeover.py`。

不能把通信检查或离线回归通过当作真机动作验证；启动前仍需检查 Leo3 三路相机、硬件状态和录像磁盘空间。

## 终端输入与连接恢复

SSH 邮箱默认连接超时 15 秒、单次调用超时 60 秒；只读/握手请求失败额外重试最多 3 次（退避 2/4/8 秒）。可用 `--ssh-connect-timeout`、`--ssh-timeout`、`--ssh-retries` 调整；响应提交超时不重发，以免重复动作。身份、协议或校验错误不按网络故障重试。

更新后的 Leo3 runner 仍在心跳超过 30 秒时停止放行动作。仅由通信故障产生的暂停，在桥接恢复且新鲜心跳持续 2 秒后自动用新观测重试，最多 3 次/会话，单次恢复窗口 300 秒。短暂 SSH 故障也通过连接代数强制废弃旧请求。硬件检查不绕过，不清故障锁存；模型主动暂停不走通信重试。桥接 `--no-auto-reconnect` 可关闭自动恢复。

本次用户指定 `--astra-pause-handback`：仅对显式启用的新会话，Astra 的 `pause_kind=planning/collision_risk/observation_uncertainty` 主动暂停转为交还 WAM。包括碰撞风险判断；交还不证明风险消失或恢复成功。保留 L2/预算，丢弃旧动作，以新观测推理。`pause_kind=hardware_fault`、硬件检查失败、未分类决策或程序错误仍暂停。监督提示会明确要求分类；不修改旧暂停记录。默认不开启，原 Linux/macOS 运行策略可保留。

Windows 与 macOS 监督调用使用项目已有的 HTTP-only Responses 提供商配置，避免代理路径上反复失败的 WebSocket 握手；Linux 本机监督调用保留原传输配置。

Leo3 键盘读取按完整终端序列过滤方向键、功能键、鼠标和终端响应，启用 bracketed paste 并丢弃其内容。独立 B 仍执行原有复位；每次新的 B 请求记录时间、字节十六进制和 TTY。该日志只证明输入来源是终端流，不证明人工按键。2026-10-05 会话的异常输入来源仍未确认，方向键误识别已在隔离 PTY 复现并修复。新增 `test_terminal_input.py` 只测试解析和虚拟终端，不连接机器人。

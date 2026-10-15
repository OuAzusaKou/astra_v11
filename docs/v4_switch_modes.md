# V4 接管与 L2 切换方式

两个独立参数，可以任意组合，仅对 V4 生效：

- `--takeover-mode manual`：关闭 VLM 停滞自动切入，按 T 从 WAM 切到 Astra，再按 T 切回 WAM。
- `--takeover-mode vlm`：保留 VLM 停滞自动接管，也允许随时按 T 双向切换。
- `--l2-switch-mode manual`：关闭 VLM 自动推进 L2，在 WAM 阶段按 P 选择下一条，再按 Enter 确认开始。Astra 阶段先 T 切回 WAM。
- `--l2-switch-mode vlm`：按既有 VLM 进度条件自动推进 L2，P/Enter 仍可人工切换；最终 L2 不自动宣告成功。

人工接管模式按操作者确认，仍保留 `--astra-pause-handback`：模型最终报错或停止时由代码自动交还 WAM。硬件故障/急停保持暂停，T 也不能解除硬件互锁。首轮和 B 复位后仍需 Enter。切换不会重置预算、不会重放旧动作；P 沿预设列表前进，不循环。

VLM 观察窗口及录像继续运行；manual 只取消对应自动触发，不关闭观测。启动终端、manifest.json 和 status.json 会记录实际选择。旧命令保持兼容：未指定新参数时，L2 默认 vlm；接管由旧 `--vlm-auto-takeover` 决定。显式 `--takeover-mode` 优先于旧开关。

笔记本任务和抽屉任务的启动脚本增加 `TAKEOVER_MODE`、`L2_SWITCH_MODE` 常量，当前保持之前的 vlm/vlm 默认值。可手动改为 manual，也可命令行覆盖：

```powershell
python wam_direct/start_notebook_test.py --takeover-mode manual --l2-switch-mode manual
```

Linux 的 `python3 -m astra.run run` 同样接受这两个参数。新参数不用于 V8，V8 原有候选流程和10cm默认距离保持不变。修改启动配置只影响新会话，不自动重启或切换现有测试。

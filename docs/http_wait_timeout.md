# QuickRouter 等待时间

2026-10-06：HTTP 连接/读等待上限从120秒改为300秒，适用于 Windows/Linux 的 QuickRouter 监督桥接。默认请求函数也使用300秒。每次调用仍受整轮600秒剩余预算限制；例如只剩41秒时，该次请求只等待41秒。最多3次尝试，重试不重置整轮预算，取消仍会终止 HTTP 子进程。

`supervisor/api_transport.jsonl` 新增 `http_wait_budget`，记录实际 `socket_timeout_seconds` 和 `turn_remaining_seconds`；传输诊断也记录 socket_timeout_seconds。已启动的桥接进程需要重启才加载父进程的新上限，已发送请求不能延长。

此项是客户端等待上限，不是 QuickRouter 上游网关配置。既有记录在60.2～61.1秒报 RemoteDisconnected，发生在取得响应头之前，原客户端上限却是120秒，因此不能据此认为客户端超时导致断开。若对方仍在60秒主动关闭，设置300秒也不能阻止。当前 Clash/Mihomo 配置未找到对应的60秒响应等待项，不能用 TCP keep-alive、连接建立或健康检查的 timeout 冒充该响应上限。未擅自修改代理服务配置或声称已更改 QuickRouter 服务端。

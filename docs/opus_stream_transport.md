# Opus response delivery

On 2026-10-06, Windows non-streaming Opus takeover requests repeatedly failed
with `RemoteDisconnected` while waiting for headers at about 60 seconds,
despite a 300-second client timeout. QuickRouter success reported by the
operator does not prove the response reached this client.

A communication-only replay of a saved takeover input (same images, prompt,
bootstrap history and read-only tool definitions; no tools or robot actions
executed) returned HTTP 200 streaming headers through the existing proxy in
8.374 seconds and a complete `stop` / `[DONE]` response after 89.809 seconds.
Direct non-streaming access reset with Windows error 10054 after 0.588 seconds.
This supports using streaming for this route; it does not identify which
proxy/gateway closes the non-streaming connection or guarantee future uptime.

`claude-opus-5-5` now requests SSE streaming. The transport assembles text,
indexed function-call deltas and provider extensions, requires a finish reason
and `[DONE]`, and returns the existing completion structure to the supervisor.
Only then can the supervisor run read-only tools or validate the final JSON
and current request identity. Partial decisions are never submitted. Premature
EOF is a retryable transport failure; cancellation still terminates the isolated
HTTP child. The 300-second socket wait, 600-second turn budget and three-attempt
limit remain. Other models keep their existing transport setting.

Transport logs include `stream`, header elapsed time, first stream event,
stream event count and completion size. Logs do not contain credentials or
stream text. A new Windows bridge process is required to load the updated
supervisor; deploying files alone does not restart a robot session.

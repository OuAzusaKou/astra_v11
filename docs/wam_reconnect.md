# WAM inference connection recovery

The notebook run `notebook_pen_v4_astra_20261006_163143_206b2a` returned
from Astra through the configured model-pause fallback at 16:55:54 on
2026-10-06 (Asia/Shanghai). The next WAM inference failed with
`model server disconnected`; the runner paused at step 570. T presses at
16:56:09 and 17:00:25 were ignored because WAM was already paused.
`pending Mac response invalidated` was normal cancellation of the old request.
The server's reason for closing the connection has not been confirmed.

The inference client now closes sockets idle for at least 30 seconds before
the next request. Network failures close the socket and raise a distinct
transport error. The controller makes at most three inference attempts, with
0.5/1-second backoffs, taking a new observation and generating a new request
on each attempt. Only the successful attempt enters WAM history/queue.
These are inference retries, never action-publication retries.

Operator stop or pending control/subtask/reset switch cancels retries. Hardware
checks run before each observation and after success. Protocol, server profile,
decoder and action validation failures are not transport retries. Exhausted
network retries still pause; T does not clear a WAM fault. The terminal now
prints the pause reason explicitly. Events include `wam_reconnect`,
`wam_reconnected`, and `wam_reconnect_exhausted`.

This covers manual T and code-owned fallback, including V4/V8. Changes take
effect in new runner processes; deployment does not resume a paused robot.

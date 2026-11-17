# Kimi streaming admission errors before HTTP 200

Real KVV replay with the repaired Python frontend and the original native ABI passed 99 of 108 selected original failures. Nine streaming negatives still received HTTP 200: five dynamic-tool validation cases and four recursive-schema rejections. Their nonstreaming counterparts correctly returned 400. The Python rejection was therefore not the missing behavior; HTTP had already committed SSE success.

For Kimi K3/K3.1 chat-completion models, the native handler now selects its existing `BackendErrorCheck::UntilFirstEvent` pre-commit path instead of the service's default Skip/timing-window policy. It waits through normal leading annotations, maps an initial typed backend admission error to its HTTP status, and only then returns the streaming response. Other model families retain their configured policy. No error is converted into a fabricated successful completion and no fixed timing window is used to approximate correctness.

Tradeoff: successful Kimi stream response headers can be delayed until the first backend event. Existing client-disconnect cancellation remains active during that wait; deployment proxy/read timeouts must accommodate real model first-event latency. This does not introduce retries or change generation content/usage.

The original nine KVV nodeids are the regression coverage. Per-case replay results and paired frontend/native version fingerprints are recorded in workspace `engine侧修复` documents; the Python-only replay is not evidence that a new native library has been loaded.

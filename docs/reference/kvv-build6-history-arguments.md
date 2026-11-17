# Kimi build6 historical tool arguments

Cases 086/095/096/098 failed before Python preprocessing because the native HTTP validator required historical assistant tool-call arguments to be JSON object strings. Kimi's encoder instead preserves the original argument string and renders malformed/non-object history through its raw JSON representation.

`HistoryArgumentsPolicy` separates the unchanged Strict behavior, the existing GLM malformed-JSON compatibility behavior, and explicit Kimi AsSent replay. AsSent does not parse, reserialize or wrap the original string; tool_call_id ordering and uniqueness checks remain active. This does not relax validation of newly generated tool calls.

Enable only for explicitly configured Kimi model IDs on the SGLang frontend:

```
--dyn-chat-processor sglang --kimi-history-arguments-models kimi-k3.1
```

The equivalent environment variable is `DYN_KIMI_HISTORY_ARGUMENTS_MODELS`. The default model list is empty; unrelated models retain their existing policy. A newly built native extension is required: Python-only hot updates cannot bypass an earlier native HTTP rejection.

Regression coverage includes the policy matrix and tool_call_id protection in the Rust validators. Per-case live results and commit mapping are maintained in the workspace `0ai/kimi-k31/kvv测试/engine侧修复` directory; documentation alone is not evidence of a successful live replay.

The Kimi Python materialization path also bypasses the generic `json.loads` conversion of historical arguments. That conversion turned array/scalar strings into unsupported Python values and reserialized valid nested literals. Other templates keep their existing mapping conversion. Validation therefore covers the actual encoder path, not only native HTTP acceptance.

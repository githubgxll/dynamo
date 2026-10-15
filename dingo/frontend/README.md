<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## SGLang chat generation limits and stops

The SGLang chat processor defaults to 8192 generated tokens when both
`max_completion_tokens` and `max_tokens` are omitted. An explicit
`max_completion_tokens` takes precedence over `max_tokens`. The limit includes
reasoning and answer tokens; reaching it returns `finish_reason: "length"` and
can leave the answer empty if all tokens were spent reasoning.

For the exact model ID `glm-5.3` (case-insensitive, also repository-qualified
IDs such as `zai-org/GLM-5.3`), negative `frequency_penalty` values are accepted
but projected to `0.0` for sampling. This compatibility policy targets normal
completion, as observed with negative values on the official GLM-5.3 API; it
does not assert how that API implements the parameter. Nonnegative values and
other models retain their existing behavior. Arbitrary serving aliases do not
activate this policy. Reasoning remains enabled, and the generation limit
remains a fallback rather than a guarantee that every prompt will finish.

With an active reasoning parser, string `stop` sequences apply after reasoning
separation, to the non-reasoning output (including tool syntax). A match stops
worker generation and returns `finish_reason: "stop"` (or `"tool_calls"` when
applicable). Stops split across chunks are supported; `include_stop_str_in_output`
controls whether the matched string is retained. EOS and `stop_token_ids` remain
worker-side conditions. Without a reasoning parser, including requests that set
`separate_reasoning: false`, string stops retain their worker-side behavior.

When no reasoning parser is explicitly configured on the frontend or in worker
metadata, the frontend selects `glm45` for GLM-5.3 / GLM-5.3-Flash if their chat
template opens a thinking block. Qwen3 continues to require explicit `qwen3`
reasoning parser configuration.
Detection uses the discovered model's `config.json`, so serving aliases do not
affect it. Explicit parser settings take precedence. Unknown model types with
thinking templates produce a warning requesting explicit parser configuration.

## GLM-5.3 malformed historical tool arguments

For a GLM-5.3 deployment using `--dyn-chat-processor sglang`, opt in with
`--glm-history-arguments-models glm-5.3` (environment default:
`DYN_GLM_HISTORY_ARGUMENTS_MODELS`). This is a comma-separated list of exact,
case-sensitive canonical served model names, after request alias resolution.
An empty list, including an explicit empty CLI value, keeps strict validation.

The setting applies to `/v1/chat/completions` and `/v1/responses`, including
streaming. It permits malformed JSON **syntax** in historical assistant
`tool_calls[].function.arguments` / Responses `function_call.arguments`.
Valid JSON arrays, strings, numbers, booleans and null remain invalid; the
existing acceptance of empty argument strings is unchanged. Other validation
and newly generated tool-call handling are unchanged. This does not make
malformed arguments safe to execute: the client must return a tool error and
let the model retry with valid arguments.

Rust owns admission and receives this deployment setting explicitly through
PyO3. Python preserves malformed strings and adapts the discovered GLM Jinja
in memory for the named models, including preprocessing subprocesses. Object
arguments retain the original `<arg_key>` / `<arg_value>` rendering; strings
are preserved verbatim after the tool name. No shared model files or
frontend/worker wire schemas change. Unknown template revisions fail model
initialization rather than silently applying an unverified transformation.
An explicitly supplied `--chat-template` is subject to the same check.

Use a dedicated GLM frontend and the same immutable image on frontend/workers
for initial rollout. Verify valid/invalid history, streaming, Responses and a
tool-error/retry round trip before production rollout. Removing the option
and restarting the frontend restores strict admission and the original
template. Native `sglang serve` does not use this Dingo configuration.

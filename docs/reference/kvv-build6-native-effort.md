# Kimi native effort precedence

Build6 cases 023/046 (`test_reasoning_effort_ignored_when_effort_present`, nonstream/stream) send `thinking.effort=low` together with `reasoning_effort=max`. The native normalizer previously extracted only thinking type and top-level effort, then discarded the thinking object before Python saw it. A Python-only change cannot recover the discarded nested value.

The Kimi-specific native branch now resolves nested effort before dropping the object, synchronizes the downstream root effort and exports the encoder-level decision through template arguments. Explicit disabled thinking wins. Kimi effort-only objects are accepted consistently with the Kimi adapter; other model families retain their previous normalization order.

Rust regressions cover nested-vs-top precedence, effort-only Kimi payloads, disabled precedence and non-Kimi isolation. The Kimi Python adapter consumes the normalized result without reinstating a stale root value. Real build6 case results are tracked in the workspace per-case documentation, separately from local/module test results.

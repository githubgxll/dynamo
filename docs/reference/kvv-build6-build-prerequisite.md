# Native build prerequisite: unavailable AISimulate pin

The existing workspace and Python binding pinned `aisimulate-core=0.13.0-dev.202610060000000067`. The public crates.io index did not contain that version and its crate download returned HTTP 404, preventing even a feature-disabled native extension build from resolving dependencies.

With explicit user approval, both manifests select the published `0.13.0-dev.2`, which retains the existing handoff/replay/request structures and the required G2Scope, EngineLaunchConfig, KvEventTier and agentic-profile APIs. Cargo regenerates both lockfiles. The initial October 29 candidate added incompatible fields/events; the September 27 candidate lacked required APIs. Both were rejected rather than changing unrelated mocker semantics or inventing adapters. This is a separate build prerequisite, not a KVV protocol workaround.

The KVV frontend build does not enable the optional AIS forward-pass feature, although AIS types are still compiled through the mocker dependency. AIS-specific runtime behavior is not verified by KVV; deployments using those functions must run their own compatibility checks. No placeholder crate, forged version, dropped lifecycle event or disabled protocol validation is used.

The native baseline also referenced `self.logit_bias` after that field had moved into the inner OpenAI request in dynamo-protocols 6.1.0. Validation now reads `self.inner.logit_bias`; the validation rule and client field remain unchanged. Linux build prerequisites include the standard protobuf headers (`libprotobuf-dev`), not just the protoc executable.

The Python EntrypointArgs signature and runtime-config setter already exposed `tokenizer_fallback`, but the Rust constructor argument was missing. Restoring its `Option<bool>` argument aligns the existing API and setter; it does not add a new fallback policy.

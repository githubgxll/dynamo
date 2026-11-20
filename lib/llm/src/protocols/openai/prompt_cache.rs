// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prompt-cache TTL admission for the customer-facing protocols.
//!
//! One strict parser shared by Chat Completions (`prompt_cache_options`),
//! Responses (`prompt_cache_options`) and Anthropic Messages
//! (`cache_control`). The public contract (AGDC2-73 §1) is:
//!
//! - explicit TTLs match `^[1-9][0-9]*(s|m|h)$` and convert to whole seconds;
//!   `0s`, `05m`, `30M`, decimals and bare numbers are rejected with a 400
//!   that names the original protocol field;
//! - the customer entry enforces a per-model upper bound, defaulting to the
//!   conservative P1 value of 3600s and overridable per model. 60m and 1h
//!   are the same duration; the price tier never rewrites the applied
//!   seconds;
//! - a Messages marker (`cache_control` with `type: "ephemeral"`) that omits
//!   `ttl` defaults to 300s; `type: "readonly"` (or any unknown type) is a
//!   400 naming `cache_control.type`;
//! - multiple markers in one Messages request take the maximum TTL;
//! - requests without any marker/options leave the internal TTL unset: the
//!   engine then treats the prefix best-effort and no protection or premium
//!   is claimed.
//!
//! Internal wire name is `cache_retention_ttl_s` (whole seconds, `None` =
//! unset), matching SGLang's `GenerateReqInput.cache_retention_ttl_s`. The
//! value travels on the trusted, server-owned request field of the same
//! name — never through the client-writable `chat_template_args`.

use dynamo_protocols::types::anthropic::{CacheControl, CacheControlType};

/// Default maximum explicit TTL (P1 bound). Per-model configuration can
/// lower it; no model may exceed it without an explicit product decision.
pub const DEFAULT_MAX_TTL_SECONDS: u64 = 3600;
/// TTL applied when a Messages marker omits `ttl` (5m price tier).
pub const DEFAULT_MARKER_TTL_SECONDS: u64 = 300;

/// Process-wide bound from `DYN_PROMPT_CACHE_MAX_TTL_S` (whole seconds), read
/// once at first use. An unset, non-numeric, zero, or negative value means
/// "not configured": the built-in default applies. A configured value can
/// only lower the built-in default — a higher value is clamped so no
/// deployment can widen the bound past [`DEFAULT_MAX_TTL_SECONDS`] through
/// the environment.
static PROCESS_MAX_TTL_SECONDS: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
    match std::env::var(dynamo_runtime::config::environment_names::llm::DYN_PROMPT_CACHE_MAX_TTL_S) {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(seconds) if seconds > 0 => seconds.min(DEFAULT_MAX_TTL_SECONDS),
            _ => {
                tracing::warn!(
                    value = %raw,
                    env = dynamo_runtime::config::environment_names::llm::DYN_PROMPT_CACHE_MAX_TTL_S,
                    "Invalid process-wide prompt-cache max TTL: expected positive integer seconds; using default"
                );
                DEFAULT_MAX_TTL_SECONDS
            }
        },
        Err(_) => DEFAULT_MAX_TTL_SECONDS,
    }
});

/// The effective admission bound for one request, in whole seconds.
///
/// Priority: the model's per-model `prompt_cache.max_ttl_s` runtime
/// configuration (worker-published, may tighten per model), else the
/// process-wide env override, else the built-in default. A `None`
/// `model_max_ttl` simply falls through to the process bound. The user
/// request never influences this value.
pub fn admission_max_ttl_seconds(model_max_ttl: Option<u64>) -> u64 {
    match model_max_ttl {
        Some(seconds) if seconds > 0 => seconds.min(*PROCESS_MAX_TTL_SECONDS),
        _ => *PROCESS_MAX_TTL_SECONDS,
    }
}

/// Parse a strict explicit TTL string into whole seconds.
///
/// Returns `None` for anything that is not `^[1-9][0-9]*(s|m|h)$` — including
/// `0s`, zero-padded values, decimals and bare numbers. The caller decides
/// which protocol field to name in the rejection.
pub fn parse_ttl_seconds(ttl: &str) -> Option<u64> {
    let (digits, unit) = ttl.split_at(ttl.len().checked_sub(1)?);
    if digits.is_empty()
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
        // A leading zero (`05m`) is a format error, not 5m.
        || digits.starts_with('0')
    {
        return None;
    }
    let multiplier = match unit {
        "s" => 1,
        "m" => 60,
        "h" => 3600,
        _ => return None,
    };
    digits
        .parse::<u64>()
        .ok()
        .and_then(|value| value.checked_mul(multiplier))
}

/// Check an already-parsed TTL against the effective model bound.
fn check_ttl_bound(seconds: u64, ttl: &str, field: &str, max_ttl: u64) -> Result<u64, String> {
    if seconds > max_ttl {
        return Err(format!(
            "`{field}` exceeds the {max_ttl}s model limit; got \"{ttl}\""
        ));
    }
    Ok(seconds)
}

/// Validate `prompt_cache_options` (Chat Completions and Responses shape).
///
/// `mode` must be absent or `"implicit"`; anything else (including
/// `"ephemeral"`) is rejected naming the `mode` field. `ttl` must parse
/// strictly and stay within `max_ttl` (the caller's per-model bound, at most
/// [`DEFAULT_MAX_TTL_SECONDS`]).
///
/// The outer `Option` separates "no options at all" (`None` → the caller
/// forwards no internal TTL) from "options accepted" (`Some`), whose inner
/// `Option<u64>` is `None` when the options object omitted `ttl`: Chat and
/// Responses have no Messages-style marker default, so such a request also
/// proceeds without an internal TTL.
pub fn prompt_cache_options_ttl(
    options: &serde_json::Value,
    max_ttl: u64,
) -> Result<Option<Option<u64>>, String> {
    let Some(fields) = options.as_object() else {
        return Err("`prompt_cache_options` must be an object".to_string());
    };
    match fields.get("mode") {
        None | Some(serde_json::Value::Null) => {}
        Some(serde_json::Value::String(mode)) if mode == "implicit" => {}
        Some(_) => {
            // `ephemeral` is not a mode; `explicit` is a reserved design
            // value that must not be treated as an accepted capability.
            return Err(
                "`prompt_cache_options.mode` must be \"implicit\" when present".to_string(),
            );
        }
    }
    let Some(ttl) = fields.get("ttl").filter(|value| !value.is_null()) else {
        // Options accepted without an explicit TTL. No marker semantics to
        // default here; the caller forwards no internal TTL.
        return Ok(Some(None));
    };
    let Some(ttl) = ttl.as_str() else {
        return Err(
            "`prompt_cache_options.ttl` must be a string like \"90s\", \"5m\" or \"1h\""
                .to_string(),
        );
    };
    let Some(seconds) = parse_ttl_seconds(ttl) else {
        return Err(format!(
            "`prompt_cache_options.ttl` must match `^[1-9][0-9]*(s|m|h)$`; got \"{ttl}\""
        ));
    };
    check_ttl_bound(seconds, ttl, "prompt_cache_options.ttl", max_ttl).map(|s| Some(Some(s)))
}

/// TTL seconds for one Messages `cache_control` marker.
///
/// `type` must be `ephemeral` — the upstream `CacheControlType::Unknown`
/// fallback means anything else (e.g. `readonly`) reached the typed struct
/// without matching, so it is rejected naming `cache_control.type`. A missing
/// `ttl` applies the 5m marker default.
fn cache_control_ttl(marker: &CacheControl, max_ttl: u64) -> Result<u64, String> {
    if marker.control_type != CacheControlType::Ephemeral {
        return Err("`cache_control.type` must be \"ephemeral\"".to_string());
    }
    let Some(ttl) = marker.ttl.as_deref() else {
        return Ok(DEFAULT_MARKER_TTL_SECONDS);
    };
    let Some(seconds) = parse_ttl_seconds(ttl) else {
        return Err(format!(
            "`cache_control.ttl` must match `^[1-9][0-9]*(s|m|h)$`; got \"{ttl}\""
        ));
    };
    check_ttl_bound(seconds, ttl, "cache_control.ttl", max_ttl)
}

/// Resolve the internal TTL for a Messages request.
///
/// Collects every `cache_control` marker (top-level, system, message content
/// blocks, tools). No markers → `Ok(None)` (best-effort, nothing claimed).
/// Any invalid marker rejects the whole request; otherwise the maximum TTL
/// wins (P1 does not implement per-breakpoint segmented protection).
///
/// `blocks` yields `(message_index, block_index, marker)` so rejections can
/// locate the offending marker precisely.
pub fn messages_request_ttl<'a>(
    top_level: Option<&'a CacheControl>,
    system: Option<&'a CacheControl>,
    blocks: impl Iterator<Item = (usize, usize, &'a CacheControl)>,
    tools: impl Iterator<Item = &'a CacheControl>,
    max_ttl: u64,
) -> Result<Option<u64>, String> {
    fn push_marker(
        location: &str,
        marker: &CacheControl,
        max_ttl: u64,
        candidates: &mut Vec<u64>,
    ) -> Result<(), String> {
        let seconds = match cache_control_ttl(marker, max_ttl) {
            Ok(seconds) => seconds,
            Err(error) => {
                // Errors already name the leaf field (`cache_control.type`,
                // `cache_control.ttl`). For the top-level marker that path is
                // the complete location — prefixing it with its own container
                // would produce `cache_control.cache_control.type`. Nested
                // markers (system/tools/blocks) prefix their real container.
                if location.is_empty() {
                    return Err(error);
                }
                let leaf = error.trim_start_matches('`');
                return Err(format!("`{location}.{leaf}"));
            }
        };
        candidates.push(seconds);
        Ok(())
    }

    let mut candidates = Vec::new();
    if let Some(marker) = top_level {
        // The top-level marker's own error path (`cache_control.*`) is the
        // complete location; no container prefix.
        push_marker("", marker, max_ttl, &mut candidates)?;
    }
    if let Some(marker) = system {
        push_marker(
            "system.cache_control",
            marker,
            max_ttl,
            &mut candidates,
        )?;
    }
    for (message_index, block_index, marker) in blocks {
        push_marker(
            &format!("messages[{message_index}].content[{block_index}].cache_control"),
            marker,
            max_ttl,
            &mut candidates,
        )?;
    }
    for marker in tools {
        push_marker(
            "tools.cache_control",
            marker,
            max_ttl,
            &mut candidates,
        )?;
    }
    Ok(candidates.into_iter().max())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const MAX: u64 = DEFAULT_MAX_TTL_SECONDS;

    fn marker(ttl: Option<&str>, control_type: &str) -> CacheControl {
        serde_json::from_value(json!({
            "type": control_type,
            "ttl": ttl,
        }))
        .unwrap()
    }

    #[test]
    fn strict_ttl_grammar() {
        assert_eq!(parse_ttl_seconds("1s"), Some(1));
        assert_eq!(parse_ttl_seconds("90s"), Some(90));
        assert_eq!(parse_ttl_seconds("5m"), Some(300));
        assert_eq!(parse_ttl_seconds("301s"), Some(301));
        assert_eq!(parse_ttl_seconds("30m"), Some(1800));
        assert_eq!(parse_ttl_seconds("60m"), Some(3600));
        assert_eq!(parse_ttl_seconds("1h"), Some(3600));
        for bad in ["0s", "05m", "30M", "1.5h", "300", "", "-5m", "5 m", "1H"] {
            assert_eq!(parse_ttl_seconds(bad), None, "{bad} must be rejected");
        }
    }

    #[test]
    fn prompt_cache_options_modes() {
        assert_eq!(
            prompt_cache_options_ttl(&json!({"mode": "implicit", "ttl": "5m"}), MAX),
            Ok(Some(Some(300)))
        );
        assert_eq!(
            prompt_cache_options_ttl(&json!({"ttl": "90s"}), MAX),
            Ok(Some(Some(90)))
        );
        assert_eq!(
            prompt_cache_options_ttl(&json!({"mode": "implicit"}), MAX),
            Ok(Some(None))
        );
        let err = prompt_cache_options_ttl(&json!({"mode": "ephemeral", "ttl": "5m"}), MAX)
            .unwrap_err();
        assert!(err.contains("prompt_cache_options.mode"), "{err}");
        let err = prompt_cache_options_ttl(&json!({"mode": "explicit"}), MAX).unwrap_err();
        assert!(err.contains("prompt_cache_options.mode"), "{err}");
    }

    #[test]
    fn prompt_cache_options_ttl_rejections() {
        let err = prompt_cache_options_ttl(&json!({"ttl": "30M"}), MAX).unwrap_err();
        assert!(err.contains("prompt_cache_options.ttl"), "{err}");
        assert!(err.contains("30M"), "{err}");
        let err = prompt_cache_options_ttl(&json!({"ttl": "3601s"}), MAX).unwrap_err();
        assert!(err.contains("prompt_cache_options.ttl"), "{err}");
        let err = prompt_cache_options_ttl(&json!({"ttl": 300}), MAX).unwrap_err();
        assert!(err.contains("prompt_cache_options.ttl"), "{err}");
        let err = prompt_cache_options_ttl(&json!({"ttl": "0s"}), MAX).unwrap_err();
        assert!(err.contains("prompt_cache_options.ttl"), "{err}");
        // Per-model bound: a stricter model rejects what the default accepts.
        let err = prompt_cache_options_ttl(&json!({"ttl": "30m"}), 600).unwrap_err();
        assert!(err.contains("600s"), "{err}");
        assert!(prompt_cache_options_ttl(&json!({"ttl": "5m"}), 600).is_ok());
    }

    #[test]
    fn marker_type_and_defaults() {
        assert_eq!(
            messages_request_ttl(
                Some(&marker(None, "ephemeral")),
                None,
                [].into_iter(),
                [].into_iter(),
                MAX
            ),
            Ok(Some(DEFAULT_MARKER_TTL_SECONDS))
        );
        let err = messages_request_ttl(
            Some(&marker(Some("5m"), "readonly")),
            None,
            [].into_iter(),
            [].into_iter(),
            MAX,
        )
        .unwrap_err();
        // Top-level marker errors carry the bare `cache_control.*` path —
        // no duplicated container prefix.
        assert!(
            err.starts_with("`cache_control.type`"),
            "top-level error must not duplicate the container: {err}"
        );
        assert!(
            !err.contains("cache_control.cache_control"),
            "duplicated prefix regression: {err}"
        );
        let err = messages_request_ttl(
            Some(&marker(Some("30M"), "ephemeral")),
            None,
            [].into_iter(),
            [].into_iter(),
            MAX,
        )
        .unwrap_err();
        assert!(err.contains("cache_control.ttl"), "{err}");
        assert!(
            !err.contains("cache_control.cache_control"),
            "duplicated prefix regression: {err}"
        );
        // Per-model bound applies to markers too.
        let err = messages_request_ttl(
            Some(&marker(Some("1h"), "ephemeral")),
            None,
            [].into_iter(),
            [].into_iter(),
            600,
        )
        .unwrap_err();
        assert!(err.contains("cache_control.ttl"), "{err}");
    }

    #[test]
    fn multiple_markers_take_max() {
        let small = marker(Some("5m"), "ephemeral");
        let large = marker(Some("30m"), "ephemeral");
        let bare = marker(None, "ephemeral");
        assert_eq!(
            messages_request_ttl(
                None,
                Some(&small),
                [(1, 0, &large)].into_iter(),
                [].into_iter(),
                MAX
            ),
            Ok(Some(1800))
        );
        assert_eq!(
            messages_request_ttl(
                None,
                None,
                [(0, 1, &bare), (2, 0, &small)].into_iter(),
                [].into_iter(),
                MAX
            ),
            Ok(Some(300))
        );
    }

    #[test]
    fn no_markers_means_unset() {
        assert_eq!(
            messages_request_ttl(None, None, [].into_iter(), [].into_iter(), MAX),
            Ok(None)
        );
    }

    #[test]
    fn admission_bound_resolution() {
        // No per-model config → process bound (default in a clean env).
        assert_eq!(
            admission_max_ttl_seconds(None),
            *PROCESS_MAX_TTL_SECONDS
        );
        // A valid per-model bound tightens the process bound...
        assert_eq!(admission_max_ttl_seconds(Some(600)), 600.min(*PROCESS_MAX_TTL_SECONDS));
        // ...but can never widen past it.
        assert_eq!(
            admission_max_ttl_seconds(Some(u64::MAX)),
            *PROCESS_MAX_TTL_SECONDS
        );
        // Invalid per-model values (zero/non-positive) fall back rather than
        // widen; the caller treats them as "not configured".
        assert_eq!(admission_max_ttl_seconds(Some(0)), *PROCESS_MAX_TTL_SECONDS);
    }
}

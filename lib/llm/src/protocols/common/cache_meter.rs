// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request cache-write metering extracted from the engine's prefill receipt.
//!
//! The engine (SGLang K3.1 worker) reports the *actual* per-request prefix
//! cache accounting on the earliest `engine_data`-bearing chunk, ahead of the
//! decode stream: `engine_data.kv_cache_write` carries the prompt token count,
//! the cached (prefix-hit) count, the page-aligned newly written token count
//! (W) and the post-precheck applied TTL (a, seconds). When the engine does
//! not report the receipt, no protection is claimed: no header, no write
//! accounting, no charge. Values are never synthesized from the request TTL.

use serde::{Deserialize, Serialize};

/// Upper bound of the public metering values: the four `msh-usage-*` HTTP
/// headers and every public usage field this receipt projects onto carry
/// `0..=2147483647` (the range agreed with the customer packages so both
/// sides parse as signed 32-bit). A receipt with any P/C/W value above it
/// is rejected as a whole — never clamped, wrapped or saturated into a
/// billable substitute. The applied TTL (A) is not a public metering value
/// and keeps its u32 seconds range.
pub const MSH_USAGE_VALUE_MAX: u32 = 2147483647;

/// Engine receipt shape. Every field is a required integer; anything else is
/// rejected so a malformed receipt can never fabricate a protection claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCacheWriteReceipt {
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub cache_write_tokens: u64,
    pub cache_write_ttl_s: u64,
}

/// The applied cache-write meter for one request, derived strictly from the
/// engine's prefill receipt. `prompt_tokens`/`cached_tokens` keep the real
/// receipt values so every protocol projects P, C and W from one consistent
/// source (a PD metadata-only chunk does not necessarily update
/// `completion_usage.cached_tokens`, so the decode-side usage must not be
/// allowed to masquerade as prefill accounting). `None` of this struct means
/// "no valid receipt": nothing may be reported then.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct CacheWriteMeter {
    /// Real prompt token count from the receipt (P).
    pub prompt_tokens: u32,
    /// Real prefix-hit token count from the receipt (C).
    pub cached_tokens: u32,
    /// Page-aligned tokens newly inserted by this request (W).
    pub write_tokens: u32,
    /// Post-precheck applied protection TTL in seconds (a). 0 = unprotected.
    pub applied_ttl_s: u32,
}

/// Price-band boundary: writes at or below 300 s bill at the 5m band,
/// above 300 s at the 1h band. The band never changes the protected seconds.
pub const CACHE_WRITE_5M_BAND_MAX_S: u32 = 300;

impl CacheWriteMeter {
    /// Extract the meter from `engine_data.kv_cache_write`, the prefill
    /// receipt location agreed with the transport slice. Returns `None` when
    /// the object is absent — or **invalid**: any missing/non-integer field,
    /// a public value beyond the `0..=2147483647` header range, or a
    /// breakdown where `cached + written > prompt` is rejected outright. A
    /// rejected receipt reports nothing: no headers, no usage claims, no
    /// clamped or saturated substitutes (an engine that overflows or
    /// contradicts itself must not silently become a billable number).
    pub fn from_engine_data(engine_data: Option<&serde_json::Value>) -> Option<Self> {
        Self::from_receipt(engine_data?.get("kv_cache_write")?)
    }

    /// The Python SGLang processor bypasses the Rust delta generator. Accept
    /// its raw receipt under the same validation rules, not caller-supplied
    /// accounting. Malformed receipts suppress metering, not generation.
    pub(crate) fn deserialize_prefill_receipt<'de, D>(
        deserializer: D,
    ) -> Result<Option<Self>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        Ok(Self::from_receipt(&value))
    }

    fn from_receipt(receipt: &serde_json::Value) -> Option<Self> {
        let receipt = <KvCacheWriteReceipt as Deserialize>::deserialize(receipt).ok()?;
        let prompt_tokens = u32::try_from(receipt.prompt_tokens).ok()?;
        let cached_tokens = u32::try_from(receipt.cached_tokens).ok()?;
        let write_tokens = u32::try_from(receipt.cache_write_tokens).ok()?;
        let applied_ttl_s = u32::try_from(receipt.cache_write_ttl_s).ok()?;
        // P/C/W are projected onto the public `msh-usage-*` headers and the
        // usage fields, whose fixed range is 0..=2147483647. Any value above
        // rejects the whole receipt: no header subset, no truncation. The
        // applied TTL keeps the full u32 seconds range (bounded by request
        // admission, not by this public wire range).
        // C, W and the uncached remainder all fit when P fits and C+W <= P.
        if prompt_tokens > MSH_USAGE_VALUE_MAX
            || u64::from(cached_tokens) + u64::from(write_tokens) > u64::from(prompt_tokens)
        {
            return None;
        }
        Some(Self {
            prompt_tokens,
            cached_tokens,
            write_tokens,
            applied_ttl_s,
        })
    }

    /// True when the engine actually claimed protection for this request.
    pub fn is_protected(&self) -> bool {
        self.applied_ttl_s > 0
    }

    /// Split the write tokens into the two price-band headers by the applied
    /// TTL: `(5m_value, 1h_value)`. When protection was claimed, exactly one
    /// band carries W — **including W=0**, which must surface as an explicit
    /// zero per the customer contract (a valid receipt reports measured
    /// zeros, they are never treated as missing). No protection (`a=0`) or
    /// no receipt at all (`None` upstream) emits nothing.
    pub fn msh_split(&self) -> (u32, u32) {
        if !self.is_protected() {
            return (0, 0);
        }
        if self.applied_ttl_s <= CACHE_WRITE_5M_BAND_MAX_S {
            (self.write_tokens, 0)
        } else {
            (0, self.write_tokens)
        }
    }

    /// The four confirmed initial headers for one protected request:
    /// `(prompt_tokens, cached_tokens, write_5m, write_1h)`, all derived
    /// from this one receipt so the P/C pair and the two band headers can
    /// never contradict the body usage. The group exists only when the
    /// receipt is valid **and** the engine claimed protection (`A > 0`):
    /// a missing, malformed or unprotected receipt emits no header at all
    /// — absence is never a zero and a measured zero is never dropped.
    pub fn msh_usage_headers(&self) -> Option<(u32, u32, u32, u32)> {
        if !self.is_protected() {
            return None;
        }
        let (write_5m, write_1h) = self.msh_split();
        Some((self.prompt_tokens, self.cached_tokens, write_5m, write_1h))
    }

    /// Project this receipt onto the Chat/Responses cache fields
    /// (`cached_tokens`, `cache_write_tokens`). Called only with a valid
    /// receipt; both values are explicit integers then, so a measured zero
    /// stays a zero instead of being dropped.
    pub fn openai_projection(&self) -> (u32, u32) {
        (self.cached_tokens, self.write_tokens)
    }

    /// Project this receipt onto the Anthropic Messages fields
    /// `(input_tokens, cache_read_input_tokens, cache_creation_input_tokens)`,
    /// applying Anthropic's non-overlapping accounting: `input_tokens` is the
    /// uncached remainder, reads and writes are the disjoint C and W of the
    /// same receipt, and the three always sum back to P.
    pub fn anthropic_projection(&self) -> (u32, u32, u32) {
        let input = self
            .prompt_tokens
            .saturating_sub(self.cached_tokens)
            .saturating_sub(self.write_tokens);
        (input, self.cached_tokens, self.write_tokens)
    }
}

/// Render an applied TTL in seconds as the customer-facing duration string
/// (`90s`, `5m`, `30m`, `1h`), the same shape the request accepted. The
/// customer validates that a 30m request is echoed as 30m, never rounded up
/// to 1h. Falls back to seconds for non-minute-friendly values.
pub fn applied_ttl_string(seconds: u32) -> String {
    match seconds {
        0 => "0s".to_string(),
        s if s < 60 => format!("{s}s"),
        s if s % 3600 == 0 => format!("{}h", s / 3600),
        s if s % 60 == 0 => format!("{}m", s / 60),
        s => format!("{s}s"),
    }
}

/// HTTP response header names for the two cache-write price bands.
pub const MSH_CACHE_WRITE_HEADER_5M: &str = "msh-usage-cache-write-tokens-5m";
pub const MSH_CACHE_WRITE_HEADER_1H: &str = "msh-usage-cache-write-tokens-1h";

/// HTTP response header for the public final input token count P — the same
/// value the final usage reports (Kimi uncounted scaffold deducted at the
/// producer), never the raw physical prompt length.
pub const MSH_PROMPT_TOKENS_HEADER: &str = "msh-usage-prompt-tokens";
/// HTTP response header for the real prefix-cache hit count C, before any
/// downstream hit-rate rewrite.
pub const MSH_CACHED_TOKENS_HEADER: &str = "msh-usage-cached-tokens";

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn engine_data(receipt: serde_json::Value) -> Option<serde_json::Value> {
        Some(json!({"kv_cache_write": receipt}))
    }

    #[test]
    fn extracts_meter_from_full_receipt() {
        let data = engine_data(json!({
            "prompt_tokens": 6080,
            "cached_tokens": 0,
            "cache_write_tokens": 6016,
            "cache_write_ttl_s": 300
        }));
        let meter = CacheWriteMeter::from_engine_data(data.as_ref()).unwrap();
        assert_eq!(meter.write_tokens, 6016);
        assert_eq!(meter.applied_ttl_s, 300);
        assert_eq!(meter.msh_split(), (6016, 0));
    }

    #[test]
    fn python_processor_receipt_drives_wire_usage_without_leaking_internal_fields() {
        use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        let mut frame = json!({
            "id": "r", "object": "chat.completion.chunk", "created": 0,
            "model": "m", "choices": [],
            "usage": {"prompt_tokens": 130, "completion_tokens": 1, "total_tokens": 131},
            "prefill_cache_write": {"prompt_tokens": 130, "cached_tokens": 64,
                "cache_write_tokens": 64, "cache_write_ttl_s": 1800}
        });
        let parsed: NvCreateChatCompletionStreamResponse =
            serde_json::from_value(frame.clone()).unwrap();
        assert_eq!(parsed.cache_meter.unwrap().msh_split(), (0, 64));
        let public = serde_json::to_value(parsed).unwrap();
        assert_eq!(public["usage"]["prompt_tokens_details"]["cache_write_tokens"], 64);
        assert!(public.get("prefill_cache_write").is_none());
        frame["prefill_cache_write"]["cached_tokens"] = json!(u32::MAX);
        frame["prefill_cache_write"]["prompt_tokens"] = json!(u32::MAX);
        let invalid: NvCreateChatCompletionStreamResponse = serde_json::from_value(frame).unwrap();
        assert!(invalid.cache_meter.is_none());
    }

    #[test]
    fn long_band_routes_to_1h_header() {
        // 30m keeps 1800 protected seconds and lands in the 1h price band.
        let data = engine_data(json!({
            "prompt_tokens": 6080,
            "cached_tokens": 512,
            "cache_write_tokens": 5504,
            "cache_write_ttl_s": 1800
        }));
        let meter = CacheWriteMeter::from_engine_data(data.as_ref()).unwrap();
        assert!(meter.is_protected());
        assert_eq!(meter.msh_split(), (0, 5504));
    }

    #[test]
    fn boundary_301s_is_long_band() {
        let data = engine_data(json!({
            "prompt_tokens": 10,
            "cached_tokens": 0,
            "cache_write_tokens": 8,
            "cache_write_ttl_s": 301
        }));
        let meter = CacheWriteMeter::from_engine_data(data.as_ref()).unwrap();
        assert_eq!(meter.msh_split(), (0, 8));
    }

    #[test]
    fn zero_ttl_is_unprotected() {
        // Precheck downgrade: a=0 → no headers, no write accounting claim.
        let data = engine_data(json!({
            "prompt_tokens": 10,
            "cached_tokens": 0,
            "cache_write_tokens": 8,
            "cache_write_ttl_s": 0
        }));
        let meter = CacheWriteMeter::from_engine_data(data.as_ref()).unwrap();
        assert!(!meter.is_protected());
        assert_eq!(meter.msh_split(), (0, 0));
    }

    #[test]
    fn missing_or_malformed_receipt_is_none() {
        assert!(CacheWriteMeter::from_engine_data(None).is_none());
        assert!(CacheWriteMeter::from_engine_data(Some(&json!({}))).is_none());
        // Unrelated keys must not be mistaken for a receipt.
        let hit = json!({"kv_cache_hit": {"prompt_tokens": 9, "reused_tokens": 9}});
        assert!(CacheWriteMeter::from_engine_data(Some(&hit)).is_none());
        // Missing any required integer field rejects the whole object.
        let partial = engine_data(json!({
            "prompt_tokens": 10,
            "cached_tokens": 0,
            "cache_write_tokens": 8
        }));
        assert!(CacheWriteMeter::from_engine_data(partial.as_ref()).is_none());
        // Non-integer values are rejected, never coerced.
        let stringy = engine_data(json!({
            "prompt_tokens": 10,
            "cached_tokens": 0,
            "cache_write_tokens": "8",
            "cache_write_ttl_s": 300
        }));
        assert!(CacheWriteMeter::from_engine_data(stringy.as_ref()).is_none());
    }

    #[test]
    fn overflow_is_rejected_not_saturated() {
        // A receipt beyond the u32 wire formats is invalid, not a billable
        // u32::MAX; saturating would fabricate a charge.
        let data = engine_data(json!({
            "prompt_tokens": u64::MAX,
            "cached_tokens": 0,
            "cache_write_tokens": u64::MAX,
            "cache_write_ttl_s": u64::MAX
        }));
        assert!(CacheWriteMeter::from_engine_data(data.as_ref()).is_none());
    }

    #[test]
    fn inconsistent_breakdown_is_rejected() {
        // C + W must never exceed P; a contradicting engine must not become a
        // clamped charge.
        let data = engine_data(json!({
            "prompt_tokens": 10,
            "cached_tokens": 6,
            "cache_write_tokens": 8,
            "cache_write_ttl_s": 300
        }));
        assert!(CacheWriteMeter::from_engine_data(data.as_ref()).is_none());
    }

    #[test]
    fn exactly_prompt_sized_breakdown_is_accepted() {
        let data = engine_data(json!({
            "prompt_tokens": 10,
            "cached_tokens": 6,
            "cache_write_tokens": 4,
            "cache_write_ttl_s": 300
        }));
        let meter = CacheWriteMeter::from_engine_data(data.as_ref()).unwrap();
        assert_eq!(meter.write_tokens, 4);
        assert_eq!(meter.msh_split(), (4, 0));
    }

    #[test]
    fn protocol_projections_agree_on_one_receipt() {
        // One receipt → all three protocol projections plus the headers must
        // never contradict each other. Second-round read: C large, W small.
        let data = engine_data(json!({
            "prompt_tokens": 6080,
            "cached_tokens": 5504,
            "cache_write_tokens": 576,
            "cache_write_ttl_s": 300
        }));
        let meter = CacheWriteMeter::from_engine_data(data.as_ref()).unwrap();

        // Chat / Responses: same (C, W) pair.
        assert_eq!(meter.openai_projection(), (5504, 576));

        // Messages: non-overlapping accounting, three parts sum back to P.
        let (input, read, creation) = meter.anthropic_projection();
        assert_eq!((input, read, creation), (0, 5504, 576));
        assert_eq!(input + read + creation, meter.prompt_tokens);

        // Headers carry W once, in the band the applied TTL selects.
        assert_eq!(meter.msh_split(), (576, 0));

        // Cold write (C=0) and pure read (W=0) both stay explicit zeros.
        let cold = engine_data(json!({
            "prompt_tokens": 6080,
            "cached_tokens": 0,
            "cache_write_tokens": 6016,
            "cache_write_ttl_s": 1800
        }));
        let cold = CacheWriteMeter::from_engine_data(cold.as_ref()).unwrap();
        assert_eq!(cold.openai_projection(), (0, 6016));
        assert_eq!(cold.anthropic_projection(), (64, 0, 6016));
        assert_eq!(cold.msh_split(), (0, 6016));

        let read_only = engine_data(json!({
            "prompt_tokens": 6080,
            "cached_tokens": 6080,
            "cache_write_tokens": 0,
            "cache_write_ttl_s": 300
        }));
        let read_only = CacheWriteMeter::from_engine_data(read_only.as_ref()).unwrap();
        // Valid receipt with W=0: explicit zero in usage, explicit zero in
        // the band headers (protection claimed), never a dropped field.
        assert_eq!(read_only.openai_projection(), (6080, 0));
        assert_eq!(read_only.anthropic_projection(), (0, 6080, 0));
        assert_eq!(read_only.msh_split(), (0, 0));
        // The 5m band carries the measured zero explicitly: the split must
        // be (0, 0) *because W=0*, not because protection is absent.
        assert!(read_only.is_protected());
    }

    #[test]
    fn four_header_group_is_all_or_nothing_by_protection() {
        // T01/T02: a protected receipt yields the full P/C/band group from
        // one source; the band split never leaks into the P/C pair.
        let five_m = engine_data(json!({
            "prompt_tokens": 1000, "cached_tokens": 0,
            "cache_write_tokens": 960, "cache_write_ttl_s": 300
        }));
        let meter = CacheWriteMeter::from_engine_data(five_m.as_ref()).unwrap();
        assert_eq!(meter.msh_usage_headers(), Some((1000, 0, 960, 0)));

        let one_h = engine_data(json!({
            "prompt_tokens": 1000, "cached_tokens": 600,
            "cache_write_tokens": 384, "cache_write_ttl_s": 3600
        }));
        let meter = CacheWriteMeter::from_engine_data(one_h.as_ref()).unwrap();
        assert_eq!(meter.msh_usage_headers(), Some((1000, 600, 0, 384)));

        // T03/T06: protected receipt with W=0 and/or C=0 keeps the group
        // with confirmed zeros — a zero is never dropped to "absent".
        let zeros = engine_data(json!({
            "prompt_tokens": 1000, "cached_tokens": 960,
            "cache_write_tokens": 0, "cache_write_ttl_s": 300
        }));
        let meter = CacheWriteMeter::from_engine_data(zeros.as_ref()).unwrap();
        assert_eq!(meter.msh_usage_headers(), Some((1000, 960, 0, 0)));

        let both_zero = engine_data(json!({
            "prompt_tokens": 1000, "cached_tokens": 0,
            "cache_write_tokens": 0, "cache_write_ttl_s": 300
        }));
        let meter = CacheWriteMeter::from_engine_data(both_zero.as_ref()).unwrap();
        assert_eq!(meter.msh_usage_headers(), Some((1000, 0, 0, 0)));

        // T11: valid receipt but A=0 (precheck downgrade) — no header group
        // at all, even though real C/W exist for the body.
        let unprotected = engine_data(json!({
            "prompt_tokens": 1000, "cached_tokens": 600,
            "cache_write_tokens": 384, "cache_write_ttl_s": 0
        }));
        let meter = CacheWriteMeter::from_engine_data(unprotected.as_ref()).unwrap();
        assert_eq!(meter.openai_projection(), (600, 384));
        assert_eq!(meter.msh_usage_headers(), None);
    }

    #[test]
    fn public_i32_boundary_values_are_representable() {
        // T10: 2147483646 and 2147483647 are valid public values and must be
        // surfaced exactly (T02-style 1h band here).
        let max = engine_data(json!({
            "prompt_tokens": 2147483647, "cached_tokens": 2147483646,
            "cache_write_tokens": 1, "cache_write_ttl_s": 3600
        }));
        let meter = CacheWriteMeter::from_engine_data(max.as_ref()).unwrap();
        assert_eq!(meter.msh_usage_headers(), Some((2147483647, 2147483646, 0, 1)));
        assert_eq!(meter.anthropic_projection(), (1, 2147483646, 1));
    }

    #[test]
    fn public_i32_overflow_rejects_whole_receipt() {
        // T10: 2147483648 and beyond must not output a confirmed receipt:
        // no clamping to i32::MAX, no wrapping, no partial header group.
        for over in [u64::from(MSH_USAGE_VALUE_MAX) + 1, u64::from(u32::MAX), u64::MAX] {
            let over_p = engine_data(json!({
                "prompt_tokens": over, "cached_tokens": 0,
                "cache_write_tokens": 0, "cache_write_ttl_s": 300
            }));
            assert!(
                CacheWriteMeter::from_engine_data(over_p.as_ref()).is_none(),
                "P={over} must reject the whole receipt"
            );
            let over_c = engine_data(json!({
                "prompt_tokens": 10, "cached_tokens": over,
                "cache_write_tokens": 10, "cache_write_ttl_s": 300
            }));
            assert!(
                CacheWriteMeter::from_engine_data(over_c.as_ref()).is_none(),
                "C={over} must reject the whole receipt"
            );
            let over_w = engine_data(json!({
                "prompt_tokens": 10, "cached_tokens": 0,
                "cache_write_tokens": over, "cache_write_ttl_s": 300
            }));
            assert!(
                CacheWriteMeter::from_engine_data(over_w.as_ref()).is_none(),
                "W={over} must reject the whole receipt"
            );
        }
        // The applied TTL is NOT part of the public i32 range: a large-but-u32
        // TTL keeps the receipt billable in the band it selects.
        let big_ttl = engine_data(json!({
            "prompt_tokens": 10, "cached_tokens": 0,
            "cache_write_tokens": 8, "cache_write_ttl_s": 2147483648u64
        }));
        let meter = CacheWriteMeter::from_engine_data(big_ttl.as_ref()).unwrap();
        assert_eq!(meter.applied_ttl_s, 2147483648);
        assert_eq!(meter.msh_usage_headers(), Some((10, 0, 0, 8)));
    }
}

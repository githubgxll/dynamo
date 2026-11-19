// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Map, Value};

/// Non-boolean stream values verified against GLM-5.3 on 2026-11-19.
/// Other representations retain typed decoding errors instead of guessing truthiness.
pub(crate) fn compatible_stream(value: &Value) -> Option<bool> {
    match value {
        Value::String(text) if text == "true" => Some(true),
        Value::Number(number) => match number.as_i64() {
            Some(0) => Some(false),
            Some(1) => Some(true),
            _ => None,
        },
        _ => None,
    }
}

/// GLM-5.3 Chat Completions HTTP fields, verified against the official API on 2026-11-19.
/// Only omit absent optional values; requested scores, real refusals and tools survive.
pub(super) fn project_response(response: &mut Map<String, Value>, request_id: &str) {
    response.insert("request_id".into(), Value::String(request_id.into()));
    if let Some(choices) = response.get_mut("choices").and_then(Value::as_array_mut) {
        for choice in choices {
            if choice.get("logprobs").is_some_and(Value::is_null) {
                choice.as_object_mut().unwrap().remove("logprobs");
            }
            if let Some(message) = choice.get_mut("message").and_then(Value::as_object_mut) {
                if message.get("refusal").is_some_and(Value::is_null) {
                    message.remove("refusal");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stream_compatibility_only_converts_verified_values() {
        for (value, expected) in [
            (json!("true"), Some(true)),
            (json!(1), Some(true)),
            (json!(0), Some(false)),
            (json!("false"), None),
            (json!(2), None),
            (json!(1.0), None),
            (json!(null), None),
            (json!(true), None),
            (json!([]), None),
        ] {
            assert_eq!(compatible_stream(&value), expected, "{value}");
        }
    }

    #[test]
    fn official_fields_preserve_real_scores_refusals_and_tools() {
        let mut value = json!({"choices":[
            {"logprobs":null,"message":{"content":"hi","refusal":null}},
            {"logprobs":{"content":[]},"message":{"refusal":"blocked","tool_calls":[]}}
        ]});
        project_response(value.as_object_mut().unwrap(), "frontend-request");
        assert_eq!(value["request_id"], "frontend-request");
        assert!(value["choices"][0].get("logprobs").is_none());
        assert!(value["choices"][0]["message"].get("refusal").is_none());
        assert_eq!(value["choices"][1]["logprobs"], json!({"content":[]}));
        assert_eq!(value["choices"][1]["message"]["refusal"], "blocked");
        assert_eq!(value["choices"][1]["message"]["tool_calls"], json!([]));
    }
}

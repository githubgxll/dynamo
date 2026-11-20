// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Map, Value};

pub(super) fn validate_user_id(value: Option<&Value>) -> Result<(), &'static str> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(id)) if (6..=128).contains(&id.chars().count()) => Ok(()),
        _ => Err("GLM-5.3 user_id must be a string of 6 to 128 characters"),
    }
}

pub(super) fn validate_stop_count(count: usize) -> Result<(), &'static str> {
    if count > 4 {
        Err("GLM-5.3 allows at most 4 stop words")
    } else {
        Ok(())
    }
}

/// HTTP correlation only: never use a client ID to key internal request state.
pub(super) fn client_request_id(value: Option<&Value>) -> Result<Option<&str>, &'static str> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id)) if (6..=64).contains(&id.chars().count()) => Ok(Some(id)),
        _ => Err("GLM-5.3 request_id must be a string of 6 to 64 characters"),
    }
}

/// Acceptance follows the recorded GLM-5.3 API probes. The safe canonical
/// values are our frontend policy, not a claim about the provider's internals.
pub(super) fn compatible_temperature(value: Option<f32>) -> Option<f32> {
    value.map(|v| if v.is_finite() && v < 0.0 { 0.01 } else { v })
}

pub(super) fn compatible_presence_penalty(value: Option<f32>, minimum: f32) -> Option<f32> {
    value.map(|v| if v.is_finite() && v < minimum { 0.0 } else { v })
}

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
    fn extra_field_and_stop_limits_follow_glm53_contract() {
        for length in [6, 128] {
            assert!(validate_user_id(Some(&json!("u".repeat(length)))).is_ok());
        }
        for value in [json!("u".repeat(5)), json!("u".repeat(129)), json!(42)] {
            assert!(validate_user_id(Some(&value)).is_err());
        }
        assert!(validate_user_id(None).is_ok());
        assert!(validate_stop_count(4).is_ok());
        assert!(validate_stop_count(5).is_err());
        assert!(validate_stop_count(17).is_err());
    }

    #[test]
    fn client_ids_validate_and_echo_without_replacing_internal_ids() {
        for length in [6, 64] {
            let value = json!("a".repeat(length));
            let id = client_request_id(Some(&value)).unwrap().unwrap();
            let mut response = json!({"id":"server-owned-id"}).as_object().unwrap().clone();
            project_response(&mut response, id);
            assert_eq!(response["request_id"], value);
            assert_eq!(response["id"], "server-owned-id");
        }
        for value in [
            json!("a".repeat(5)),
            json!("a".repeat(65)),
            json!(123),
            json!({}),
        ] {
            assert!(client_request_id(Some(&value)).is_err());
        }
        assert_eq!(client_request_id(None).unwrap(), None);
        assert_eq!(client_request_id(Some(&Value::Null)).unwrap(), None);
    }

    #[test]
    fn negative_sampling_values_become_safe_backend_values() {
        assert_eq!(compatible_temperature(Some(-1.0)), Some(0.01));
        assert_eq!(compatible_presence_penalty(Some(-5.0), -2.0), Some(0.0));
        for value in [None, Some(0.0), Some(0.7), Some(3.0), Some(f32::INFINITY)] {
            assert_eq!(compatible_temperature(value), value);
        }
        for value in [
            None,
            Some(-2.0),
            Some(-1.0),
            Some(0.0),
            Some(3.0),
            Some(f32::NEG_INFINITY),
        ] {
            assert_eq!(compatible_presence_penalty(value, -2.0), value);
        }
    }

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

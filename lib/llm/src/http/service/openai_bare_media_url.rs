// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility for bare image URL strings in chat message content parts.

use serde::de::DeserializeOwned;
use serde_json::Value;

/// Called only after strict chat request parsing and existing JSON repairs fail.
/// Keep the canonical fast path allocation-free beyond typed deserialization.
pub(super) fn parse_compatible<T: DeserializeOwned>(body: &[u8]) -> Option<T> {
    let mut value = serde_json::from_slice::<UniqueValue>(body).ok()?.0;
    if !normalize_bare_image_urls(&mut value) {
        return None;
    }
    serde_json::from_value(value).ok()
}

// A Value round trip must not hide duplicate fields that typed serde rejects.
// Construct the compatibility tree while rejecting duplicate object keys.
struct UniqueValue(Value);

impl<'de> serde::Deserialize<'de> for UniqueValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueValue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("JSON without duplicate object keys")
            }

            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Null))
            }

            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Bool(value)))
            }

            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }

            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }

            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                self.visit_string(value.to_owned())
            }

            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::String(value)))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element::<UniqueValue>()? {
                    values.push(value.0);
                }
                Ok(UniqueValue(Value::Array(values)))
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(serde::de::Error::custom("duplicate JSON object key"));
                    }
                    values.insert(key, map.next_value::<UniqueValue>()?.0);
                }
                Ok(UniqueValue(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

fn normalize_bare_image_urls(body: &mut Value) -> bool {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut changed = false;
    for message in messages {
        if !matches!(message.get("role").and_then(Value::as_str), Some("user" | "tool")) {
            continue;
        }
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in parts {
            if part.get("type").and_then(Value::as_str) != Some("image_url") {
                continue;
            }
            if let Some(url) = part.get_mut("image_url").filter(|url| url.is_string()) {
                let string = url.take();
                *url = serde_json::json!({"url": string});
                changed = true;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::CreateChatCompletionRequest;
    use serde_json::json;

    #[test]
    fn tool_images_parse_without_losing_order_or_correlation() {
        let value = json!({"model":"kimi-k3.1","messages":[
            {"role":"tool","tool_call_id":"call-1","content":[
                {"type":"text","text":"before"},
                {"type":"image_url","image_url":"https://example.com/tool.png"},
                {"type":"image_url","image_url":{"url":"https://example.com/detail.png","detail":"low"}},
                {"type":"text","text":"after"}
            ]},
            {"role":"user","content":[{"type":"image_url","image_url":"https://example.com/user.png"}]}
        ]});
        let request: CreateChatCompletionRequest =
            parse_compatible(&serde_json::to_vec(&value).unwrap()).unwrap();
        let wire = serde_json::to_value(request).unwrap();
        let tool = &wire["messages"][0];
        assert_eq!(tool["role"], "tool");
        assert_eq!(tool["tool_call_id"], "call-1");
        assert_eq!(tool["content"][0]["text"], "before");
        assert_eq!(tool["content"][1]["image_url"]["url"], "https://example.com/tool.png");
        assert_eq!(tool["content"][2]["image_url"]["detail"], "low");
        assert_eq!(tool["content"][3]["text"], "after");
        assert_eq!(wire["messages"][1]["content"][0]["image_url"]["url"], "https://example.com/user.png");
    }

    #[test]
    fn normalization_never_changes_schemas_arguments_or_text_results() {
        let mut value = json!({"messages":[
            {"role":"tool","tool_call_id":"call-1","content":[
                {"type":"image_url","image_url":"https://example.com/tool.png"}
            ]},
            {"role":"assistant","tool_calls":[{"id":"call-2","type":"function","function":{
                "name":"f","arguments":"{\"image_url\":\"leave alone\"}"
            }}]},
            {"role":"tool","tool_call_id":"call-2","content":"{\"image_url\":\"leave alone\"}"}
        ],"tools":[{"type":"function","function":{"name":"f","parameters":{
            "type":"object","properties":{"image_url":{"type":"string"}}
        }}}]});
        let original = value.clone();
        assert!(normalize_bare_image_urls(&mut value));
        assert_eq!(value["messages"][1], original["messages"][1]);
        assert_eq!(value["messages"][2], original["messages"][2]);
        assert_eq!(value["tools"], original["tools"]);
    }

    #[test]
    fn canonical_and_text_only_tool_requests_keep_the_strict_path() {
        for content in [
            json!("tool output"),
            json!([{"type":"text","text":"tool output"}]),
            json!([{"type":"image_url","image_url":{"url":"https://example.com/a.png","detail":"high"}}]),
        ] {
            let value = json!({"model":"kimi-k3.1","messages":[
                {"role":"tool","tool_call_id":"call-1","content":content}
            ]});
            let request: CreateChatCompletionRequest = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(serde_json::to_value(request).unwrap()["messages"][0]["content"], content);
            let mut normalized = value.clone();
            assert!(!normalize_bare_image_urls(&mut normalized));
            assert_eq!(normalized, value);
        }
    }

    #[test]
    fn invalid_media_and_unrelated_type_errors_are_not_accepted() {
        for image_url in [json!(42), json!(["https://example.com/a.png"]), json!("not a URL")] {
            let value = json!({"model":"kimi-k3.1","messages":[
                {"role":"tool","tool_call_id":"call-1","content":[{"type":"image_url","image_url":image_url}]}
            ]});
            assert!(parse_compatible::<CreateChatCompletionRequest>(&serde_json::to_vec(&value).unwrap()).is_none());
        }
        let value = json!({"model":"kimi-k3.1","max_tokens":"invalid","messages":[
            {"role":"tool","tool_call_id":"call-1","content":[{"type":"image_url","image_url":"https://example.com/a.png"}]}
        ]});
        assert!(parse_compatible::<CreateChatCompletionRequest>(&serde_json::to_vec(&value).unwrap()).is_none());
        for body in [
            r#"{"model":"m","messages":[{"role":"tool","tool_call_id":"c","content":[{"type":"image_url","image_url":"https://example.com/a.png"}]}],"model":"duplicate"}"#,
            r#"{"model":"m","messages":[{"role":"tool","tool_call_id":"c","content":[{"type":"image_url","image_url":"https://example.com/a.png","image_url":"https://example.com/b.png"}]}]}"#,
        ] {
            assert!(parse_compatible::<CreateChatCompletionRequest>(body.as_bytes()).is_none());
        }
    }
}

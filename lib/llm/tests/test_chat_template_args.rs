// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::engines::ValidateRequest;
use serde_json::json;

use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

#[test]
fn test_both_fields_fails() {
    // Test that when both are present, serde will fail with duplicate field error
    // This test documents that behavior
    let json_with_both = r#"{
        "model": "test-model",
        "messages": [],
        "chat_template_args": {
            "enable_thinking": true
        },
        "chat_template_kwargs": {
            "enable_thinking": false
        }
    }"#;

    // This will fail with duplicate field error
    let result: Result<NvCreateChatCompletionRequest, _> = serde_json::from_str(json_with_both);
    assert!(result.is_err());
}

#[test]
fn test_chat_template_kwargs_alias() {
    // Test that chat_template_kwargs is accepted as an alias for chat_template_args
    let json_with_kwargs = r#"{
        "model": "test-model",
        "messages": [],
        "chat_template_kwargs": {
            "enable_thinking": false
        }
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_with_kwargs).unwrap();
    assert!(request.chat_template_args.is_some());
    assert_eq!(
        request.chat_template_args.unwrap().get("enable_thinking"),
        Some(&serde_json::json!(false))
    );
}

#[test]
fn test_chat_template_args() {
    // Test that chat_template_args still works as the primary field name
    let json_with_args = r#"{
        "model": "test-model",
        "messages": [],
        "chat_template_args": {
            "enable_thinking": true
        }
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_with_args).unwrap();
    assert!(request.chat_template_args.is_some());
    assert_eq!(
        request.chat_template_args.unwrap().get("enable_thinking"),
        Some(&serde_json::json!(true))
    );
}

// TODO: Add template rendering test that verifies chat_template_args/chat_template_kwargs
// values are actually passed to the Jinja template context during rendering.
// This would require setting up PromptFormatter with ChatTemplate/ContextMixins.

fn reasoning_request(controls: serde_json::Value) -> NvCreateChatCompletionRequest {
    let mut payload = json!({
        "model": "generic-model",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    payload
        .as_object_mut()
        .unwrap()
        .extend(controls.as_object().unwrap().clone());
    serde_json::from_value(payload).unwrap()
}

#[test]
fn test_reasoning_zcode_payload_preserves_unrelated_fields_and_is_idempotent() {
    for (model, stream) in [
        ("zai-org/GLM-5.3", false),
        ("zai-org/GLM-5.3", true),
        ("generic-model", false),
        ("generic-model", true),
    ] {
        let mut request = reasoning_request(json!({
            "model": model,
            "stream": stream,
            "thinking": {"type": "enabled"},
            "enable_thinking": true,
            "reasoning_effort": "max",
            "reasoning": {"effort": "max"},
            "max_completion_tokens": 128000,
            "tool_choice": "auto",
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            "chat_template_kwargs": {"thinking": true, "enable_thinking": true, "thinking_mode": "thinking", "reasoning_effort": "max", "custom": {"nested": [1, null]}}
        }));
        let before = serde_json::to_value(&request).unwrap();
        request.normalize_reasoning_template_args().unwrap();
        assert!(request.unsupported_fields.is_empty());
        assert!(request.thinking.is_none());
        ValidateRequest::validate(&request).unwrap();
        let normalized = serde_json::to_value(&request).unwrap();
        for field in ["tools", "tool_choice", "max_completion_tokens", "stream"] {
            assert_eq!(normalized[field], before[field]);
        }
        assert_eq!(normalized["reasoning_effort"], "max");
        assert_eq!(
            normalized["chat_template_args"]["custom"],
            before["chat_template_args"]["custom"]
        );
        assert_eq!(normalized["chat_template_args"]["thinking_mode"], "enabled");
        assert_eq!(normalized["chat_template_args"]["reasoning_effort"], "max");
        request.normalize_reasoning_template_args().unwrap();
        assert_eq!(serde_json::to_value(&request).unwrap(), normalized);
    }
}

#[test]
fn test_reasoning_single_aliases_and_nested_controls() {
    for controls in [
        json!({"reasoning": {"effort": "max"}}),
        json!({"chat_template_kwargs": {"reasoning_effort": "max"}}),
    ] {
        let mut request = reasoning_request(controls);
        request.normalize_reasoning_template_args().unwrap();
        assert_eq!(
            serde_json::to_value(&request.inner.reasoning_effort).unwrap(),
            "max"
        );
        assert_eq!(
            request.chat_template_args.as_ref().unwrap()["reasoning_effort"],
            "max"
        );
        assert!(request.unsupported_fields.is_empty());
    }
    for (controls, enabled) in [
        (json!({"enable_thinking": false}), false),
        (json!({"enable_thinking": true}), true),
        (
            json!({"chat_template_kwargs": {"thinking_mode": "chat"}}),
            false,
        ),
        (
            json!({"chat_template_kwargs": {"thinking_mode": "thinking"}}),
            true,
        ),
        (
            json!({"thinking": false, "enable_thinking": false, "chat_template_kwargs": {"thinking_mode": "disabled", "thinking": false}}),
            false,
        ),
    ] {
        let mut request = reasoning_request(controls);
        request.normalize_reasoning_template_args().unwrap();
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args["thinking"], enabled);
        assert_eq!(args["enable_thinking"], enabled);
    }
    let mut adaptive = reasoning_request(
        json!({"thinking": {"type": "adaptive"}, "chat_template_kwargs": {"thinking_mode": "adaptive"}}),
    );
    adaptive.normalize_reasoning_template_args().unwrap();
    let args = adaptive.chat_template_args.as_ref().unwrap();
    assert_eq!(args["thinking_mode"], "adaptive");
    assert!(!args.contains_key("thinking"));
    assert!(!args.contains_key("enable_thinking"));
    adaptive.normalize_reasoning_template_args().unwrap();
}

#[test]
fn test_reasoning_conflicts_and_invalid_aliases_are_atomic() {
    for controls in [
        json!({"thinking": true, "enable_thinking": false}),
        json!({"thinking": {"type": "adaptive"}, "enable_thinking": true}),
        json!({"reasoning_effort": "high", "reasoning": {"effort": "max"}}),
        json!({"reasoning": {"effort": "max"}, "chat_template_kwargs": {"reasoning_effort": "low"}}),
        json!({"chat_template_kwargs": {"thinking": true, "enable_thinking": false}}),
        json!({"chat_template_kwargs": {"thinking": true, "thinking_mode": "chat"}}),
        json!({"enable_thinking": null}),
        json!({"enable_thinking": "true"}),
        json!({"enable_thinking": 1}),
        json!({"reasoning": null}),
        json!({"reasoning": "max"}),
        json!({"reasoning": []}),
        json!({"reasoning": {"effort": null}}),
        json!({"reasoning": {"effort": true}}),
        json!({"reasoning": {"effort": {"max": null}}}),
        json!({"chat_template_kwargs": {"reasoning_effort": {"max": null}}}),
        json!({"reasoning": {"effort": "unknown"}}),
        json!({"reasoning": {"effort": "max", "summary": "auto"}}),
        json!({"reasoning": {"budget": 1024}}),
        json!({"enable_thinking": true, "chat_template_kwargs": {"reasoning_effort": null}}),
        json!({"chat_template_kwargs": {"thinking_mode": "unknown"}}),
        json!({"model": "GLM-5.3", "enable_thinking": false}),
        json!({"model": "GLM-5.3", "reasoning": {"effort": "none"}}),
        json!({"model": "GLM-5.3", "chat_template_kwargs": {"reasoning_effort": "minimal"}}),
    ] {
        let mut request = reasoning_request(controls.clone());
        let before = serde_json::to_value(&request).unwrap();
        let aliases = request.unsupported_fields.clone();
        assert!(
            request.normalize_reasoning_template_args().is_err(),
            "{controls}"
        );
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            before,
            "{controls}"
        );
        assert_eq!(request.unsupported_fields, aliases, "{controls}");
    }
}

#[test]
fn test_reasoning_defaults_and_existing_model_policy() {
    for controls in [
        json!({}),
        json!({"chat_template_kwargs": {"custom": [1, null]}}),
    ] {
        let mut request = reasoning_request(controls);
        let before = serde_json::to_value(&request).unwrap();
        request.normalize_reasoning_template_args().unwrap();
        assert_eq!(serde_json::to_value(&request).unwrap(), before);
    }
    for effort in ["low", "medium", "high", "max"] {
        let mut request = reasoning_request(json!({
            "model": "GLM-5.3", "reasoning": {"effort": effort}
        }));
        request.normalize_reasoning_template_args().unwrap();
        assert_eq!(
            serde_json::to_value(&request.inner.reasoning_effort).unwrap(),
            effort
        );
    }
    let mut request = reasoning_request(json!({"reasoning": {}}));
    request.normalize_reasoning_template_args().unwrap();
    assert!(request.unsupported_fields.is_empty());
    assert!(request.chat_template_args.is_none());
}

#[test]
fn test_reasoning_only_consumes_supported_aliases() {
    let mut request = reasoning_request(
        json!({"enable_thinking": true, "reasoning": {}, "unknown_parameter": 7}),
    );
    request.normalize_reasoning_template_args().unwrap();
    assert_eq!(request.unsupported_fields.len(), 1);
    assert_eq!(request.unsupported_fields["unknown_parameter"], 7);
    assert!(ValidateRequest::validate(&request).is_err());
}

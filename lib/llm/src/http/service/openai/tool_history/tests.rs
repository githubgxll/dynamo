// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::protocols::openai::{
    chat_completions::{NvCreateChatCompletionRequest, ToolCallCompletion},
    validate::validate_messages_with_history_policy,
};
use serde_json::{Value, json};

const CLIENT_ERROR: &str = "Error: Tool 'unknown_tool' failed with ValidationError: tool_call_id Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]";

fn real_call() -> Value {
    json!({"id":"call_ok","type":"function","function":{"name":"bash","arguments":"\"command\":\"echo test\"}"}})
}

fn stub() -> Value {
    json!({"id":null,"type":"function","function":{"name":"","arguments":"{}"}})
}

fn history(with_stub: bool) -> NvCreateChatCompletionRequest {
    let mut calls = vec![real_call()];
    if with_stub {
        calls.insert(0, stub());
    }
    let body = json!({"model":"glm-5.3","messages":[
        {"role":"assistant","content":null,"tool_calls":calls},
        {"role":"tool","tool_call_id":PLACEHOLDER_ID,"content":CLIENT_ERROR},
        {"role":"tool","tool_call_id":"call_ok","content":"invalid arguments"}
    ]});
    super::super::parse_json_request("chat completions", &serde_json::to_vec(&body).unwrap())
        .unwrap()
}

#[test]
fn document_history_and_already_cleaned_history_retain_real_results() {
    for with_stub in [false, true] {
        let mut request = history(with_stub);
        assert!(validate_messages_with_history_policy(&request.inner.messages, true).is_err());
        repair_placeholder_results(&mut request.inner.messages);
        assert_eq!(request.inner.messages.len(), 2);
        validate_messages_with_history_policy(&request.inner.messages, true).unwrap();
        let value = serde_json::to_value(&request.inner.messages).unwrap();
        assert_eq!(value[0]["tool_calls"][0], real_call());
        assert_eq!(value[1]["tool_call_id"], "call_ok");
    }
}

#[test]
fn unrelated_or_declared_placeholder_results_are_preserved() {
    for variant in 0..3 {
        let mut request = history(false);
        match variant {
            0 => {
                if let ChatCompletionRequestMessage::Tool(tool) = &mut request.inner.messages[1] {
                    tool.content =
                        ChatCompletionRequestToolMessageContent::Text("normal result".into());
                }
            }
            1 => {
                if let ChatCompletionRequestMessage::Assistant(assistant) =
                    &mut request.inner.messages[0]
                {
                    let calls = assistant.tool_calls.as_mut().unwrap();
                    let mut declared = calls[0].clone();
                    declared.id = PLACEHOLDER_ID.into();
                    calls.push(declared);
                }
            }
            _ => request.inner.messages.insert(
                1,
                serde_json::from_value(json!({"role":"user","content":"new turn"})).unwrap(),
            ),
        }
        let before = serde_json::to_value(&request.inner.messages).unwrap();
        repair_placeholder_results(&mut request.inner.messages);
        assert_eq!(
            serde_json::to_value(&request.inner.messages).unwrap(),
            before
        );
    }
}

fn frame(choices: Value) -> NvCreateChatCompletionStreamResponse {
    serde_json::from_value(json!({
        "id":"chat_test","object":"chat.completion.chunk","created":1,"model":"glm-5.3",
        "choices":choices
    }))
    .unwrap()
}

fn delta(index: u32, calls: Value, finish: Value) -> Value {
    json!({"index":index,"delta":{"tool_calls":calls},"finish_reason":finish})
}

fn chunk(index: u32, id: Value, name: Value, arguments: Value) -> Value {
    json!({"index":index,"id":id,"type":"function","function":{"name":name,"arguments":arguments}})
}

#[test]
fn delayed_identity_filters_stub_and_remaps_completion_without_losing_arguments() {
    let mut guard = Guard::default();
    let mut first = frame(json!([delta(
        0,
        json!([
            chunk(0, Value::Null, json!(""), json!("{}")),
            chunk(1, Value::Null, json!(""), json!("{\"x\":"))
        ]),
        Value::Null
    )]));
    first.inner.choices[0].delta.reasoning_content = Some("思考".into());
    first.tool_call_completion.push(ToolCallCompletion {
        choice_index: 0,
        tool_index: 1,
        complete: true,
    });
    guard.apply(&mut first).unwrap();
    assert!(first.inner.choices[0].delta.tool_calls.is_none());
    assert_eq!(
        first.inner.choices[0].delta.reasoning_content.as_deref(),
        Some("思考")
    );
    assert!(first.tool_call_completion.is_empty());
    let mut second = frame(json!([delta(
        0,
        json!([chunk(1, json!("call_ok"), json!("echo"), json!("1}"))]),
        Value::Null
    )]));
    guard.apply(&mut second).unwrap();
    let calls = second.inner.choices[0].delta.tool_calls.as_ref().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].index, 0);
    assert_eq!(
        calls[0].function.as_ref().unwrap().arguments.as_deref(),
        Some("{\"x\":1}")
    );
    assert_eq!(second.tool_call_completion[0].tool_index, 0);
    let mut final_frame = frame(json!([delta(0, Value::Null, json!("tool_calls"))]));
    guard.apply(&mut final_frame).unwrap();
    guard.finish().unwrap();
}

#[test]
fn parallel_choices_continuations_and_metadata_are_preserved() {
    let mut guard = Guard::default();
    let mut initial = frame(json!([
        delta(
            0,
            json!([
                chunk(7, json!("a"), json!("get_"), json!("{")),
                chunk(8, json!("b"), json!("other"), json!("{}"))
            ]),
            Value::Null
        ),
        delta(
            1,
            json!([chunk(5, json!("c"), json!("echo"), json!("{}"))]),
            Value::Null
        )
    ]));
    guard.apply(&mut initial).unwrap();
    assert_eq!(
        initial.inner.choices[0].delta.tool_calls.as_ref().unwrap()[1].index,
        1
    );
    assert_eq!(
        initial.inner.choices[1].delta.tool_calls.as_ref().unwrap()[0].index,
        0
    );
    let mut continuation = frame(json!([delta(
        0,
        json!([chunk(7, Value::Null, json!("weather"), json!("}"))]),
        Value::Null
    )]));
    guard.apply(&mut continuation).unwrap();
    let call = &continuation.inner.choices[0]
        .delta
        .tool_calls
        .as_ref()
        .unwrap()[0];
    assert_eq!(call.index, 0);
    assert_eq!(
        call.function.as_ref().unwrap().name.as_deref(),
        Some("weather")
    );
    assert_eq!(
        call.function.as_ref().unwrap().arguments.as_deref(),
        Some("}")
    );
    let mut usage = frame(json!([]));
    usage.nvext = Some(json!({"keep":"metadata"}));
    let before = usage.clone();
    guard.apply(&mut usage).unwrap();
    assert_eq!(usage, before);
    guard.finish().unwrap();
}

#[test]
fn invalid_or_only_empty_calls_fail_at_terminal_and_eof() {
    for call in [
        chunk(0, Value::Null, json!(""), json!("{}")),
        chunk(0, json!("id"), json!(""), json!("{\"x\":1}")),
    ] {
        let mut guard = Guard::default();
        let mut input = frame(json!([delta(0, json!([call]), Value::Null)]));
        guard.apply(&mut input).unwrap();
        assert!(guard.finish().is_err());
        let mut terminal = frame(json!([delta(0, Value::Null, json!("tool_calls"))]));
        assert!(guard.apply(&mut terminal).is_err());
    }
}

#[test]
fn pending_identity_memory_is_bounded() {
    let mut guard = Guard::default();
    let mut input = frame(json!([delta(
        0,
        json!([chunk(
            0,
            Value::Null,
            Value::Null,
            json!("x".repeat(MAX_PENDING_BYTES + 1))
        )]),
        Value::Null
    )]));
    assert_eq!(guard.apply(&mut input), Err(LIMIT));
}

#[test]
fn valid_large_arguments_do_not_inherit_the_pending_identity_limit() {
    let arguments = format!("{{\"text\":\"{}\"}}", "x".repeat(3_500_000));
    let mut input = frame(json!([delta(
        0,
        json!([chunk(
            0,
            json!("call_large"),
            json!("echo"),
            json!(arguments)
        )]),
        json!("tool_calls")
    )]));
    Guard::default().apply(&mut input).unwrap();
    assert_eq!(
        input.inner.choices[0].delta.tool_calls.as_ref().unwrap()[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_ref()
            .unwrap(),
        &arguments
    );
}

#[tokio::test]
async fn filtered_stream_aggregates_and_round_trips_as_valid_history() {
    use crate::protocols::openai::{
        ParsingOptions, chat_completions::aggregator::ChatCompletionAggregator,
    };
    let input = frame(json!([delta(
        0,
        json!([
            chunk(0, Value::Null, json!(""), json!("{}")),
            chunk(1, json!("call_real"), json!("echo"), json!("{\"x\":1}"))
        ]),
        json!("tool_calls")
    )]));
    let mut usage = frame(json!([]));
    usage.inner.usage = Some(
        serde_json::from_value(json!({
            "prompt_tokens":10,"completion_tokens":5,"total_tokens":15
        }))
        .unwrap(),
    );
    let mut response = NvCreateChatCompletionResponse::from_annotated_stream(
        guard_stream(futures::stream::iter(vec![
            Annotated::from_data(input),
            Annotated::from_data(usage),
        ])),
        ParsingOptions::default(),
    )
    .await
    .unwrap();
    guard_response(&mut response).unwrap();
    assert_eq!(response.inner.usage.as_ref().unwrap().total_tokens, 15);
    let calls = response.inner.choices[0]
        .message
        .tool_calls
        .as_ref()
        .unwrap();
    assert_eq!(calls.len(), 1);
    let messages: Vec<ChatCompletionRequestMessage> = serde_json::from_value(json!([
        {"role":"assistant","content":null,"tool_calls":calls},
        {"role":"tool","tool_call_id":"call_real","content":"done"}
    ]))
    .unwrap();
    validate_messages_with_history_policy(&messages, false).unwrap();
}

#[tokio::test]
async fn stream_errors_are_terminal_and_upstream_errors_preserved() {
    let empty = frame(json!([delta(
        0,
        json!([chunk(0, Value::Null, json!(""), json!("{}"))]),
        json!("tool_calls")
    )]));
    let output = guard_stream(futures::stream::iter(vec![Annotated::from_data(empty)]))
        .collect::<Vec<_>>()
        .await;
    assert_eq!(output.len(), 1);
    assert!(output[0].is_error());
    let output = guard_stream(futures::stream::iter(vec![Annotated::from_error(
        "upstream failed",
    )]))
    .collect::<Vec<_>>()
    .await;
    assert!(
        output[0]
            .error
            .as_ref()
            .unwrap()
            .to_string()
            .contains("upstream failed")
    );
}

#[test]
fn final_response_filters_only_empty_stubs_and_rejects_partial_identities() {
    for (calls, succeeds) in [
        (
            json!([{"id":"","type":"function","function":{"name":"","arguments":"{}"}}, real_call()]),
            true,
        ),
        (
            json!([{"id":"","type":"function","function":{"name":"","arguments":"{}"}}]),
            false,
        ),
        (
            json!([{"id":"id","type":"function","function":{"name":" ","arguments":"{}"}}]),
            false,
        ),
    ] {
        let mut response: NvCreateChatCompletionResponse = serde_json::from_value(json!({
            "id":"test","object":"chat.completion","created":1,"model":"glm-5.3",
            "choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":calls},"finish_reason":"tool_calls"}]
        })).unwrap();
        assert_eq!(guard_response(&mut response).is_ok(), succeeds);
        if succeeds {
            assert_eq!(
                response.inner.choices[0]
                    .message
                    .tool_calls
                    .as_ref()
                    .unwrap()
                    .len(),
                1
            );
        }
    }
}

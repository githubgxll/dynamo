// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP chat compatibility and output identity validation. Backend/parser owns
//! argument generation; frontend owns exposing usable tool identities and indices.
//! No worker wire format changes are required, including mixed-version workers.

use std::collections::BTreeMap;

use dynamo_protocols::types::{
    ChatCompletionMessageToolCallChunk, ChatCompletionRequestMessage,
    ChatCompletionRequestToolMessageContent, FunctionCallStream, FunctionType,
};
use futures::{Stream, StreamExt};

use super::{Annotated, NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse};

const PLACEHOLDER_ID: &str = "missing_tool_call_id";
const INVALID: &str = "Backend returned a tool call without a usable id or function name";
const LIMIT: &str = "Backend exceeded the pending tool identity buffer limit";
const MAX_PENDING_BYTES: usize = 1024 * 1024;
const MAX_CALLS: usize = 128;
const MAX_CHOICES: usize = 128;

fn blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn empty_arguments(value: &str) -> bool {
    matches!(value.trim(), "" | "{}")
}

/// Compatibility with histories from the October 2026 GLM-5.3 empty-call bug.
/// Remove once those client histories have expired/migrated. This is deliberately
/// not general orphan-result repair: leave other results intact for validation.
pub(super) fn repair_placeholder_results(messages: &mut Vec<ChatCompletionRequestMessage>) {
    let mut eligible_group = false;
    messages.retain_mut(|message| {
        match message {
            ChatCompletionRequestMessage::Assistant(assistant) => {
                eligible_group = false;
                if let Some(calls) = assistant.tool_calls.as_mut() {
                    // Also handle typed empty strings, which do not trigger the
                    // existing deserialization fallback for null ids.
                    let has_real_call = calls
                        .iter()
                        .any(|call| !blank(&call.id) && !blank(&call.function.name));
                    if has_real_call {
                        calls.retain(|call| {
                            !(blank(&call.id)
                                && blank(&call.function.name)
                                && empty_arguments(&call.function.arguments))
                        });
                    }
                    eligible_group =
                        has_real_call && !calls.iter().any(|call| call.id == PLACEHOLDER_ID);
                }
            }
            ChatCompletionRequestMessage::Tool(tool) => {
                if eligible_group
                    && tool.tool_call_id == PLACEHOLDER_ID
                    && let ChatCompletionRequestToolMessageContent::Text(text) = &tool.content
                    && text.starts_with("Error: Tool 'unknown_tool' failed")
                    && text.contains("tool_call_id")
                    && text.contains("input_value=None")
                    && text.contains("string_type")
                {
                    return false;
                }
            }
            _ => eligible_group = false,
        }
        true
    });
}

#[derive(Default)]
struct Call {
    seen_delta: bool,
    id: String,
    name: String,
    arguments: String,
    output_index: Option<u32>,
    completion: Option<bool>,
}

impl Call {
    fn placeholder(&self) -> bool {
        blank(&self.id) && blank(&self.name) && empty_arguments(&self.arguments)
    }

    fn bytes(&self) -> usize {
        self.id.len() + self.name.len() + self.arguments.len()
    }
}

#[derive(Default)]
struct Choice {
    calls: BTreeMap<u32, Call>,
    emitted: u32,
    finished: bool,
}

impl Choice {
    fn finish(&mut self) -> Result<(), &'static str> {
        if self
            .calls
            .values()
            .any(|call| call.seen_delta && call.output_index.is_none() && !call.placeholder())
            || (self.calls.values().any(|call| call.seen_delta) && self.emitted == 0)
        {
            return Err(INVALID);
        }
        self.finished = true;
        Ok(())
    }
}

#[derive(Default)]
struct Guard {
    choices: BTreeMap<u32, Choice>,
    pending_bytes: usize,
}

impl Guard {
    fn apply(
        &mut self,
        response: &mut NvCreateChatCompletionStreamResponse,
    ) -> Result<(), &'static str> {
        for choice in &mut response.inner.choices {
            if !self.choices.contains_key(&choice.index) && self.choices.len() >= MAX_CHOICES {
                return Err(LIMIT);
            }
            let state = self.choices.entry(choice.index).or_default();
            let mut output = Vec::new();
            for mut chunk in choice.delta.tool_calls.take().unwrap_or_default() {
                if state.finished {
                    return Err(INVALID);
                }
                if !state.calls.contains_key(&chunk.index) && state.calls.len() >= MAX_CALLS {
                    return Err(LIMIT);
                }
                let call = state.calls.entry(chunk.index).or_default();
                call.seen_delta = true;
                if let Some(index) = call.output_index {
                    chunk.index = index;
                    // Absent identity on an arguments-only delta is normal. Do
                    // not let an empty identity overwrite the valid opener.
                    if chunk.id.as_deref().is_some_and(blank) {
                        chunk.id = None;
                    }
                    if let Some(function) = chunk.function.as_mut()
                        && function.name.as_deref().is_some_and(blank)
                    {
                        function.name = None;
                    }
                    output.push(chunk);
                    continue;
                }
                self.pending_bytes -= call.bytes();
                if let Some(id) = chunk.id {
                    call.id.push_str(&id);
                }
                if let Some(function) = chunk.function {
                    if let Some(name) = function.name {
                        call.name.push_str(&name);
                    }
                    if let Some(arguments) = function.arguments {
                        call.arguments.push_str(&arguments);
                    }
                }
                self.pending_bytes += call.bytes();
                if !blank(&call.id) && !blank(&call.name) {
                    let index = state.emitted;
                    state.emitted += 1;
                    call.output_index = Some(index);
                    self.pending_bytes -= call.bytes();
                    output.push(ChatCompletionMessageToolCallChunk {
                        index,
                        id: Some(std::mem::take(&mut call.id)),
                        r#type: Some(FunctionType::Function),
                        function: Some(FunctionCallStream {
                            name: Some(std::mem::take(&mut call.name)),
                            arguments: Some(std::mem::take(&mut call.arguments)),
                        }),
                    });
                } else if self.pending_bytes > MAX_PENDING_BYTES {
                    // This caps only data waiting for an identity, not valid
                    // large argument deltas or the existing HTTP body limit.
                    return Err(LIMIT);
                }
            }
            choice.delta.tool_calls = (!output.is_empty()).then_some(output);
            if choice.finish_reason.is_some() {
                state.finish()?;
            }
        }

        // Completion evidence is internal metadata used by aggregation/dispatch.
        // Keep it on the same remapped index as its call, including late openers.
        for evidence in std::mem::take(&mut response.tool_call_completion) {
            if !self.choices.contains_key(&evidence.choice_index)
                && self.choices.len() >= MAX_CHOICES
            {
                return Err(LIMIT);
            }
            let choice = self.choices.entry(evidence.choice_index).or_default();
            if !choice.calls.contains_key(&evidence.tool_index) && choice.calls.len() >= MAX_CALLS {
                return Err(LIMIT);
            }
            let call = choice.calls.entry(evidence.tool_index).or_default();
            call.completion = Some(call.completion.unwrap_or(false) || evidence.complete);
        }
        for (&choice_index, choice) in &mut self.choices {
            for call in choice.calls.values_mut() {
                if let Some(tool_index) = call.output_index
                    && let Some(complete) = call.completion.take()
                {
                    response.tool_call_completion.push(
                        crate::protocols::openai::chat_completions::ToolCallCompletion {
                            choice_index,
                            tool_index,
                            complete,
                        },
                    );
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), &'static str> {
        for choice in self.choices.values_mut() {
            if !choice.finished {
                choice.finish()?;
            }
        }
        Ok(())
    }
}

/// Owns and polls the original stream directly: no detached task, no full-stream
/// buffering, and dropping this wrapper drops upstream on client disconnect.
pub(super) fn guard_stream<S>(
    input: S,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
{
    async_stream::stream! {
        let mut guard = Guard::default();
        futures::pin_mut!(input);
        while let Some(mut response) = input.next().await {
            if response.is_error() || response.error.is_some() {
                yield response;
                return;
            }
            if let Some(data) = response.data.as_mut()
                && let Err(message) = guard.apply(data)
            {
                yield Annotated::from_error(message);
                return;
            }
            // Keep envelope, usage, content, reasoning and arrival accounting,
            // even when a tool fragment is temporarily withheld.
            yield response;
        }
        if let Err(message) = guard.finish() {
            yield Annotated::from_error(message);
        }
    }
}

/// Raw text can be parsed into tools during non-streaming aggregation, so check
/// final identities as well as the already-structured upstream deltas.
pub(super) fn guard_response(
    response: &mut NvCreateChatCompletionResponse,
) -> Result<(), &'static str> {
    for choice in &mut response.inner.choices {
        if let Some(calls) = choice.message.tool_calls.as_mut() {
            let had_calls = !calls.is_empty();
            calls.retain(|call| {
                !(blank(&call.id)
                    && blank(&call.function.name)
                    && empty_arguments(&call.function.arguments))
            });
            if (had_calls && calls.is_empty())
                || calls
                    .iter()
                    .any(|call| blank(&call.id) || blank(&call.function.name))
            {
                return Err(INVALID);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;

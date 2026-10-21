# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This test verifies that the HTTP server can be started and responds correctly to requests.

import asyncio
import json
import time
from typing import AsyncGenerator, Dict

import aiohttp
import pytest

from dynamo.llm import HttpAsyncEngine, HttpError, HttpService
from dynamo.runtime import DistributedRuntime

MSG_CONTAINS_ERROR = "This message contains an 400error."
MSG_CONTAINS_STATUS_ERROR = "This message contains a 415 status error."
MSG_CONTAINS_INTERNAL_ERROR = "This message contains an internal server error."


class _StatusLikeError(Exception):
    """Mimics dingo.common.http.HttpStatusError's .status + .message shape."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


class MockHttpEngine:
    """A mock engine that returns a completion or raises an error."""

    def __init__(self, model_name: str = "test_model"):
        self.model_name = model_name

    async def generate(self, request: Dict, context) -> AsyncGenerator[Dict, None]:
        """
        Raises HttpError if message contains 'error', otherwise streams a mock response.
        """
        user_message = ""
        for message in request.get("messages", []):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break
        # verifies that cancellation is propagated
        if context.is_stopped():
            print(f"Request {context.id()} was cancelled before starting.")
            return

        if MSG_CONTAINS_ERROR.lower() in user_message.lower():
            raise HttpError(code=400, message=MSG_CONTAINS_ERROR)
        elif MSG_CONTAINS_STATUS_ERROR.lower() in user_message.lower():
            raise _StatusLikeError(status=415, message=MSG_CONTAINS_STATUS_ERROR)
        elif MSG_CONTAINS_INTERNAL_ERROR.lower() in user_message.lower():
            raise ValueError("Simulated internal error")

        envelope = None
        if user_message.startswith("annotated:"):
            _, category, timing = user_message.split(":")
            envelope = {
                "_dynamo_annotated": True,
                "event": "error",
                "error": {
                    "error_type": {"Backend": category},
                    "message": "invalid image count"
                    if category == "InvalidArgument"
                    else "secret /private/backend.py UUID/data mismatch",
                },
            }
            if timing == "early":
                yield envelope
                return

        # Stream a mock response
        created = int(time.time())
        response_text = "This is a mock response."
        for i, char in enumerate(response_text):
            finish_reason = "stop" if i == len(response_text) - 1 else None
            yield {
                "id": f"chatcmpl-{context.id()}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": char},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            if envelope is not None:
                yield envelope
                return
            await asyncio.sleep(0.01)


@pytest.fixture(scope="function", autouse=False)
async def http_server(runtime: DistributedRuntime):
    """Fixture to start a mock HTTP server using HttpService, contributed by Baseten."""
    port = 8008
    model_name = "test_model"
    start_done = asyncio.Event()
    checksum = "abc123"  # Checksum of ModelDeplomentCard for that model
    service = HttpService(port=port)  # Create service outside worker so we can shutdown

    async def worker():
        """The server worker task."""
        try:
            loop = asyncio.get_running_loop()
            python_engine = MockHttpEngine(model_name)
            engine = HttpAsyncEngine(python_engine.generate, loop)

            service.add_chat_completions_model(model_name, checksum, engine)
            service.enable_endpoint("chat", True)

            shutdown_signal = service.run(runtime)
            print("Starting service on port", port)
            start_done.set()
            await shutdown_signal
        except Exception as e:
            print("Server encountered an error:", e)
            start_done.set()
            raise ValueError(f"Server failed to start: {e}")

    server_task = asyncio.create_task(worker())
    await asyncio.wait_for(start_done.wait(), timeout=30.0)
    if server_task.done() and server_task.exception():
        raise ValueError(f"Server task failed to start {server_task.exception()}")
    yield f"http://localhost:{port}", model_name

    # Teardown: Cancel the server task if it's still running
    service.shutdown()  # Shutdown service
    await asyncio.sleep(0.1)  # Give some time for graceful shutdown
    if not server_task.done():
        server_task.cancel()
        try:
            # Await cancellation to ensure proper cleanup for up to 10s
            await asyncio.wait_for(server_task, timeout=10.0)
        except asyncio.CancelledError:
            print("Server task cancelled during teardown.")
            pass


@pytest.mark.asyncio
@pytest.mark.forked
async def test_chat_completion_success(http_server):
    """Tests a successful chat completion request."""
    base_url, model_name = http_server
    url = f"{base_url}/v1/chat/completions"
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello, this is a test."}],
        "stream": True,
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        async with session.post(url, json=data) as response:
            response.raise_for_status()

            content = ""
            async for line in response.content:
                if line.startswith(b"data: "):
                    chunk_data = line[len(b"data: ") :]
                    if chunk_data.strip() == b"[DONE]":
                        break
                    chunk = json.loads(chunk_data)
                    if (
                        chunk["choices"]
                        and chunk["choices"][0]["delta"]
                        and chunk["choices"][0]["delta"].get("content")
                    ):
                        content += chunk["choices"][0]["delta"]["content"]

            assert content == "This is a mock response."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "msg_to_code",
    [
        (MSG_CONTAINS_ERROR, 400),
        (MSG_CONTAINS_STATUS_ERROR, 415),
        (MSG_CONTAINS_INTERNAL_ERROR, 500),
    ],
)
@pytest.mark.forked
async def test_chat_completion_http_error(http_server, msg_to_code: tuple[str, int]):
    """Tests that an HttpError is raised when the message contains 'error'."""
    base_url, model_name = http_server
    url = f"{base_url}/v1/chat/completions"
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": msg_to_code[0]}],
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(url, json=data) as response:
            assert response.status == msg_to_code[1]
            error_json = await response.json()
            if msg_to_code[0] == MSG_CONTAINS_ERROR:
                # 4xx HTTP protocol contract: backend message is forwarded
                # to the client so callers can react to validation errors.
                assert MSG_CONTAINS_ERROR in str(error_json)
            elif msg_to_code[0] == MSG_CONTAINS_STATUS_ERROR:
                # Same 4xx contract via the duck-typed `.status` path.
                assert MSG_CONTAINS_STATUS_ERROR in str(error_json)
            elif msg_to_code[0] == MSG_CONTAINS_INTERNAL_ERROR:
                # 5xx is sanitized: the client receives a static message
                # and never the raw backend error text; the underlying
                # detail is logged server-side.
                assert "simulated internal error" not in str(error_json).lower()
                assert "internal server error" in str(error_json).lower()


@pytest.mark.asyncio
@pytest.mark.forked
@pytest.mark.parametrize("category,code", [("InvalidArgument", 400), ("Unknown", 500)])
@pytest.mark.parametrize("timing", ["early", "late"])
@pytest.mark.parametrize("stream", [False, True])
async def test_annotated_error_survives_python_http_hop(
    http_server, category, code, timing, stream
):
    """Tagged errors never enter chat-data decoding; HTTP retains safe 400/500 errors."""
    base_url, model_name = http_server
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session,
        session.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {"role": "user", "content": f"annotated:{category}:{timing}"}
                ],
                "stream": stream,
            },
        ) as response,
    ):
        body = await response.text()
        assert "missing field" not in body
        assert "secret" not in body
        assert "/private/" not in body
        if stream and response.status == 200:
            frames = [
                json.loads(line[6:])
                for line in body.splitlines()
                if line.startswith("data: ") and line[6:] != "[DONE]"
            ]
            errors = [frame["error"] for frame in frames if "error" in frame]
            assert len(errors) == 1
            assert frames[-1] == {"error": errors[0]}
            assert errors[0]["code"] == code
            assert "data: [DONE]" in body
            if timing == "late":
                assert any("choices" in frame for frame in frames[:-1])
        else:
            assert response.status == code
            error = json.loads(body)
            assert error["code"] == code
            assert error["message"]
            assert error["type"]
        if code == 400:
            assert "invalid image count" in body
        else:
            assert "internal server error" in body.lower()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check Python engine errors through a real TCP/etcd worker and frontend.

Validate HTTP statuses, SSE error payloads and termination, and sanitization
of unexpected backend failures.
"""

from __future__ import annotations

import json
from typing import Generator

import pytest
import requests

from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import ServicePorts

MODEL_NAME = "test-http-status-prop"
ENDPOINT_PATH = "test.http_status_prop.generate"
EXPECTED_STATUS = 415
EXPECTED_MESSAGE = "unsupported-media-via-wire"

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
]


class _WorkerProcess(ManagedProcess):
    def __init__(self, request, *, frontend_port: int) -> None:
        command = ["python3", "-m", "tests.frontend.http_status_propagation_worker"]
        scenario = getattr(request, "param", None)
        if scenario is not None:
            error_kind, message, after_chunk = scenario
            command.extend(["--error-kind", error_kind, "--message", message])
            if after_chunk:
                command.append("--after-chunk")
        super().__init__(
            command=command,
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._model_listed)
            ],
            timeout=60,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m tests.frontend.http_status_propagation_worker"],
            log_dir=f"{request.node.name}_worker",
        )

    @staticmethod
    def _model_listed(response: requests.Response) -> bool:
        try:
            if response.status_code != 200:
                return False
            data = response.json()
        except (ValueError, KeyError):
            return False
        return any(m.get("id") == MODEL_NAME for m in data.get("data", []))


@pytest.fixture(scope="function")
def services(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports: ServicePorts,
) -> Generator[int, None, None]:
    _ = runtime_services_dynamic_ports
    frontend_port = dynamo_dynamic_ports.frontend_port
    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_args=["--discovery-backend", "etcd", "--request-plane", "tcp"],
        terminate_all_matching_process_names=False,
    ):
        with _WorkerProcess(request, frontend_port=frontend_port):
            yield frontend_port


def test_http_status_propagates_through_wire(services: int) -> None:
    response = requests.post(
        f"http://localhost:{services}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hello"}],
        },
        timeout=30,
    )
    assert response.status_code == EXPECTED_STATUS, response.text
    assert response.json()["message"] == EXPECTED_MESSAGE


_LIMIT_ERRORS = [
    (
        "invalid_argument",
        "At most 16 image(s) may be provided in one prompt. (parameter=image)",
    ),
    (
        "invalid_argument",
        "At most 1 video(s) may be provided in one prompt. (parameter=video)",
    ),
    (
        "invalid_argument",
        "At most 0 audio(s) may be provided in one prompt. (parameter=audio)",
    ),
    ("value_error", "At most 8 image(s) may be provided in one prompt."),
    ("http_status", EXPECTED_MESSAGE),
    ("unknown", "private backend implementation detail"),
]


@pytest.mark.parametrize(
    "services,error_kind,message",
    [
        pytest.param((kind, message, False), kind, message, id=f"{kind}-{index}")
        for index, (kind, message) in enumerate(_LIMIT_ERRORS)
    ],
    indirect=["services"],
)
def test_validation_errors_propagate_through_wire(
    services: int, error_kind: str, message: str
) -> None:
    expected_code = (
        500 if error_kind == "unknown" else 415 if error_kind == "http_status" else 400
    )
    for stream in (False, True):
        response = requests.post(
            f"http://localhost:{services}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
            timeout=30,
        )
        assert response.status_code == (200 if stream else expected_code), response.text
        if stream:
            frames = [
                line.removeprefix("data: ")
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            assert frames[-1] == "[DONE]", response.text
            assert frames.count("[DONE]") == 1, response.text
            payloads = [json.loads(frame) for frame in frames[:-1]]
            errors = [payload["error"] for payload in payloads if "error" in payload]
            assert len(errors) == 1, response.text
            assert "error" in payloads[-1], response.text
            error = errors[0]
        else:
            error = response.json()
        assert error["code"] == expected_code, error
        if error_kind == "unknown":
            assert message not in response.text
        elif error_kind == "value_error":
            # The generic Python-exception bridge retains the exception class.
            assert error["message"] == f"ValueError: {message}", error
        else:
            assert error["message"] == message, error


@pytest.mark.parametrize(
    "services",
    [("invalid_argument", _LIMIT_ERRORS[0][1], True)],
    indirect=True,
)
def test_validation_error_after_chunk_terminates_stream(services: int) -> None:
    response = requests.post(
        f"http://localhost:{services}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
        timeout=30,
    )
    assert response.status_code == 200, response.text
    frames = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert frames[-1] == "[DONE]", response.text
    assert frames.count("[DONE]") == 1, response.text
    payloads = [json.loads(frame) for frame in frames[:-1]]
    errors = [payload["error"] for payload in payloads if "error" in payload]
    assert len(errors) == 1, response.text
    assert payloads[-1]["error"] == errors[0]
    assert errors[0]["code"] == 400
    assert errors[0]["message"] == _LIMIT_ERRORS[0][1]
    chunks = [payload for payload in payloads if "choices" in payload]
    assert chunks, response.text
    assert all(
        choice.get("finish_reason") is None
        for chunk in chunks
        for choice in chunk["choices"]
    ), response.text

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, asdict

import pytest

from dingo.common.task_protocol import (
    ENVELOPE_KEY,
    ControlRequest,
    InputDescriptor,
    ResultDescriptor,
    TaskIdentity,
    TaskResponse,
    WorkerCapabilities,
    parse_request,
    resolve_artifact,
)
from dingo.common.video_task_protocol import DetachedTaskIdentity, detached_envelope


def identity():
    return TaskIdentity(
        "deployment",
        "context_ir",
        "compiler",
        "context-ir-1",
        1,
        "a" * 32,
        "worker-1",
        "generation-1",
    )


def request(op="submit"):
    task = identity()
    payload = InputDescriptor(
        "shared",
        "task/request.json",
        "task/manifest.json",
        "b" * 64,
        "task/route.json",
        "c" * 64,
        "d" * 64,
    )
    return ControlRequest(
        op,
        task.deployment_id,
        task.pool_id,
        task.instance_id,
        task.generation,
        task if op not in {"capabilities", "drain"} else None,
        payload if op == "submit" else None,
        123456 if op == "submit" else None,
        1000 if op == "wait" else None,
    )


@pytest.mark.parametrize(
    "op", ["submit", "status", "wait", "cancel", "capabilities", "drain"]
)
def test_round_trip(op):
    original = request(op)
    assert parse_request(original.to_dict()) == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 1),
        ("schema_version", 2.0),
        ("schema_version", True),
        ("op", "wait_terminal"),
        ("op", "result"),
        ("op", []),
        ("attempt", True),
        ("attempt", 0),
        ("attempt", "1"),
        ("task_kind", "llm"),
        ("task_kind", []),
        ("task_id", "../other"),
        ("deployment_id", None),
        ("generation", ""),
        ("instance_id", 10),
        ("execution_token", "bad"),
        ("deadline_at_ms", -1),
        ("deadline_at_ms", True),
        ("extra", "unknown"),
    ],
)
def test_reject_invalid_envelope(field, value):
    message = request().to_dict()
    message[ENVELOPE_KEY][field] = value
    with pytest.raises(ValueError):
        parse_request(message)


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../secret",
        "a/../secret",
        "a//b",
        "./a",
        "C:/a",
        "a\\b",
        "a\x00b",
        "",
    ],
)
def test_reject_invalid_paths(path):
    message = request().to_dict()
    message[ENVELOPE_KEY]["payload"]["request_path"] = path
    with pytest.raises(ValueError):
        parse_request(message)


def test_operation_fields_and_control_limit():
    for op, field, value in [
        ("status", "payload", {}),
        ("drain", "task_id", "x"),
        ("wait", "wait_timeout_ms", 30001),
    ]:
        message = request(op).to_dict()
        message[ENVELOPE_KEY][field] = value
        with pytest.raises(ValueError):
            parse_request(message)
    message = request().to_dict()
    del message[ENVELOPE_KEY]["payload"]["request_digest"]
    with pytest.raises(ValueError):
        parse_request(message)
    message = request().to_dict()
    message[ENVELOPE_KEY]["payload"]["request_path"] = "x" * (2 * 1024 * 1024)
    with pytest.raises(ValueError, match="size limit"):
        parse_request(message)
    message = request().to_dict()
    message["prompt"] = "not allowed"
    with pytest.raises(ValueError):
        parse_request(message)


def test_symlink_escape_and_regular_files(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    file = root / "input.json"
    file.write_text("{}")
    assert resolve_artifact(root, "input.json") == file
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (root / "link").symlink_to(outside)
    (root / "directory").mkdir()
    for path in ("link", "directory"):
        with pytest.raises(ValueError):
            resolve_artifact(root, path)


def test_v1_remains_separate_and_strict():
    old = detached_envelope(
        op="submit",
        deployment_id="d",
        pool_id="p",
        task_id="video-1",
        attempt=1,
        execution_token="a" * 32,
    )
    envelope = old["_dingo_video_task"]
    assert DetachedTaskIdentity.from_envelope(envelope).task_id == "video-1"
    with pytest.raises(ValueError):
        parse_request(old)
    with pytest.raises(ValueError, match="unknown"):
        DetachedTaskIdentity.from_envelope({**envelope, "task_kind": "context_ir"})


def test_result_states_and_descriptor_validation():
    base = {"schema_version": 2, "identity": asdict(identity()), "state": "running"}
    assert TaskResponse.parse(base).result is None
    result = asdict(ResultDescriptor("shared", "attempt/result.json", "e" * 64, 100))
    success = {**base, "state": "succeeded", "result": result}
    assert TaskResponse.parse(success).result.size_bytes == 100
    for invalid in [
        {**base, "state": "succeeded"},
        {**base, "result": result},
        {**base, "state": "failed"},
        {**base, "state": "unknown"},
        {**base, "state": "not_accepted", "error_code": "bad\ntrace"},
        {**success, "result": {**result, "size_bytes": True}},
        {**success, "result": {**result, "size_bytes": 524289}},
    ]:
        with pytest.raises(ValueError):
            TaskResponse.parse(invalid)
    assert TaskResponse.parse(
        {**base, "state": "not_accepted", "error_code": "draining"}
    )
    assert TaskResponse.parse({**base, "state": "not_found"})


def capabilities():
    return {
        "instance_id": "w",
        "generation": "g",
        "capacity_domain": "qwen",
        "storage_id": "shared",
        "protocol_versions": [2],
        "task_kinds": ["context_ir"],
        "execution_capacity": 16,
        "prefetch_capacity": 0,
        "model_adapters": {"minimax_h3": 1},
        "protocol_adapters": {"open_h3_ir": 1},
        "queue_allowed": True,
        "accepting": True,
        "draining": False,
    }


def test_capabilities_are_immutable_and_validate_admission():
    wire = capabilities()
    parsed = WorkerCapabilities.parse(wire)
    wire["model_adapters"]["minimax_h3"] = 99
    assert parsed.model_adapters == (("minimax_h3", 1),)
    with pytest.raises(FrozenInstanceError):
        parsed.execution_capacity = 32
    for patch in [
        {"draining": True},
        {"execution_capacity": True},
        {"queue_allowed": False},
        {"prefetch_capacity": 1},
        {"task_kinds": [[]]},
        {"protocol_versions": [True]},
        {"protocol_versions": [2, 2]},
        {"model_adapters": {"h3": 0}},
    ]:
        with pytest.raises(ValueError):
            WorkerCapabilities.parse({**capabilities(), **patch})
    assert WorkerCapabilities.parse(
        {**capabilities(), "accepting": False, "queue_allowed": False, "draining": True}
    )


def test_capabilities_bootstrap_does_not_require_unknown_generation():
    message = request("capabilities").to_dict()
    del message[ENVELOPE_KEY]["generation"]
    parsed = parse_request(message)
    assert parsed.generation is None
    assert parsed.to_dict() == message
    message[ENVELOPE_KEY]["op"] = "drain"
    with pytest.raises(ValueError):
        parse_request(message)

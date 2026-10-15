# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "collector_v2", Path(__file__).resolve().parents[1] / "collector.py"
)
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)
SOURCE = {
    "cluster_id": "hd04",
    "namespace": "elm-test",
    "pod_uid": "uid",
    "pod": "gateway-1",
    "container": "engine",
    "restart": 0,
    "image": "image@sha256:test",
}
EVENT = {
    "log_type": "video_task_lifecycle",
    "event": "completed",
    "deployment_id": "d1",
    "task_id": "t1",
    "model": "minimax-h3",
    "pool_id": "fl2va",
    "timestamp_ms": 1791914400000,
    "performance": {
        "schema_version": 1,
        "generation": {"seconds": 13, "num_inference_steps": 30},
        "counts": {"image": 1},
        "model_execution": {"schema": "test", "stages": {"diffuse": 3.5}},
    },
}


def line(stamp="2026-10-14T01:00:00.123456789Z", event=None):
    return (
        stamp + " INFO " + json.dumps(EVENT if event is None else event) + "\n"
    ).encode()


def test_config_defaults():
    assert c.configuration({})["namespaces"] == ["token-factory"]
    assert c.configuration({"namespaces": []})["namespaces"] == []
    assert c.configuration({"namespaces": ["b", "a", "b"]})["namespaces"] == ["a", "b"]


@pytest.mark.parametrize("versions", [[1], [3], [2, 3]])
def test_unknown_database_version_is_rejected_before_schema_writes(
    monkeypatch, versions
):
    import psycopg

    class Database:
        closed = False

        def execute(self, query):
            if query.startswith("SELECT pg_try_advisory_lock"):
                self.result = (True,)
            elif query.startswith("SELECT to_regclass"):
                self.result = ("video_performance", "collector_schema")
            elif query.startswith("SELECT array_agg(version)"):
                self.result = (versions,)
            else:
                raise AssertionError("Unsupported database must not be mutated")
            return self

        def fetchone(self):
            return self.result

        def close(self):
            self.closed = True

    db = Database()
    monkeypatch.setenv("DB_PASSWORD", "test-only")
    monkeypatch.setattr(psycopg, "connect", lambda **_: db)
    with pytest.raises(RuntimeError, match="unsupported_database_schema"):
        c.connect()
    assert db.closed


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"unknown": 1},
        {"namespaces": "elm-test"},
        {"namespaces": ["../other"]},
        {"namespaces": ["A"]},
        {"poll_interval_s": 0},
        {"poll_interval_s": True},
        {"pause_fraction": float("nan")},
        {"warning_fraction": 0.99},
        {"storage_reserve_bytes": 10737418240},
        {"max_read_bytes": 1024},
    ],
)
def test_bad_config(bad):
    with pytest.raises(ValueError):
        c.configuration(bad)


def test_bad_config_preserves_previous(tmp_path):
    config = c.configuration({"namespaces": ["elm-test"]})
    got, error = c.load_configuration(tmp_path / "missing", config)
    assert got is config and error


def pod(command, name="engine", annotation=None):
    return {
        "metadata": {
            "name": "p",
            "namespace": "elm-test",
            "uid": "u",
            "annotations": {c.ANNOTATION: annotation} if annotation else {},
        },
        "spec": {"containers": [{"name": name, "command": command}]},
        "status": {
            "containerStatuses": [
                {
                    "name": name,
                    "restartCount": 2,
                    "state": {"running": {"startedAt": "now"}},
                }
            ]
        },
    }


@pytest.mark.parametrize(
    "command",
    [
        ["python", "-m", "dingo.video_gateway"],
        ["bash", "-c", "exec python -m dingo.video_gateway --config a.yaml"],
    ],
)
def test_discovery(command):
    result = list(c.discover(pod(command), "hd04"))
    assert (
        len(result) == 1
        and result[0]["container"] == "engine"
        and result[0]["restart"] == 2
    )


@pytest.mark.parametrize(
    "command",
    [
        ["python", "-m", "dynamo.frontend"],
        ["python", "-m", "dingo.video_gateway_fake"],
        ["python", "-m", "dingo.vllm.omni"],
    ],
)
def test_discovery_excludes_other_services(command):
    assert not list(c.discover(pod(command), "hd04"))


def test_explicit_container_annotation():
    assert list(c.discover(pod(["/start.sh"], annotation="engine"), "hd04"))
    assert not list(c.discover(pod(["/start.sh"], annotation="wrong"), "hd04"))


def test_optional_and_dynamic_stages():
    value = c.event_row(EVENT, SOURCE)
    assert value["seconds"] == 13 and value["stages"] == {"diffuse": 3.5}
    assert value["environment_revision"] == "unknown" and value["video_frames"] is None
    assert value["namespace"] == "elm-test"
    assert c.event_row(EVENT, {**SOURCE, "namespace": "other"})["namespace"] == "other"


def test_cancelled_before_first_dispatch_has_attempt_zero():
    assert (
        c.event_row({**EVENT, "event": "cancelled", "attempt": 0}, SOURCE)["attempt"]
        == 0
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        (0, "0"),
        (1696626444545984353, "1696626444545984353"),
        (18446744073709551615, "18446744073709551615"),
        ("1696626444545984353", "1696626444545984353"),
        ("worker-a", "worker-a"),
    ],
)
def test_worker_identity_preserves_integer_precision(value, expected):
    result = c.event_row({**EVENT, "worker_instance_id": value}, SOURCE)
    assert result["details"]["worker_id"] == expected


@pytest.mark.parametrize(
    "value", [True, False, -1, 18446744073709551616, 1.5, [], {}, "x" * 513]
)
def test_invalid_worker_identity_is_rejected(value):
    with pytest.raises(ValueError):
        c.event_row({**EVENT, "worker_instance_id": value}, SOURCE)


def test_no_private_fields_and_reference_allowlist():
    event = copy.deepcopy(EVENT)
    event["prompt"] = "PRIVATE"
    event["performance"]["prompt"] = "PRIVATE"
    event["performance"]["references"] = [
        {"kind": "image", "url": "PRIVATE", "bytes": 1024, "width": 1920}
    ]
    result = c.event_row(event, SOURCE)
    assert "PRIVATE" not in str(result)
    assert result["details"]["performance"]["references"][0]["width"] == 1920


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, "3", True])
def test_bad_numeric(value):
    event = copy.deepcopy(EVENT)
    event["performance"]["generation"]["seconds"] = value
    with pytest.raises(ValueError):
        c.event_row(event, SOURCE)


def test_unsupported_schema():
    event = copy.deepcopy(EVENT)
    event["performance"]["schema_version"] = 2
    with pytest.raises(ValueError):
        c.event_row(event, SOURCE)


def test_batch_deduplicates_same_timestamp_and_nanoseconds():
    raw = (
        line()
        + line(event={**EVENT, "task_id": "t2"})
        + line("2026-10-14T01:00:00.123456790Z")
    )
    rows, stamp, count, invalid, truncated, _ = c.read_batch(
        io.BytesIO(raw), SOURCE, "2026-10-14T01:00:00.123456789Z", 1, c.DEFAULTS
    )
    assert len(rows) == 2 and rows[0]["task_id"] == "t2"
    assert stamp.endswith("790Z") and count == 1 and invalid == 0 and not truncated


def test_prefix_limit_continuation():
    raw = line() * 5
    config = {**c.DEFAULTS, "max_read_bytes": len(line()) * 2}
    first = c.read_batch(io.BytesIO(raw), SOURCE, None, 0, config)
    assert first[2] == 2 and first[4]
    with pytest.raises(ValueError, match="without_progress"):
        c.read_batch(io.BytesIO(raw), SOURCE, first[1], first[2], config)
    # Explicit blockage, not silently skipping an arbitrary suffix.
    second = c.read_batch(io.BytesIO(raw), SOURCE, first[1], first[2], c.DEFAULTS)
    assert len(second[0]) == 3 and second[2] == 5


def test_malformed_event_isolated():
    raw = line(event={**EVENT, "performance": {"generation": "bad"}}) + line()
    result = c.read_batch(io.BytesIO(raw), SOURCE, None, 0, c.DEFAULTS)
    assert len(result[0]) == 1 and result[3] == 1


def test_oversized_line_not_checkpointed():
    with pytest.raises(ValueError, match="partial"):
        c.read_batch(
            io.BytesIO(line()), SOURCE, None, 0, {**c.DEFAULTS, "max_line_bytes": 10}
        )


def test_non_monotonic_logs_block():
    with pytest.raises(ValueError, match="non_monotonic"):
        c.read_batch(
            io.BytesIO(line() + line("2025-10-14T01:00:00Z")),
            SOURCE,
            None,
            0,
            c.DEFAULTS,
        )


def test_ordinary_logs_advance_cursor():
    result = c.read_batch(
        io.BytesIO(b"2026-10-14T01:00:00Z hello\n"), SOURCE, None, 0, c.DEFAULTS
    )
    assert result[0] == [] and result[2] == 1


def test_timezone_offset_preserves_nanosecond_order():
    assert c.timestamp_key("2026-10-14T09:00:00.123456789+08:00") == c.timestamp_key(
        "2026-10-14T01:00:00.123456789Z"
    )
    assert c.timestamp_key("2026-10-14T00:00:00.000000001-01:00") > c.timestamp_key(
        "2026-10-14T01:00:00Z"
    )

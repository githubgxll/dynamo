# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib
import json
import threading
import urllib.error
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture
def pull(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("http_pull")


TARGET = {
    "name": "gateway-a",
    "cluster_id": "region-a",
    "namespace": "example-video",
    "deployment_id": "example",
    "url": "https://gateway.example.invalid",
    "token_env": "TEST_PERFORMANCE_TOKEN",
}


def page():
    return {
        "schema_version": 1,
        "instance_id": "instance-a",
        "next_cursor": "cursor-a",
        "next_seq": 1,
        "has_more": False,
        "gap_detected": False,
        "instance_changed": False,
        "source": {
            "deployment_id": "example",
            "namespace": "example-video",
            "pod": "gateway-a",
            "pod_uid": "uid-a",
            "container": "gateway",
            "image": "image-a",
        },
        "records": [
            {
                "seq": 1,
                "event": {
                    "log_type": "video_task_lifecycle",
                    "event": "completed",
                    "deployment_id": "example",
                    "task_id": "video-a",
                    "model": "model-a",
                    "pool_id": "pool-a",
                    "attempt": 1,
                    "worker_instance_id": 1696626444545984353,
                    "timestamp_ms": 1791977739000,
                    "performance": {
                        "schema_version": 1,
                        "generation": {"seconds": 5},
                        "counts": {},
                        "model_execution": {
                            "schema": "other_model.v1",
                            "stages": {"new_stage": 1.25},
                        },
                    },
                },
            }
        ],
    }


def test_defaults_and_target_validation(pull):
    assert pull.configuration({})["targets"] == []
    target = pull.configuration({"targets": [TARGET]})["targets"][0]
    assert target["poll_interval_s"] == 5 and target["timeout_s"] == 3
    assert (
        target["binding"]
        == pull.configuration({"targets": [TARGET]})["targets"][0]["binding"]
    )


def test_http_renderer_is_namespaced_and_colocates_database(pull):
    target = pull.configuration({"targets": [TARGET]})["targets"][0]
    renderer = importlib.import_module("render_http")
    manifest = renderer.render(
        "example-observability", "example/collector:test", {"targets": [TARGET]}
    )
    items = manifest["items"]
    assert len(items) == 3
    assert {item["kind"] for item in items} == {"ConfigMap", "Service", "Deployment"}
    assert all(
        item["metadata"]["namespace"] == "example-observability" for item in items
    )
    workloads = [x for x in items if x["kind"] == "Deployment"]
    assert len(workloads) == 1
    for deployment in workloads:
        pod = deployment["spec"]["template"]["spec"]
        assert pod["automountServiceAccountToken"] is False
        assert {c["name"] for c in pod["containers"]} == {"postgres", "collector"}
        assert "nodeName" not in pod and "nvidia.com/gpu" not in str(pod)
        assert {v["name"] for v in pod["volumes"]} == {"data", "config"}
        collector = next(c for c in pod["containers"] if c["name"] == "collector")
        postgres = next(c for c in pod["containers"] if c["name"] == "postgres")
        assert collector["command"][-1].endswith("http_pull.py")
        env = {e["name"]: e.get("value") for e in collector["env"]}
        assert env["PGHOST"] == "127.0.0.1"
        assert env["DATA_PATH"] == "/var/lib/postgresql/data"
        assert any(
            v["name"] == "data" and v.get("readOnly") for v in collector["volumeMounts"]
        )
        assert any(
            v["name"] == "data" and not v.get("readOnly")
            for v in postgres["volumeMounts"]
        )
        service = next(x for x in items if x["kind"] == "Service")
        assert (
            service["spec"]["selector"] == deployment["spec"]["selector"]["matchLabels"]
        )
    assert (
        target["binding"]
        != pull.configuration(
            {"targets": [{**TARGET, "url": "https://new.example.invalid"}]}
        )["targets"][0]["binding"]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"url": "http://gateway.example.invalid"},
        {"url": "https://user:password@host"},
        {"url": "https://host/?token=secret"},
        {"url": "file:///tmp/test"},
        {"poll_interval_s": 0},
        {"timeout_s": float("nan")},
        {"token_env": "bad-env"},
        {"namespace": None},
        {"unknown": True},
    ],
)
def test_invalid_targets(pull, change):
    with pytest.raises(ValueError):
        pull.configuration({"targets": [{**TARGET, **change}]})


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"next_seq": True},
        {"has_more": "yes"},
        {"has_more": True, "records": []},
        {"source": {"deployment_id": "wrong"}},
        {"records": [{"seq": 3}, {"seq": 2}]},
    ],
)
def test_bad_envelope_does_not_reach_database(pull, change):
    target = pull.configuration({"targets": [TARGET]})["targets"][0]
    with pytest.raises(ValueError):
        pull.validate_page(target, {**page(), **change})


class FakeDB:
    def __init__(self, fail_commit=False):
        self.samples = []
        self.state = None
        self.rejected = []
        self.fail_commit = fail_commit

    @contextmanager
    def transaction(self):
        before = copy.deepcopy((self.samples, self.state, self.rejected))
        try:
            yield
            if self.fail_commit:
                raise RuntimeError("commit failed")
        except Exception:
            self.samples, self.state, self.rejected = before
            raise

    def execute(self, sql, params=()):
        if sql.startswith("SELECT binding"):
            self.result = self.state
        elif sql.startswith("INSERT INTO performance_http_sources"):
            self.state = (params[1], params[2], params[3])
        elif sql.startswith("INSERT INTO performance_http_rejections"):
            self.rejected.append(params)
        return self

    def fetchone(self):
        return self.result


def test_rows_and_cursor_roll_back_on_failed_commit(pull, monkeypatch):
    target = pull.configuration({"targets": [TARGET]})["targets"][0]

    def insert(db, row):
        db.samples.append(row)
        return "inserted"

    monkeypatch.setattr(pull, "insert_row", insert)
    db = FakeDB(fail_commit=True)
    with pytest.raises(RuntimeError):
        pull.apply_page(db, target, page())
    assert db.samples == [] and db.state is None
    db.fail_commit = False
    assert pull.apply_page(db, target, page()) == (1, 0)
    assert db.state[2] == "cursor-a"
    assert db.samples[0]["source"] == "gateway_http"
    assert db.samples[0]["stages"] == {"new_stage": 1.25}
    assert db.samples[0]["details"]["worker_id"] == "1696626444545984353"


def test_duplicates_and_conflicts_are_distinguished(pull, monkeypatch):
    target = pull.configuration({"targets": [TARGET]})["targets"][0]
    db = FakeDB()
    monkeypatch.setattr(pull, "insert_row", lambda *_: "duplicate")
    assert pull.apply_page(db, target, page()) == (0, 0)
    monkeypatch.setattr(pull, "insert_row", lambda *_: "conflict")
    assert pull.apply_page(db, target, page()) == (0, 1)
    assert db.rejected[-1][-1] == "conflict"


def test_invalid_event_is_quarantined_without_raw_payload(pull):
    target = pull.configuration({"targets": [TARGET]})["targets"][0]
    value = page()
    value["records"][0]["event"] = {"prompt": "must-not-be-persisted"}
    db = FakeDB()
    assert pull.apply_page(db, target, value) == (0, 1)
    assert db.state[2] == value["next_cursor"]
    assert "must-not-be-persisted" not in str(db.rejected)


def test_fetch_uses_token_and_refuses_redirects(pull, monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            if self.path.startswith("/redirect/"):
                self.send_response(302)
                self.send_header("Location", "/somewhere-else")
                self.end_headers()
                return
            body = json.dumps(page()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    monkeypatch.setenv(TARGET["token_env"], "test-token-long-enough")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        target = pull.configuration(
            {
                "targets": [
                    {
                        **TARGET,
                        "url": f"http://127.0.0.1:{server.server_port}",
                        "allow_insecure_http": True,
                    }
                ]
            }
        )["targets"][0]
        assert pull.fetch(target, None)["instance_id"] == "instance-a"
        assert seen == ["Bearer test-token-long-enough"]
        with pytest.raises(urllib.error.HTTPError, match="302"):
            pull.fetch({**target, "url": target["url"] + "/redirect"}, None)
        assert len(seen) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

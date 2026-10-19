# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query the real VM engine with partial histogram scrapes and a slow-request control.

Run with VICTORIAMETRICS_BINARY=/path/to/victoria-metrics-prod. All storage and
requests belong to a temporary local process; production endpoints are not used.
"""
import copy
import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def panels(items):
    for panel in items:
        yield panel
        yield from panels(panel.get("panels", []))


@pytest.fixture
def vm(tmp_path):
    binary = os.environ.get("VICTORIAMETRICS_BINARY") or shutil.which(
        "victoria-metrics-prod"
    )
    if not binary:
        pytest.skip(
            "set VICTORIAMETRICS_BINARY to run the real-engine histogram regression"
        )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with (tmp_path / "vm.log").open("w") as log:
        process = subprocess.Popen(
            [
                binary,
                f"-httpListenAddr=127.0.0.1:{port}",
                f"-storageDataPath={tmp_path / 'data'}",
                "-memory.allowedBytes=134217728",
                "-search.maxConcurrentRequests=2",
                "-search.disableCache=true",
                "-search.latencyOffset=0s",
            ],
            stdout=log,
            stderr=log,
        )
        try:
            for _ in range(100):
                assert process.poll() is None, (tmp_path / "vm.log").read_text()
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                pytest.fail("temporary VM did not become ready")
            yield base
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_partial_scrapes_do_not_create_false_p95_and_real_slow_requests_remain(vm):
    records = [
        json.loads(line)
        for line in (ROOT / "tests/fixtures/histogram-partial-scrapes.jsonl")
        .read_text()
        .splitlines()
    ]
    # Keep the exact 15-second query / 30-second scrape phase, but avoid retention
    # dependence on the original incident date.
    shift = ((int(time.time()) - 1792120800 - 60) // 15) * 15
    for record in records:
        record["timestamps"] = [ts + shift * 1000 for ts in record["timestamps"]]
    control = []
    for record in records:
        if record["metric"]["stage"] != "encode_prompt":
            continue
        altered = copy.deepcopy(record)
        altered["metric"]["pool"] = "true-slow-request"
        # One formerly fast observation actually takes 480 seconds in this control.
        if float(altered["metric"]["le"]) < 480:
            altered["values"] = [
                value - int(ts >= 1792120591193 + shift * 1000)
                for ts, value in zip(altered["timestamps"], altered["values"])
            ]
        control.append(altered)
    payload = "".join(
        json.dumps(record) + "\n" for record in records + control
    ).encode()
    request = urllib.request.Request(
        vm + "/api/v1/import",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status == 204
    with urllib.request.urlopen(vm + "/internal/force_flush", timeout=10):
        pass

    def query(expression, start=1792120200, end=1792120800):
        params = {
            "query": expression,
            "start": start + shift,
            "end": end + shift,
            "step": "15s",
            "nocache": "1",
        }
        with urllib.request.urlopen(
            vm + "/api/v1/query_range?" + urllib.parse.urlencode(params), timeout=10
        ) as response:
            result = json.load(response)
        assert result["status"] == "success", result
        return {
            (series["metric"]["pool"], series["metric"]["stage"], ts - shift): float(
                value
            )
            for series in result["data"]["result"]
            for ts, value in series["values"]
        }

    dashboard = json.loads((ROOT / "grafana/video-gateway.json").read_text())
    by_id = {panel["id"]: panel for panel in panels(dashboard["panels"])}
    expression = by_id[91]["targets"][0]["expr"]
    expression = expression.replace("${snapshot_interval}", "15s")
    # Preserve the original incident's five-minute counterexample. The current
    # one-minute production window and idle tail are exercised below.
    expression = expression.replace("[1m:15s]", "[5m:15s]")
    for name in ["namespace", "pool", "timing_schema", "model_stage"]:
        expression = expression.replace("$" + name, ".*")
    current = query(expression)
    observed = {
        key: value for key, value in current.items() if key[0] == "observed-samples"
    }
    assert len(observed) == 82
    assert max(observed.values()) <= 0.25
    for timestamp in [1792120410, 1792120425]:
        assert observed[
            ("observed-samples", "encode_prompt", timestamp)
        ] == pytest.approx(0.235)
    for timestamp in [1792120710, 1792120725]:
        assert observed[
            ("observed-samples", "encode_prompt", timestamp)
        ] == pytest.approx(0.0975)
    for timestamp in [1792120650, 1792120665]:
        assert observed[
            ("observed-samples", "audio_decode", timestamp)
        ] == pytest.approx(0.2425)
    assert current[("true-slow-request", "encode_prompt", 1792120710)] == pytest.approx(
        570
    )
    assert query(expression, 1792121160, 1792121220) == {}

    # Verify that this fixture really exercises the defect, rather than only
    # asserting a normal output for data that could never trigger the old query.
    old = query(unguarded_expression(expression).replace("increase(", "rate("))
    assert old[("observed-samples", "encode_prompt", 1792120410)] == pytest.approx(450)
    assert old[("observed-samples", "audio_decode", 1792120650)] == pytest.approx(1530)
    assert (
        sum(value > 1 for key, value in old.items() if key[0] == "observed-samples")
        == 6
    )

    repeated = by_id[92]["targets"][1]["expr"]
    repeated = repeated.replace("${snapshot_interval}", "15s")
    repeated = repeated.replace("[1m:15s]", "[5m:15s]")
    for name in ["namespace", "pool", "timing_schema"]:
        repeated = repeated.replace("$" + name, ".*")
    repeated = repeated.replace("$model_stage", "encode_prompt")
    assert query(repeated) == {
        key: value for key, value in current.items() if key[1] == "encode_prompt"
    }


def dashboard_expression(filename, panel_id, ref):
    dashboard = json.loads((ROOT / "grafana" / filename).read_text())
    panel = next(p for p in panels(dashboard["panels"]) if p["id"] == panel_id)
    expression = next(t["expr"] for t in panel["targets"] if t["refId"] == ref)
    expression = expression.replace("${snapshot_interval}", "15s")
    expression = expression.replace("$__rate_interval", "5m")
    return re.sub(r"\$[a-z_]+", ".*", expression)


def unguarded_expression(expression):
    """Recover the previous plain rollup for a counterexample on the same input."""
    pattern = re.compile(
        r"increase\(\((?P<series>[\w:]+\{[^{}]*\}).*?\)\[(?P<window>[^\[\]]+):15s\]\)"
    )
    return pattern.sub(lambda m: f"increase({m['series']}[{m['window']}])", expression)


def import_samples(vm, records):
    shift = ((int(time.time()) - 1792120800 - 60) // 15) * 15
    for record in records:
        record["timestamps"] = [ts + shift * 1000 for ts in record["timestamps"]]
    payload = "".join(json.dumps(record) + "\n" for record in records).encode()
    request = urllib.request.Request(
        vm + "/api/v1/import",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status == 204
    with urllib.request.urlopen(vm + "/internal/force_flush", timeout=10):
        pass

    def query(expression, timestamp, end=None):
        # Match the dashboard's 15-second range-query step. VM also uses the
        # step when selecting samples preceding the explicit lookbehind window.
        params = {
            "query": expression,
            "start": timestamp + shift,
            "end": (timestamp if end is None else end) + shift,
            "step": "15s",
            "nocache": "1",
        }
        with urllib.request.urlopen(
            vm + "/api/v1/query_range?" + urllib.parse.urlencode(params), timeout=10
        ) as response:
            result = json.load(response)
        assert result["status"] == "success", result
        return [
            float(value)
            for series in result["data"]["result"]
            for _, value in series["values"]
        ]

    return query


def partial_samples():
    return [
        json.loads(line)
        for line in (ROOT / "tests/fixtures/histogram-partial-scrapes.jsonl")
        .read_text()
        .splitlines()
    ]


@pytest.mark.parametrize(
    "filename,panel_id,ref",
    [
        ("video-gateway.json", 4, "A"),
        ("video-gateway.json", 7, "A"),
        ("video-worker-omni.json", 5, "B"),
        ("video-worker-omni.json", 6, "B"),
    ],
)
def test_other_p95_queries_handle_partial_scrapes(vm, filename, panel_id, ref):
    expression = dashboard_expression(filename, panel_id, ref)
    metric = re.search(r"([\w:]+_bucket)\{", expression).group(1)
    records = [r for r in partial_samples() if r["metric"]["stage"] == "audio_decode"]
    for record in records:
        record["metric"].update(
            __name__=metric, workflow="fl2va", model="test-model", operation="kv/txn"
        )
    query = import_samples(vm, records)
    for timestamp in (1792120650, 1792120665):
        assert query(expression, timestamp) == pytest.approx([0.2425])
        assert query(
            unguarded_expression(expression).replace("increase(", "rate("), timestamp
        ) == pytest.approx([1530])
    assert query(expression, 1792121220) == []


@pytest.mark.parametrize(
    "filename,panel_id,ref",
    [
        ("video-gateway.json", 3, "A"),
        ("video-gateway.json", 90, "A"),
        ("video-gateway.json", 92, "A"),
        ("video-worker-omni.json", 5, "A"),
        ("video-worker-omni.json", 6, "A"),
        ("video-worker-omni.json", 7, "A"),
        ("video-worker-omni.json", 8, "A"),
        ("video-worker-omni.json", 9, "A"),
        ("video-worker-omni.json", 9, "B"),
        ("video-worker-omni.json", 11, "A"),
    ],
)
def test_averages_with_partial_scrapes_are_sample_weighted(vm, filename, panel_id, ref):
    expression = dashboard_expression(filename, panel_id, ref)
    # Reproduce the historical five-minute normalization mismatch exactly.
    expression = expression.replace("[1m:15s]", "[5m:15s]")
    metric = re.search(r"([\w:]+)_sum\{", expression).group(1)
    templates = {
        r["metric"]["le"]: r
        for r in partial_samples()
        if r["metric"]["stage"] == "encode_prompt"
    }
    records = []
    unit = 15 if metric.endswith("num_inference_steps") else 0.06
    # Instance 2 has twice as many observations, each three times as expensive.
    # The third group verifies that a genuinely large mean is not clipped.
    for instance, multiplier, mean, group in [
        ("worker-1", 1, unit, "observed-samples"),
        ("worker-2", 2, unit * 3, "observed-samples"),
        ("worker-slow", 1, 480, "true-slow-request"),
    ]:
        for suffix, template, factor in [
            ("sum", "600.0", multiplier * mean),
            ("count", "300.0", multiplier),
        ]:
            record = copy.deepcopy(templates[template])
            del record["metric"]["le"]
            record["metric"].update(
                __name__=metric + "_" + suffix,
                pod=instance,
                instance=instance,
                pool=group,
                workflow=group,
                model="test-model",
            )
            record["values"] = [value * factor for value in record["values"]]
            records.append(record)
    query = import_samples(vm, records)
    expected_mean = unit * 7 / 3
    for timestamp in (1792120410, 1792120425, 1792120710, 1792120725):
        assert sorted(query(expression, timestamp)) == pytest.approx(
            [expected_mean, 480]
        )
    # At 11:13:30 the same count increment spans 270s for sum and 300s for
    # count. The old query falsely raises both means by 11.11 percent.
    assert sorted(
        query(
            unguarded_expression(expression).replace("increase(", "rate("), 1792120410
        )
    ) == pytest.approx([expected_mean * 300 / 270, 480 * 300 / 270])
    # The complete-scrape control keeps the same mean on both query forms.
    assert sorted(query(expression, 1792120440)) == pytest.approx([expected_mean, 480])
    assert sorted(
        query(
            unguarded_expression(expression).replace("increase(", "rate("), 1792120440
        )
    ) == pytest.approx([expected_mean, 480])
    assert query(expression, 1792121220) == []


@pytest.mark.parametrize("window", ["1m", "5m"])
def test_execution_partial_scrape_at_1029_is_not_a_thirty_minute_request(vm, window):
    records = [
        json.loads(line)
        for line in (ROOT / "tests/fixtures/execution-partial-scrapes.jsonl")
        .read_text()
        .splitlines()
    ]
    # Exact exported incident: 17 finite buckets miss 10:28:47.201, +Inf has it.
    # All finite bucket values and +Inf agree again at 10:29:17.201.
    control = []
    for record in records:
        altered = copy.deepcopy(record)
        altered["metric"]["pool"] = "true-slow-request"
        # Turn the affected 109s observation into a real 2400s observation.
        if 120 <= float(altered["metric"]["le"]) < 2400:
            altered["values"] = [
                v - int(ts >= 1792117727201)
                for ts, v in zip(altered["timestamps"], altered["values"])
            ]
        control.append(altered)
    query = import_samples(vm, records + control)
    expression = dashboard_expression("video-gateway.json", 4, "A")
    expression = expression.replace("[5m:15s]", f"[{window}:15s]")
    only_observed = expression.replace('pool=~".*"', 'pool="execution-samples"')
    for timestamp in range(1792117680, 1792117861, 15):
        result = query(only_observed, timestamp)
        assert result == [] or result == pytest.approx([117])
    if window == "1m":
        assert max(
            query(unguarded_expression(only_observed), 1792117680, end=1792117860)
        ) == pytest.approx(1800)
        assert query(only_observed, 1792117740) == []
    # A complete post-gap snapshot restores the real normal and slow values.
    assert sorted(query(expression, 1792117770)) == pytest.approx([117, 1800])
    assert query(expression, 1792118700) == []


def test_model_quiet_window_with_continuing_scrapes_is_empty_until_requests_resume(vm):
    records = [r for r in partial_samples() if r["metric"]["stage"] == "encode_prompt"]
    total = next(r for r in records if r["metric"]["le"] == "+Inf")
    last_observation = max(
        ts
        for ts, previous, value in zip(
            total["timestamps"][1:], total["values"], total["values"][1:]
        )
        if value > previous
    )
    for suffix, factor in [("sum", 0.06), ("count", 1)]:
        record = copy.deepcopy(total)
        record["metric"]["__name__"] = (
            "dingo_video_model_stage_duration_seconds_" + suffix
        )
        del record["metric"]["le"]
        record["values"] = [v * factor for v in record["values"]]
        records.append(record)
    for record in records:
        metric = record["metric"]["__name__"]
        last = record["values"][-1]
        for timestamp in range(1792120817201, 1792122017202, 30000):
            # Metrics remain available during a 15-minute idle period. Later
            # observations take 60ms; this is distinct from exporter failure.
            resumed = int(timestamp >= 1792121717201)
            if metric.endswith("_bucket"):
                delta = resumed if float(record["metric"]["le"]) >= 0.1 else 0
            else:
                delta = resumed * (0.06 if metric.endswith("_sum") else 1)
            record["timestamps"].append(timestamp)
            record["values"].append(last + delta)
    query = import_samples(vm, records)
    mean = dashboard_expression("video-gateway.json", 90, "A")
    p95 = dashboard_expression("video-gateway.json", 91, "A")
    for timestamp in (1792121280, 1792121580):
        assert query(mean, timestamp) == []
        assert query(p95, timestamp) == []
    # Continuing scrapes without observations must not keep a five-minute tail.
    quiet_timestamp = (last_observation // 15000 + 7) * 15
    assert query(mean, quiet_timestamp) == []
    assert query(p95, quiet_timestamp) == []
    assert query(mean.replace("[1m:15s]", "[5m:15s]"), quiet_timestamp)
    assert query(p95.replace("[1m:15s]", "[5m:15s]"), quiet_timestamp)
    assert query(mean, 1792121760) == pytest.approx([0.06])
    assert query(p95, 1792121760) == pytest.approx([0.0975])
    assert query(mean, 1792121805) == []
    assert query(p95, 1792121805) == []
    assert query(mean, 1792122120) == []
    assert query(p95, 1792122120) == []


def test_real_grafana_sparse_frames_preserve_missing_intervals(tmp_path):
    if not os.environ.get("GRAFANA_BUNDLE_DIR") or not shutil.which("node"):
        pytest.skip(
            "set GRAFANA_BUNDLE_DIR and install Node to replay Grafana 12.2.1 frames"
        )
    result = subprocess.run(
        ["node", str(ROOT / "tests/grafana_gap_regression.cjs")],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    (tmp_path / "grafana-gap-validation.json").write_text(result.stdout)


def test_gateway_runtime_cards_distinguish_outages_and_deduplicate_capacity(vm):
    dashboard = json.loads((ROOT / "grafana/video-gateway.json").read_text())
    cards = {p["id"]: p for p in dashboard["panels"] if p["id"] in (1023, 1024, 1025)}
    timestamp = 1792120200
    records = []

    def add(metric, value, **labels):
        records.append(
            {
                "metric": {"__name__": metric, **labels},
                "timestamps": [(timestamp - 60) * 1000, timestamp * 1000],
                "values": [value, value],
            }
        )

    for namespace, statuses in [
        ("healthy", (1, 1)),
        ("partial", (1, 0)),
        ("down", (0, 0)),
    ]:
        for instance, metric, status in zip(
            ("gateway:8000", "worker:8081"),
            ("dingo_video_workers", "dingo_video_worker_detached_execution_capacity"),
            statuses,
        ):
            add(
                "up",
                status,
                namespace=namespace,
                instance=instance,
                job="shared-scraper",
            )
            add(metric, 1, namespace=namespace, instance=instance, pool="fl2va")
        # A successful unrelated scrape must not hide a video-service outage.
        add(
            "up",
            1,
            namespace=namespace,
            instance="unrelated:9000",
            job="shared-scraper",
        )
    # Feature metrics may have several pools for one Gateway scrape target.
    add(
        "dingo_video_workers",
        1,
        namespace="healthy",
        instance="gateway:8000",
        pool="ref2va",
    )

    for deployment, pool, counts in [
        ("d1", "fl2va", (5, 3)),
        ("d1", "ref2va", (2, 2)),
        ("d2", "fl2va", (4, 4)),
    ]:
        for instance, count in zip(("g1", "g2"), counts):
            add(
                "dingo_video_worker_slots_free",
                count,
                namespace="capacity",
                region="test",
                deployment=deployment,
                pool=pool,
                instance=instance,
            )
    gib = 1024**3
    for deployment, remaining in [("d1", (100, 90)), ("d2", (40, 40))]:
        for instance, amount in zip(("g1", "g2"), remaining):
            add(
                "dingo_video_artifact_free_bytes",
                amount * gib,
                namespace="capacity",
                region="test",
                deployment=deployment,
                instance=instance,
            )
            add(
                "dingo_video_artifact_total_bytes",
                (200 if deployment == "d1" else 80) * gib,
                namespace="capacity",
                region="test",
                deployment=deployment,
                instance=instance,
            )
    for deployment, metric, amount in [
        ("missing-free", "total", 999),
        ("missing-total", "free", 999),
        ("zero-total", "total", 0),
        ("zero-total", "free", 9),
    ]:
        add(
            f"dingo_video_artifact_{metric}_bytes",
            amount * gib,
            namespace="capacity",
            region="test",
            deployment=deployment,
            instance="g1",
        )
    add(
        "dingo_video_worker_slots_free",
        999,
        namespace="unselected",
        region="test",
        deployment="d1",
        pool="fl2va",
        instance="g1",
    )
    add(
        "dingo_video_artifact_free_bytes",
        999 * gib,
        namespace="unselected",
        region="test",
        deployment="d1",
        instance="g1",
    )
    query = import_samples(vm, records)

    def evaluate(panel_id, namespace, pool=".*"):
        expression = cards[panel_id]["targets"][0]["expr"]
        return query(
            expression.replace("$namespace", namespace).replace("$pool", pool),
            timestamp,
        )

    for namespace, expected in [("healthy", 2), ("partial", 1), ("down", 0)]:
        assert evaluate(1023, namespace) == [expected]
    assert evaluate(1024, "capacity") == [9]
    assert evaluate(1024, "capacity", "fl2va") == [7]
    assert evaluate(1024, "capacity", "ref2va") == [2]
    assert evaluate(1024, "capacity", "absent") == []
    assert evaluate(1025, "capacity", "absent") == pytest.approx([100 * 130 / 280])
    for panel_id in cards:
        assert evaluate(panel_id, "missing") == []


def test_30_day_timing_queries_stay_under_subquery_point_limit(vm):
    expressions = []
    for filename in ("video-gateway.json", "video-worker-omni.json"):
        doc = json.loads((ROOT / "grafana" / filename).read_text())
        for panel in panels(doc["panels"]):
            for target in panel.get("targets", []):
                expression = target.get("expr", "")
                if "timestamp(" in expression and "increase(" in expression:
                    expression = expression.replace("$__rate_interval", "3615s")
                    expression = re.sub(r"\$[a-z_]+", ".*", expression)
                    expressions.append(
                        (filename, panel["id"], target["refId"], expression)
                    )
    # Match the real Grafana interval calculation checked by the JS regression.
    assert len(expressions) == 16
    metrics = {
        m
        for _, _, _, expression in expressions
        for m in re.findall(r"([\w:]+)\{", expression)
    }
    end = int(time.time()) // 3600 * 3600
    timestamps = [(end - 47) * 1000, (end - 17) * 1000]
    labels = {
        "namespace": "long-range",
        "pool": "fl2va",
        "schema": "test.v1",
        "stage": "encode_prompt",
        "operation": "kv/txn",
        "pod": "worker",
        "model": "test",
        "workflow": "fl2va",
        "region": "test",
        "deployment": "d1",
        "instance": "test:8000",
    }
    records = []
    for metric in metrics:
        bounds = ("0.1", "+Inf") if metric.endswith("_bucket") else (None,)
        for bound in bounds:
            records.append(
                {
                    "metric": {
                        "__name__": metric,
                        **labels,
                        **({"le": bound} if bound else {}),
                    },
                    "timestamps": timestamps,
                    "values": [0.6, 0.66] if metric.endswith("_sum") else [10, 11],
                }
            )
    payload = "".join(json.dumps(r) + "\n" for r in records).encode()
    request = urllib.request.Request(
        vm + "/api/v1/import",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status == 204
    with urllib.request.urlopen(vm + "/internal/force_flush", timeout=10):
        pass

    def query(expression, resolution):
        params = {
            "query": expression.replace("${snapshot_interval}", resolution),
            "start": end - 30 * 86400,
            "end": end,
            "step": 3600,
            "nocache": 1,
        }
        return urllib.request.urlopen(
            vm + "/api/v1/query_range?" + urllib.parse.urlencode(params), timeout=20
        )

    # Lowering outer result resolution cannot repair the old inner 15s limit.
    with pytest.raises(urllib.error.HTTPError) as caught:
        query(expressions[0][3], "15s")
    error = caught.value.read().decode()
    assert "too many points" in error and "maxPointsSubqueryPerTimeseries" in error
    for filename, panel_id, ref, expression in expressions:
        with query(expression, "30s") as response:
            result = json.load(response)
        assert result["status"] == "success", (filename, panel_id, ref, result)
        values = [
            float(value)
            for series in result["data"]["result"]
            for _, value in series["values"]
        ]
        assert values and all(math.isfinite(v) for v in values), (
            filename,
            panel_id,
            ref,
            result,
        )

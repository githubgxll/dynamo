# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable Prometheus dashboard exports for Video Gateway and Worker."""

import json
import re
from pathlib import Path

import pytest

GRAFANA = Path(__file__).resolve().parents[1] / "grafana"


def panels(items):
    for panel in items:
        yield panel
        yield from panels(panel.get("panels", []))


@pytest.mark.parametrize(
    "filename,uid,title,peer",
    [
        (
            "video-gateway.json",
            "dingo-video-overview",
            "Video Gateway",
            "dingo-video-worker",
        ),
        (
            "video-worker-omni.json",
            "dingo-video-worker",
            "Video Worker and Omni",
            "dingo-video-overview",
        ),
    ],
)
def test_portable_runtime_dashboards(filename, uid, title, peer):
    raw = (GRAFANA / filename).read_text()
    doc = json.loads(raw)
    assert doc["uid"] == uid and doc["title"] == title
    assert doc.get("id") is None and "folderUid" not in doc
    unescaped = raw.replace("\\", "")
    for value in (
        "elm-test",
        "token-factory",
        "hd04",
        "cfew29tdmmi9sc",
        "10.201.",
        "172.16.",
    ):
        assert value not in unescaped
    variables = {v["name"]: v for v in doc["templating"]["list"]}
    source = variables["data_source"]
    assert source["type"] == "datasource" and source["query"] == "prometheus"
    assert source["current"] == {} and source["options"] == []
    namespace = variables["namespace"]
    assert namespace["type"] == "query" and namespace["includeAll"]
    assert namespace["datasource"]["uid"] == "${data_source}"
    snapshot = variables["snapshot_interval"]
    assert snapshot["type"] == "interval" and snapshot["hide"] == 2
    assert snapshot["auto"] and snapshot["auto_count"] == 60000
    assert snapshot["auto_min"] == "15s" and snapshot["skipUrlSync"]
    assert any("/d/" + peer in link["url"] for link in doc["links"])
    assert all("dingo-video-performance" not in link["url"] for link in doc["links"])
    items = list(panels(doc["panels"]))
    assert len({p["id"] for p in items}) == len(items)
    for panel in items:
        for target in panel.get("targets", []):
            assert "rawSql" not in target
            assert not re.search(r"\bjob\s*(?:=|=~)", target.get("expr", ""))
        if panel.get("datasource"):
            assert panel["datasource"] == {
                "type": "prometheus",
                "uid": "${data_source}",
            }


def test_gateway_latest_controls_and_diagnostics_are_retained():
    doc = json.loads((GRAFANA / "video-gateway.json").read_text())
    titles = {p["title"] for p in panels(doc["panels"])}
    assert {"每池请求数（时段内累计）", "本机后处理积压", "Worker 执行 / 预取等待"} <= titles
    assert any("模型阶段 P95" in title for title in titles)
    for panel in panels(doc["panels"]):
        if panel["id"] in (90, 91, 92):
            assert "此前1分钟" in panel["description"]
            for target in panel["targets"]:
                if "dingo_video_model_stage_duration_seconds" in target.get("expr", ""):
                    assert "[1m:${snapshot_interval}]" in target["expr"]
                    assert "[5m" not in target["expr"]
    assert {"在途请求数", "Gateway 排队", "Worker 侧请求"} <= titles
    current = {
        p["id"]: p for p in panels(doc["panels"]) if p["id"] in (1020, 1021, 1022)
    }
    assert set(current) == {1020, 1021, 1022}
    for index, panel in enumerate(current.values()):
        assert panel["type"] == "stat"
        assert panel["gridPos"] == {"x": index * 4, "y": 0, "w": 4, "h": 4}
        assert panel["options"]["graphMode"] == "area"
        assert panel["options"]["textMode"] == "value"
        assert panel["options"]["text"]["valueSize"] >= 72
        assert panel["options"]["reduceOptions"]["calcs"] == ["lastNotNull"]
        assert panel["fieldConfig"]["defaults"]["decimals"] == 0
        assert panel["fieldConfig"]["defaults"]["noValue"] == "No data"
        assert len(panel["targets"]) == 1
        target = panel["targets"][0]
        assert target["range"] and target["instant"] is False
        assert "dingo_video_tasks{" in target["expr"]
        assert "max by(region,namespace,deployment,pool,status)" in target["expr"]
        assert "$__range" not in target["expr"] and "increase(" not in target["expr"]
    assert 'status="queued"' in current[1021]["targets"][0]["expr"]
    assert (
        'status=~"dispatching|in_progress|finalizing"'
        in current[1022]["targets"][0]["expr"]
    )
    assert (
        'status=~"queued|dispatching|in_progress|finalizing"'
        in current[1020]["targets"][0]["expr"]
    )
    runtime = {p["id"]: p for p in doc["panels"] if p["id"] in (1023, 1024, 1025)}
    assert [p["title"] for p in runtime.values()] == ["空闲槽位", "制品余量", "采集状态"]
    for index, panel in enumerate(runtime.values()):
        assert panel["type"] == "stat"
        assert panel["gridPos"] == {"x": 12 + index * 4, "y": 0, "w": 4, "h": 4}
        assert panel["fieldConfig"]["defaults"]["noValue"] == "No data"
        assert "or vector(0)" not in panel["targets"][0]["expr"]
    health = runtime[1023]
    assert health["targets"][0]["instant"] and not health["targets"][0]["range"]
    assert "$pool" not in health["targets"][0]["expr"]
    assert health["options"]["graphMode"] == "none"
    mappings = health["fieldConfig"]["defaults"]["mappings"][0]["options"]
    assert {k: v["text"] for k, v in mappings.items()} == {
        "0": "全部异常",
        "1": "部分异常",
        "2": "正常",
    }
    assert "$pool" in runtime[1024]["targets"][0]["expr"]
    assert "$pool" not in runtime[1025]["targets"][0]["expr"]
    assert runtime[1025]["fieldConfig"]["defaults"]["unit"] == "percent"
    assert runtime[1025]["fieldConfig"]["defaults"]["decimals"] == 0
    assert runtime[1025]["fieldConfig"]["defaults"]["max"] == 100
    for panel_id in (1024, 1025):
        assert runtime[panel_id]["options"]["graphMode"] == "area"
        assert (
            runtime[panel_id]["options"]["text"]["valueSize"]
            == current[1020]["options"]["text"]["valueSize"]
        )
        assert (
            "sum(min by(region,namespace,deployment"
            in runtime[panel_id]["targets"][0]["expr"]
        )
    events = [p for p in doc["panels"] if p["id"] in range(1009, 1015)]
    assert len(events) == 6
    assert all(p["gridPos"]["y"] == 4 for p in events)
    # The existing three-column business curves remain aligned below the cards.
    curves = {p["id"]: p for p in doc["panels"] if p["id"] in (1, 1006, 5)}
    assert [curves[i]["gridPos"] for i in (1, 1006, 5)] == [
        {"x": x, "y": 17, "w": 8, "h": 8} for x in (0, 8, 16)
    ]
    unit = next(v for v in doc["templating"]["list"] if v["name"] == "throughput_unit")
    assert unit["current"]["value"] == "60"
    assert "example" in json.dumps(doc)


def test_worker_gpu_power_and_resource_scope_are_retained():
    doc = json.loads((GRAFANA / "video-worker-omni.json").read_text())
    assert "vllm-omni" in doc["tags"]
    resources = [
        p
        for p in panels(doc["panels"])
        if any(w in p["title"] for w in ("GPU ", "CPU ", "容器内存", "容器重启"))
    ]
    assert len(resources) == 6
    power = next(p for p in resources if "GPU 功率" in p["title"])
    assert "DCGM_FI_DEV_POWER_USAGE" in power["targets"][0]["expr"]
    for panel in resources:
        for target in panel["targets"]:
            assert "dingo_video_worker_detached_execution_capacity" in target["expr"]
            assert "namespace" in target["expr"] and "pod" in target["expr"]

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

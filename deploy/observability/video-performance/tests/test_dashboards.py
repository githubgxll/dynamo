# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest

GRAFANA = Path(__file__).resolve().parents[1] / "grafana"


@pytest.mark.parametrize(
    "filename,args",
    [("performance.json", []), ("performance-advanced.json", ["--advanced"])],
)
def test_generated_dashboards_match_exported_portable_json(filename, args):
    generated = json.loads(
        subprocess.check_output(
            [sys.executable, str(GRAFANA / "build_dashboards.py"), *args]
        )
    )
    exported = json.loads((GRAFANA / filename).read_text())
    assert generated == exported
    heartbeat = next(panel for panel in exported["panels"] if panel["id"] == 6)
    assert heartbeat["title"] == "采集器心跳距今（秒）"
    assert "不是最新性能样本距今时间" in heartbeat["description"]
    assert "last_poll" in heartbeat["targets"][0]["rawSql"]
    minute = next(panel for panel in exported["panels"] if panel["id"] == 107)
    assert minute["title"] == "每分钟完成数" and minute["type"] == "timeseries"
    assert minute["gridPos"] == {"x": 0, "y": 4, "w": 24, "h": 7}
    sql = minute["targets"][0]["rawSql"]
    assert "date_trunc('minute', completed_at)" in sql
    assert "generate_series" in sql and "coalesce(completed.completed_count, 0)" in sql
    assert "count(*)" in sql and "status='completed'" in sql
    assert "${pool:sqlstring}" in sql and "${seconds:sqlstring}" in sql
    assert minute["fieldConfig"]["defaults"]["decimals"] == 0
    for other in exported["panels"]:
        if other["id"] == minute["id"]:
            continue
        a, b = minute["gridPos"], other["gridPos"]
        assert not (
            a["x"] < b["x"] + b["w"]
            and b["x"] < a["x"] + a["w"]
            and a["y"] < b["y"] + b["h"]
            and b["y"] < a["y"] + a["h"]
        ), other["title"]
    pg = next(
        v
        for v in exported["templating"]["list"]
        if v["name"] == "performance_data_source"
    )
    assert pg["query"] == "grafana-postgresql-datasource"
    assert pg.get("hide", 0) == 0 and pg["regex"] == ""
    assert not any(v["name"] == "data_source" for v in exported["templating"]["list"])
    assert "${data_source" not in json.dumps(exported)
    assert all("dingo-video-performance" in link["url"] for link in exported["links"])
    assert all(
        "${performance_data_source:queryparam}" in link["url"]
        for link in exported["links"]
    )
    assert len({panel["id"] for panel in exported["panels"]}) == len(exported["panels"])
    assert all(
        variable["current"] == {}
        for variable in exported["templating"]["list"]
        if variable["type"] == "datasource"
    )
    serialized = json.dumps(exported)
    assert "hd04" not in serialized and "elm-test" not in serialized
    status = next(panel for panel in exported["panels"] if panel["id"] == 106)[
        "targets"
    ][0]["rawSql"]
    assert "空间预算不足" in status and "未配置 namespace" in status

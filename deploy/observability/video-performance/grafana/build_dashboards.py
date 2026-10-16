# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print the Grafana dashboard JSON; stages and input categories come from SQL."""
import json
import sys
from pathlib import Path


def datasource_variable():
    return {
        "name": "performance_data_source",
        "label": "Data source",
        "type": "datasource",
        "query": "grafana-postgresql-datasource",
        "regex": "",
        "refresh": 1,
        "multi": False,
        "includeAll": False,
        "options": [],
        "current": {},
    }


DS = {"type": "grafana-postgresql-datasource", "uid": "${performance_data_source}"}
variables = [datasource_variable()]
panels = []
common = {
    "performance_data_source",
    "namespace",
    "pool",
    "seconds",
    "steps",
    "resolution",
    "input_type",
    "sound",
    "image_count",
    "video_count",
    "audio_count",
    "stage",
}
count_dimensions = {"image_count", "video_count", "audio_count"}
numeric_dimensions = count_dimensions | {"seconds", "steps"}
advanced_conditions = []
validation_errors = []


def variable(name, label, column):
    expression = f"coalesce({column}::text,'unknown')"
    numeric_sort = (
        f"{column}::numeric" if name in numeric_dimensions else "NULL::numeric"
    )
    namespace_scope = (
        ""
        if name == "namespace"
        else " AND ('__all__' IN (${namespace:sqlstring}) OR namespace IN (${namespace:sqlstring}))"
    )
    choices = f"SELECT DISTINCT {expression} AS __text, {expression} AS __value,2 AS option_rank,{numeric_sort} AS numeric_sort FROM video_performance WHERE $__timeFilter(completed_at){namespace_scope} UNION SELECT 'All','__all__',0,NULL"
    if name in count_dimensions:
        choices += " UNION SELECT '至少 1 个（有引用）','__present__',1,NULL"
    query = f"SELECT __text,__value FROM ({choices}) choices ORDER BY option_rank,numeric_sort NULLS LAST,__text"
    variables.append(
        {
            "name": name,
            "label": label,
            "type": "query",
            "datasource": DS,
            "query": query,
            "definition": query,
            "refresh": 2,
            "sort": 0,
            "multi": True,
            "includeAll": False,
            "current": {"text": ["All"], "value": ["__all__"], "selected": True},
            "options": [],
        }
    )


dimensions = [
    ("namespace", "Namespace", "namespace"),
    ("pool", "Pool", "pool"),
    ("model", "模型", "model"),
    ("seconds", "输出秒数", "seconds"),
    ("steps", "去噪/推理步数", "steps"),
    ("resolution", "输出分辨率", "width::text||'x'||height::text"),
    ("input_type", "引用组合", "input_type"),
    ("sound", "保留声音", "sound"),
    ("image_count", "图片数量", "image_count"),
    ("video_count", "视频数量", "video_count"),
    ("audio_count", "音频数量", "audio_count"),
    ("schema", "计时 Schema", "schema_id"),
    ("config", "Gateway配置版本", "configuration_revision"),
    ("environment", "环境版本（当前未归因）", "environment_revision"),
    ("status", "结果状态", "status"),
]
for name, label, column in dimensions:
    variable(name, label, column)
where = "$__timeFilter(completed_at)"
for name, label, column in dimensions:
    present = (
        f" OR ('__present__' IN (${{{name}:sqlstring}}) AND {column} >= 1)"
        if name in count_dimensions
        else ""
    )
    where += f" AND ('__all__' IN (${{{name}:sqlstring}}) OR coalesce(({column})::text,'unknown') IN (${{{name}:sqlstring}}){present})"
    if name not in common:
        advanced_conditions.append(
            f"CASE WHEN '__all__' NOT IN (${{{name}:sqlstring}}) THEN 1 ELSE 0 END"
        )


def number_value(key, integer=False):
    text = f"btrim(${{{key}:sqlstring}})"
    pattern = "^[0-9]+$" if integer else "^([0-9]+([.][0-9]*)?|[.][0-9]+)$"
    valid = f"(length({text}) <= 32 AND {text} ~ '{pattern}')"
    # CASE protects the cast even when PostgreSQL reorders WHERE predicates.
    value = f"(CASE WHEN {valid} THEN {text}::numeric ELSE NULL END)"
    return text, valid, value


for name, label, column, scale, integer in (
    ("image_count", "图片数量", "image_count", 1, True),
    ("video_count", "视频数量", "video_count", 1, True),
    ("audio_count", "音频数量", "audio_count", 1, True),
    ("image_mp", "图片总像素(MP)", "image_pixels", 1000000, False),
    ("image_mb", "图片总大小(MiB)", "image_bytes", 1048576, False),
    ("video_s", "参考视频总秒数", "video_seconds", 1, False),
    ("video_frames", "参考视频实际总帧数", "video_frames", 1, True),
    ("video_mb", "视频总大小(MiB)", "video_bytes", 1048576, False),
    ("audio_s", "参考音频总秒数", "audio_seconds", 1, False),
):
    bounds = []
    for suffix, word, op in (("min", "下限", ">="), ("max", "上限", "<=")):
        key = f"{name}_{suffix}"
        variables.append(
            {
                "name": key,
                "label": label + word,
                "type": "textbox",
                "query": "",
                "current": {"text": "", "value": ""},
                "options": [],
            }
        )
        text, valid, value = number_value(key, integer)
        bounds.append((text, value))
        where += f" AND ({text} = '' OR ({valid} AND {column}::numeric/{scale} {op} {value}))"
        expected = "非负整数" if integer else "非负数字"
        validation_errors.append(
            f"CASE WHEN {text} <> '' AND NOT {valid} THEN '{label}{word}：请输入{expected}（最多32字符）' END"
        )
        advanced_conditions.append(f"CASE WHEN {text} <> '' THEN 1 ELSE 0 END")
    validation_errors.append(
        f"CASE WHEN {bounds[0][1]} > {bounds[1][1]} THEN '{label}：下限不能大于上限' END"
    )
variables.append(
    {
        "name": "stage",
        "label": "模型阶段",
        "type": "query",
        "datasource": DS,
        "query": f"SELECT DISTINCT stage AS __text, stage AS __value FROM video_performance_stages WHERE {where} UNION SELECT 'All','__all__' ORDER BY 1",
        "refresh": 2,
        "multi": True,
        "includeAll": False,
        "current": {"text": ["All"], "value": ["__all__"], "selected": True},
        "options": [],
    }
)
variables.append(
    {
        "name": "group_by",
        "label": "对比维度",
        "type": "custom",
        "query": "输出时长 : seconds,去噪/推理步数 : steps,输出分辨率 : resolution,引用组合 : input_type,图片数量 : image_count,视频数量 : video_count,参考视频总时长 : video_seconds,参考视频总帧数 : video_frames",
        "current": {"text": "输出时长", "value": "seconds"},
        "options": [],
    }
)
stage_where = (
    where + " AND ('__all__' IN (${stage:sqlstring}) OR stage IN (${stage:sqlstring}))"
)
group = (
    "CASE ${group_by:sqlstring} "
    + " ".join(
        f"WHEN '{key}' THEN coalesce(({column})::text,'unknown')"
        for key, column in (
            ("seconds", "seconds"),
            ("steps", "steps"),
            ("resolution", "width::text||'x'||height::text"),
            ("input_type", "input_type"),
            ("image_count", "image_count"),
            ("video_count", "video_count"),
            ("video_seconds", "video_seconds"),
            ("video_frames", "video_frames"),
        )
    )
    + " ELSE 'unknown' END"
)
numeric_group = (
    "CASE ${group_by:sqlstring} "
    + " ".join(
        f"WHEN '{key}' THEN {column}::numeric"
        for key, column in (
            ("seconds", "seconds"),
            ("steps", "steps"),
            ("image_count", "image_count"),
            ("video_count", "video_count"),
            ("video_seconds", "video_seconds"),
            ("video_frames", "video_frames"),
        )
    )
    + " ELSE NULL END"
)
group_order = f"min({numeric_group}) NULLS LAST,parameter"
stage_metadata = json.loads(Path(__file__).with_name("stage_metadata.json").read_text())
parent_cases = []
stage_help = []
for schema, stages in stage_metadata.items():
    for stage, definition in stages.items():
        parent = definition.get("parent")
        if parent:
            parent_cases.append(
                f"WHEN schema_id='{schema}' AND stage='{stage}' THEN '{stage}（属于 {parent}）'"
            )
        stage_help.append(
            f"{schema} / {stage}: {definition['description']}"
            + (f" Parent: {parent}." if parent else "")
        )
stage_display = "CASE " + " ".join(parent_cases) + " ELSE stage END"
small_sample = "CASE WHEN count(*) < 20 THEN '小样本，仅供参考' ELSE '' END AS sample_note"


def panel(title, kind, sql, x, y, w=12, h=8, unit="s", description=""):
    p = {
        "id": len(panels) + 1,
        "title": title,
        "type": kind,
        "datasource": DS,
        "description": description,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [
            {
                "refId": "A",
                "datasource": DS,
                "rawQuery": True,
                "editorMode": "code",
                "rawSql": sql,
                "format": "time_series" if kind == "timeseries" else "table",
            }
        ],
        "fieldConfig": {"defaults": {"unit": unit, "decimals": 3}, "overrides": []},
        "options": {},
    }
    if kind == "stat":
        p["options"] = {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "value",
            "graphMode": "none",
        }
    if kind == "timeseries":
        p["fieldConfig"]["defaults"]["custom"] = {
            "drawStyle": "line",
            "lineWidth": 1,
            "showPoints": "auto",
            "spanNulls": False,
        }
        p["options"] = {
            "legend": {
                "displayMode": "table",
                "placement": "bottom",
                "calcs": ["lastNotNull"],
            },
            "tooltip": {"mode": "multi"},
        }
    panels.append(p)
    return p


for i, (title, expression) in enumerate(
    [
        ("匹配请求数", "count(*)"),
        ("成功", "count(*) FILTER (WHERE status='completed')"),
        ("取消", "count(*) FILTER (WHERE status='cancelled')"),
        ("失败", "count(*) FILTER (WHERE status='failed')"),
        (
            "有模型计时样本",
            "count(*) FILTER (WHERE status='completed' AND stages <> '{}'::jsonb)",
        ),
        (
            "采集器心跳距今（秒）",
            "extract(epoch FROM now()-(SELECT last_poll FROM performance_collector_health WHERE id=1))",
        ),
    ]
):
    panel(
        title,
        "stat",
        f"SELECT {expression} AS value FROM video_performance WHERE {where}",
        i * 4,
        0,
        4,
        4,
        "short" if i != 5 else "s",
    )
success = where + " AND status='completed'"
panels[5]["targets"][0][
    "rawSql"
] = "SELECT extract(epoch FROM now()-last_poll) AS seconds FROM performance_collector_health WHERE id=1"
panels[5]["description"] = (
    "距采集器最近一次更新健康状态的秒数，不是最新性能样本距今时间；" "没有新任务时仍会更新，数值较小不代表各 Gateway 均采集成功。"
)
panel(
    "请求耗时概况（秒）",
    "table",
    f"SELECT pool,count(*) AS samples,avg(e2e_s) AS e2e_mean,percentile_cont(0.95) WITHIN GROUP(ORDER BY e2e_s) AS e2e_p95,avg(execution_s) AS execution_mean,percentile_cont(0.95) WITHIN GROUP(ORDER BY execution_s) AS execution_p95,avg(queue_s) AS gateway_queue_mean,avg(worker_queue_s) AS worker_queue_mean,avg(finalize_s) AS finalize_mean,{small_sample} FROM video_performance WHERE {success} GROUP BY pool",
    0,
    4,
    24,
    6,
    "none",
)
panel(
    "条件分组对比（不同配置分别统计）",
    "table",
    f"SELECT pool,{group} AS parameter,count(*) AS samples,avg(execution_s) AS execution_mean,percentile_cont(0.95) WITHIN GROUP(ORDER BY execution_s) AS execution_p95,avg(finalize_s) AS finalize_mean,avg(e2e_s) AS e2e_mean,{small_sample},configuration_revision,environment_revision FROM video_performance WHERE {success} GROUP BY pool,parameter,configuration_revision,environment_revision ORDER BY pool,{group_order}",
    0,
    10,
    24,
    9,
    "none",
    "先固定步数/分辨率/引用类型，再按输出秒数等参数对比。小样本不代表稳定性能。",
)
panel(
    "各阶段分组对比（动态阶段，秒）",
    "table",
    f"SELECT pool,{group} AS parameter,schema_id,{stage_display} AS stage_label,count(*) AS samples,avg(stage_s) AS mean_s,percentile_cont(0.50) WITHIN GROUP(ORDER BY stage_s) AS p50_s,percentile_cont(0.95) WITHIN GROUP(ORDER BY stage_s) AS p95_s,max(stage_s) AS max_s,{small_sample},configuration_revision,environment_revision FROM video_performance_stages WHERE {stage_where} AND status='completed' GROUP BY pool,parameter,schema_id,stage,configuration_revision,environment_revision ORDER BY pool,{group_order},stage",
    0,
    19,
    24,
    10,
    "none",
    "精确任务样本的分位数。decode 与子阶段可能重叠，不相加；缺失不补零。",
)
panel(
    "阶段均值趋势（完成时间，1分钟窗口）",
    "timeseries",
    f"SELECT $__timeGroupAlias(completed_at,'1m'),pool||' / '||schema_id||' / '||({stage_display}) AS metric,avg(stage_s) AS value FROM video_performance_stages WHERE {stage_where} AND status='completed' GROUP BY 1,2 ORDER BY 1",
    0,
    29,
)
panel(
    "阶段 P95 趋势（完成时间，1分钟窗口）",
    "timeseries",
    f"SELECT $__timeGroupAlias(completed_at,'1m'),pool||' / '||schema_id||' / '||({stage_display}) AS metric,percentile_cont(0.95) WITHIN GROUP(ORDER BY stage_s) AS value FROM video_performance_stages WHERE {stage_where} AND status='completed' GROUP BY 1,2 ORDER BY 1",
    12,
    29,
)
panel(
    "慢任务明细（最多200条）",
    "table",
    f"SELECT completed_at,task_id,pool,status,seconds,width::text||'x'||height::text AS resolution,steps,input_type,image_count,video_count,audio_count,e2e_s,queue_s,worker_queue_s,execution_s,finalize_s,details->'performance'->'references' AS references,stages,configuration_revision,environment_revision FROM video_performance WHERE {where} ORDER BY e2e_s DESC NULLS LAST LIMIT 200",
    0,
    37,
    24,
    12,
    "none",
)
variables.append(
    {
        "name": "task_id",
        "label": "单任务 ID（明细筛选）",
        "type": "textbox",
        "query": "",
        "current": {"text": "", "value": ""},
        "options": [],
    }
)
panels[-1]["targets"][0]["rawSql"] = panels[-1]["targets"][0]["rawSql"].replace(
    " ORDER BY e2e_s",
    " AND (${task_id:sqlstring}='' OR task_id=${task_id:sqlstring}) ORDER BY e2e_s",
)
panel(
    "完整单任务记录",
    "table",
    f"SELECT completed_at,task_id,attempt,status,details,stages,(SELECT snapshot FROM performance_environments e WHERE e.revision=video_performance.environment_revision) AS worker_configuration FROM video_performance WHERE {where} AND (${{task_id:sqlstring}}='' OR task_id=${{task_id:sqlstring}}) ORDER BY completed_at DESC LIMIT 20",
    0,
    49,
    24,
    8,
    "none",
)
advanced = "--advanced" in sys.argv
if not advanced:
    for variable in variables:
        if variable["name"] not in common:
            variable["hide"] = 2
for variable in variables:
    if variable["name"] == "group_by":
        variable.update(label="分组方式", hide=2)
panels[-2]["fieldConfig"]["overrides"].append(
    {
        "matcher": {"id": "byName", "options": "task_id"},
        "properties": [
            {
                "id": "links",
                "value": [
                    {
                        "title": "查看单任务输入与阶段明细",
                        "url": "/d/dingo-video-performance-advanced?${__url_time_range}&var-task_id=${__value.raw}&${performance_data_source:queryparam}",
                        "targetBlank": True,
                    }
                ],
            }
        ],
    }
)
uid = "dingo-video-performance-advanced" if advanced else "dingo-video-performance"
title = "Video 性能分析（高级筛选）" if advanced else "Video 性能分析"
links = [
    {
        "title": "返回常用视图" if advanced else "高级筛选 / 文件大小与参考时长",
        "type": "link",
        "url": "/d/dingo-video-performance"
        if advanced
        else "/d/dingo-video-performance-advanced",
        "includeVars": True,
        "keepTime": True,
    },
]
reset = "&".join(
    "var-"
    + v["name"]
    + "="
    + (
        ""
        if v["type"] == "textbox"
        else "seconds"
        if v["name"] == "group_by"
        else "__all__"
    )
    for v in variables
    if v["type"] != "datasource"
)
reset += "&${performance_data_source:queryparam}"
links.append(
    {
        "title": "重置全部筛选",
        "type": "link",
        "url": f"/d/{uid}?{reset}",
        "includeVars": False,
        "keepTime": True,
    }
)
for link in links:
    if "${performance_data_source:queryparam}" not in link["url"]:
        link["url"] += (
            "&" if "?" in link["url"] else "?"
        ) + "${performance_data_source:queryparam}"


# Keep filters in Grafana's native top bar; put grouping controls next to comparisons.
def row_section(panel_id, name, y):
    return {
        "id": panel_id,
        "type": "row",
        "title": name,
        "collapsed": False,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "panels": [],
    }


def note(panel_id, title, content, y, height=2):
    return {
        "id": panel_id,
        "type": "text",
        "title": title,
        "transparent": True,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": height},
        "options": {"mode": "markdown", "content": content},
    }


items = {p["id"]: p for p in panels}
for i in range(1, 7):
    items[i]["gridPos"].update(y=1, h=3)
    items[i]["fieldConfig"]["defaults"]["decimals"] = 0 if i != 6 else 1
items[7]["gridPos"].update(y=4, h=5)
items[8]["gridPos"].update(y=11, h=5)
items[9]["gridPos"].update(y=16, h=7)
items[10]["gridPos"].update(y=24, h=7)
items[11]["gridPos"].update(y=24, h=7)
items[12]["gridPos"].update(y=32, h=7)
items[13]["gridPos"].update(y=39, h=6)
items[7]["description"] = "上方条件筛选参与统计的任务；All 不限制。模型阶段只筛选阶段图表。时间范围按终态记录时间。"
items[8]["title"] = "请求耗时分组对比"
items[9]["title"] = "模型阶段分组对比"
items[12]["title"] = "任务耗时明细（最慢优先）"
for item in items.values():
    if item["type"] == "table":
        item["fieldConfig"]["overrides"].append(
            {
                "matcher": {"id": "byName", "options": "samples"},
                "properties": [{"id": "decimals", "value": 0}],
            }
        )
        item["fieldConfig"]["overrides"].append(
            {
                "matcher": {"id": "byName", "options": "parameter"},
                "properties": [{"id": "displayName", "value": "${group_by:raw}"}],
            }
        )
        names = {
            field: field + " (s)"
            for field in (
                "e2e_mean",
                "e2e_p95",
                "execution_mean",
                "execution_p95",
                "gateway_queue_mean",
                "worker_queue_mean",
                "finalize_mean",
            )
        }
        names["stage_label"] = "stage"
        for field, label in names.items():
            item["fieldConfig"]["overrides"].append(
                {
                    "matcher": {"id": "byName", "options": field},
                    "properties": [{"id": "displayName", "value": label}],
                }
            )
sample_help = "样本少于20条仅作提示，不是置信度保证；P95保留实际计算结果。统计包含符合条件的预热和正式请求，当前没有预热标记，不能自动排除。服务端总耗时不含请求上传、最终终态写入RPC及客户端下载。"
for panel_id in (1, 5, 7, 8, 9, 10, 11):
    items[panel_id]["description"] = (
        items[panel_id].get("description", "") + "\n\n" + sample_help
    ).strip()
items[9]["description"] += "\n\n阶段后标注所属父阶段；父子阶段不能相加。\n" + "\n".join(stage_help)
for panel_id in (10, 11):
    items[panel_id]["description"] += "\n按任务完成时间分为1分钟窗口，单个窗口可能只有1条样本。括号标注父阶段，不能重复相加。"
if not advanced:
    items[12]["transformations"] = [
        {
            "id": "filterFieldsByName",
            "options": {"exclude": {"names": ["references", "stages"]}},
        }
    ]
group_options = [
    ("seconds", "输出时长"),
    ("steps", "去噪/推理步数"),
    ("resolution", "输出分辨率"),
    ("input_type", "引用组合"),
    ("image_count", "图片数量"),
    ("video_count", "视频数量"),
    ("video_seconds", "参考视频总时长"),
    ("video_frames", "参考视频总帧数"),
]
# Carry each filter explicitly, excluding group_by so the URL has one grouping value.
carry_filters = "&".join(
    "${" + v["name"] + ":queryparam}" for v in variables if v["name"] != "group_by"
)
switch_links = " · ".join(
    f"[{label}](/d/{uid}?{carry_filters}&var-group_by={key}&${{__url_time_range}})"
    for key, label in group_options
)
group_note = note(102, "", "**分组：${group_by:text}**　|　" + switch_links, 9)
panels = [
    row_section(101, "概览", 0),
    *(items[i] for i in range(1, 8)),
    row_section(103, "分组耗时", 8),
    group_note,
    items[8],
    items[9],
    row_section(105, "阶段趋势", 23),
    items[10],
    items[11],
    row_section(104, "任务明细", 31),
    items[12],
]
if advanced:
    panels.append(items[13])

# One compact status line explains empty results and hidden filters on either page.
active_count = " + ".join(
    advanced_conditions
    + ["CASE WHEN btrim(${task_id:sqlstring}) <> '' THEN 1 ELSE 0 END"]
)
error = "coalesce(" + ",".join(validation_errors) + ",'')"
status_sql = f"""WITH filtered AS (SELECT count(*) AS n FROM video_performance WHERE {where}),
recent AS (SELECT max(completed_at) AS latest FROM video_performance),
validation AS (SELECT {error} AS error),
health AS (SELECT last_poll,last_error,
  details->'storage'->>'paused' AS storage_paused,
  jsonb_array_length(coalesce(details->'namespaces','[]'::jsonb)) AS namespace_count,
  greatest(60,coalesce((details->>'poll_interval_s')::integer,15)*4) AS stale_after_s
  FROM performance_collector_health WHERE id=1)
SELECT concat_ws(' ｜ ',
CASE WHEN validation.error <> '' THEN '输入错误：' || validation.error
     WHEN health.last_poll IS NULL OR health.last_error IS NOT NULL OR health.last_poll < now()-health.stale_after_s*interval '1 second'
     THEN '采集异常：数据可能不完整'
     WHEN health.storage_paused='true' THEN '采集暂停：空间预算不足'
     WHEN health.namespace_count=0 THEN '采集暂停：未配置 namespace'
     WHEN filtered.n=0 THEN '无匹配样本：检查时间范围与筛选条件'
     ELSE '匹配 ' || filtered.n || ' 条 · 采集正常' END,
'最新采集样本 ' || coalesce(to_char(recent.latest AT TIME ZONE 'Asia/Shanghai','MM-DD HH24:MI:SS'),'暂无'),
'高级/明细条件 ' || ({active_count})::text || ' 项（可重置）') AS status
FROM filtered CROSS JOIN recent CROSS JOIN validation LEFT JOIN health ON true"""
status_panel = {
    "id": 106,
    "title": "",
    "type": "stat",
    "transparent": True,
    "datasource": DS,
    "description": "最新采集样本时间针对全部已采集任务（北京时间）。高级条件返回常用页后仍保留，可从顶部重置全部筛选。范围输入只接受非负数字，数量和帧数要求整数。无匹配样本不代表采集故障。",
    "gridPos": {"x": 0, "y": 4, "w": 24, "h": 1},
    "targets": [
        {
            "refId": "A",
            "datasource": DS,
            "rawQuery": True,
            "editorMode": "code",
            "rawSql": status_sql,
            "format": "table",
        }
    ],
    "fieldConfig": {
        "defaults": {
            "unit": "string",
            "color": {"mode": "fixed", "fixedColor": "text"},
        },
        "overrides": [],
    },
    "options": {
        "reduceOptions": {
            "calcs": ["lastNotNull"],
            "fields": "/^status$/",
            "values": False,
        },
        "textMode": "value",
        "graphMode": "none",
        "colorMode": "value",
        "text": {"valueSize": 14},
    },
}
for item in panels:
    if item["gridPos"]["y"] >= 8:
        item["gridPos"]["y"] += 1
    if item["gridPos"]["y"] >= 4:
        item["gridPos"]["y"] += 1
panels.insert(7, status_panel)

# Fixed natural-minute counts, not a rolling window or an extrapolated rate.
minute_sql = f"""WITH minutes AS (
  SELECT generate_series(
    date_trunc('minute', $__timeFrom()::timestamptz),
    date_trunc('minute', $__timeTo()::timestamptz),
    interval '1 minute') AS time
), completed AS (
  SELECT date_trunc('minute', completed_at) AS time, count(*) AS completed_count
  FROM video_performance WHERE {where} AND status='completed'
  GROUP BY 1
)
SELECT minutes.time AS time,
  coalesce(completed.completed_count, 0)::bigint AS completed_count
FROM minutes LEFT JOIN completed USING (time) ORDER BY minutes.time"""
minute_panel = panel(
    "每分钟完成数",
    "timeseries",
    minute_sql,
    0,
    4,
    24,
    7,
    "short",
    "按任务完成时间统计每个自然分钟内成功完成的任务数，例如 12:00:00 至 12:00:59；"
    "不是向前滚动1分钟，也不是每秒速率。沿用页面任务筛选条件，无完成记录的分钟补0。"
    "首尾分钟只统计所选时间范围内的数据，当前分钟尚未结束，刷新后可能增加。"
    "仅反映已入库的性能记录，采集异常或缓存丢失会造成少计。",
)
panels.pop()
minute_panel["id"] = 107
minute_panel["fieldConfig"]["defaults"].update(decimals=0, min=0)
minute_panel["fieldConfig"]["defaults"]["custom"].update(
    axisLabel="完成数（个/分钟）", lineInterpolation="linear"
)
minute_panel["targets"][0]["interval"] = "1m"
for item in panels:
    if item["gridPos"]["y"] >= 4:
        item["gridPos"]["y"] += minute_panel["gridPos"]["h"]
panels.insert(7, minute_panel)
print(
    json.dumps(
        {
            "uid": uid,
            "title": title,
            "tags": ["dingo", "video"],
            "links": links,
            "schemaVersion": 39,
            "version": 1,
            "timezone": "Asia/Shanghai",
            "refresh": "30s",
            "time": {"from": "now-6h", "to": "now"},
            "editable": False,
            "description": "任务性能样本：按完成时间筛选。范围输入留空表示不限制，未知值不会假定为0。阶段随schema动态变化。",
            "templating": {"list": variables},
            "panels": panels,
        },
        ensure_ascii=False,
        indent=2,
    )
)

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Video Gateway and Worker Dashboards

Video Gateway、视频 Worker 和 vLLM-Omni 的运行监控面板，使用现有 Prometheus 数据源。
这两张面板不需要 PostgreSQL 或请求性能采集器；请求级筛选分析见
[Video Performance Collection](../video-performance/README.md)。
指标口径见[Gateway 可观测性与指标](../../../docs/video-gateway/observability.md)。

## 导入

| JSON | 面板名称 | 建议 Grafana 文件夹 | Dashboard UID |
|---|---|---|---|
| `grafana/video-gateway.json` | Video Gateway | `dynamo` | `dingo-video-overview` |
| `grafana/video-worker-omni.json` | Video Worker and Omni | `vLLM` | `dingo-video-worker` |

保留 UID，两个面板按 UID 互相跳转。导入时选择已有 Prometheus 数据源；文件未绑定区域、
数据源 UID、文件夹 UID 或业务 namespace。`Data source` 列出 Prometheus 数据源，
`Namespace` 从对应服务指标动态查询。导入相同 UID 时更新已有面板，无需先删除。
源码版清除了测试环境的默认数据源与固定 namespace，未改动原面板的图表布局。

Namespace 下拉为空时，先确认数据源和应用指标是否已采集。Worker 资源图依赖额外的
DCGM、cAdvisor、kube-state-metrics 数据，资源图有数据不等于 Worker 应用指标采集正常。

## 采集要求

- 按实例采集 Gateway `/metrics` 和 Worker `/metrics`，保留 `namespace`、`pod`、`instance`。
  端口以实际服务配置为准，不经业务负载均衡代替多实例抓取。
- Worker 指标保留原生 `model` 和 `workflow`，不要用 Pod 分类标签覆盖原生模型名。
  模型分类标签冲突时应在采集侧另存，而不是将其它模型混入同一条曲线。
- 查询不限定 Prometheus job 名，通过视频专用指标识别服务；可用性图按特征指标关联 `up`。
  此关联依赖最近一小时内出现过特征指标，并非 Kubernetes 中所有 Pod 的完整清单。
- Worker 执行/预取以 `dingo_video_worker_detached_*` 为准，不用 Omni 内部队列代替。
  Omni 指标依赖 Worker 转发并启用统计；不能启用 `--disable-log-stats`。
- Gateway 模型阶段图依赖 Worker 返回 profiler 计时并由 Gateway 汇总为
  `dingo_video_model_stage_duration_seconds`；缺失阶段不补零，父子阶段不能相加。
- 数据源采集间隔应与实际抓取周期一致；`rate` / Histogram 查询使用 `$__rate_interval`。
  GPU 指标按 Worker 应用指标关联过滤，避免将同 namespace 的其它模型混入。

## Gateway 中的 Worker 到 Pool 映射

Gateway 的“Worker 执行 / 预取等待”图沿用 MiniMax-H3 的 namespace + model → Pool 映射。
源码将原部署名替换为以下脱敏示例，接入时必须同时修改该图两条 PromQL 中的 namespace 正则和 Pool 名：

| 示例 Namespace | Worker 原生 model | Gateway Pool |
|---|---|---|
| `example-video` | `minimax-h3` | `minimax-h3-fl2va` |
| `example-video` | `minimax-h3-ref2va` | `minimax-h3-ref2va` |
| `example-serving` | `minimax-h3` | `dingo-minimax-h3-fl2va` |
| `example-serving` | `minimax-h3-ref2va` | `dingo-minimax-h3-ref2va` |

映射不是自动发现。新增模型或更换 Pool 命名约定时，需要按真实标签更新；仅改 Namespace 筛选不会修改映射。
其它使用 `pool` 原生标签的 Gateway 图表不需要这份映射。Worker 面板也不依赖 Gateway Pool 命名。

## 统计与访问边界

Gateway 的共享队列、准入 slot 和共享制品存储不能跨副本直接相加。
区间计数和 Histogram P95 受采样、外推和桶精度影响；吞吐单位切换是同一速率的单位换算，
不是固定区间的实际请求数。所选时段内累计曲线也受 counter 重启及采集历史限制。

GPU 功率按 Worker Pod 合计，利用率按卡平均；这些是硬件采集数据，不是 Omni 推理耗时。
没有相应 exporter 或标签时允许显示 No data，不能据此推断业务空闲。

仅在访问者可信的隔离研发环境建议开启匿名 Viewer；匿名访问、管理员入口和数据源范围说明见
[性能观测访问建议](../../../docs/video-gateway/performance-collection.md#研发环境的匿名只读访问)。

## 本地检查

```bash
python3 -m pytest deploy/observability/video-gateway/tests -q
```

检查 JSON 结构、环境脱敏、变量、跨页链接、Prometheus 数据源及关键图表是否保留。
这些离线检查不代替目标环境的 PromQL 查询验收；导入后核对 Namespace、Pool/workflow、
Worker 数量、执行/预取占用及资源图标签。此次目录迁入不会变更集群部署或 Grafana 实例。

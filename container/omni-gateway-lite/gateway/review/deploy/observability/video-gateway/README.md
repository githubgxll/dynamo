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
- 数据源采集间隔应与实际抓取周期一致；速率图使用 `rate`，Gateway 和 Worker 的 P95 使用窗口内
  Histogram 桶的 `increase`；平均值使用总量 `increase` 除以样本数 `increase`，跨实例按样本数加权。
  Gateway 模型阶段窗口固定为 `1m`，其余耗时和 Worker 平均推理步数窗口使用 `$__rate_interval`。
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

顶部第一排左侧为在途、Gateway 排队和 Worker 侧请求，右侧为空闲槽位、制品余量和采集状态；
第二排为所选区间的提交及终态事件。常用业务曲线保持每排三张，资源及诊断图保留原有分组。
采集状态使用当前查询值，区分全部正常、部分异常和全部异常；目标范围同下方采集曲线，
不是完整 Pod 清单，也不代表模型业务就绪。无数据保留 No data。
空闲槽位随 Pool 筛选，制品余量不随 Pool 筛选；空闲槽位按共享范围跨 Gateway 取 min 后汇总。
制品余量显示整数百分比，按各部署剩余量 min、总量 max 去重后，计算剩余量合计÷总量合计×100；
只统计两项均存在且总量大于零的部署。数字显示所选范围最后一个有效值，下方保留趋势线。
底层存储统计可能有刷新延迟。

Gateway 的共享队列、准入 slot 和共享制品存储不能跨副本直接相加。
区间计数和 Histogram P95 受采样、外推和桶精度影响；吞吐单位切换是同一速率的单位换算，
不是固定区间的实际请求数。所选时段内累计曲线也受 counter 重启及采集历史限制。

Gateway 和 Worker 先在每个实例内校验采样快照，再计算窗口增量并按原查询分组汇总。
P95 使用桶增量估算；平均耗时和平均推理步数使用总量增量除以样本数增量。
在 Prometheus 中，对同一个输入窗口，`increase` 与 `rate` 相差统一倍数，
因此用增量代替速率不会改变分位数或平均值；采样完整性检查另见下文。
VictoriaMetrics 的 `rate` 按每条序列的实际采样跨度归一化；桶间缺点不一致时，即使计数增量相同，
也可能产生不同的速率，形成不存在的慢请求尾部。`_sum` 与 `_count` 缺点不一致时，
同样会偏移平均值。使用窗口增量避免了这种归一化偏差。吞吐和错误速率、CPU 使用率仍使用 `rate`。
参见 [VictoriaMetrics MetricsQL 差异](https://docs.victoriametrics.com/metricsql/)。

仅替换 `rate` 仍不充分：10:28:47.201 的 execution 采样只有 `+Inf` 桶存在，17 个有限桶缺失，
短窗口会把不一致的桶增量解释成极慢请求，仍能算出 1800 秒假尖峰。
当前查询对同一实例保留桶集合完整且原始采样时间一致的快照，桶集合参考此前一小时出现的序列；
平均值则要求 `_sum` 和 `_count` 采样时间一致。通过子查询取点，再计算原窗口的增量。
隐藏的 `snapshot_interval` 使用 Grafana 自动间隔（目标 60000 点，最小 15 秒）；
最近 1 小时/6 小时仍按 15 秒取点，最近 30 天按 30 秒取点，避免固定 15 秒产生约 17.3 万点，
超过 VictoriaMetrics 默认每条子查询 10 万点的限制。此间隔独立于面板宽度，模型阶段统计窗口仍为 1 分钟。
该调整同时应用于 Gateway 和 Worker 耗时图；未增加后端点数限额。
检查在跨实例聚合之前执行，避免用另一个实例的正常采样掩盖当前实例缺点。

数据不足的窗口留空，下一份完整快照恢复后继续计算。该方法不恢复缺失采样，不能保证任意丢失、
桶结构变化或原始计数错误下仍能重建真实分布。仍需比较采集端本地数据与远端存储，检查
remote write 丢弃、去重和聚合配置。真实慢请求不设置耗时上限，完整采样的超长请求仍参与计算。

模型阶段窗口内仍有已完成样本时，短暂平线可以是正常的一分钟滚动统计；窗口内无有效样本时应断线。
短窗口可减少最后一批样本的显示拖尾，但低流量或采集缺失时更容易出现 No data，均值和 P95 也更易波动。
VictoriaMetrics 会省略 NaN 时间点，Grafana 的单条稀疏序列可能没有显式空值，单设 `spanNulls=false`
仍可能把两个有效点连成平线或斜线。普通 time-series outer join 的缺失位置实际是 `undefined`，
uPlot 会跳过这些位置，只有显式 `null` 才能形成断线。耗时图通过隐藏的 `vector(0)` 时间轴，
将 Time 临时转为毫秒字符串键，使用表格 outer join 以 `null` 补齐缺失位置，再恢复原始时间并升序排列；
每条序列的标签、图例和有效值均保留，同名图例不会导致序列合并。
`spanNulls=false` 因此能断开无数据区间。辅助时间轴不出现在曲线、图例或提示中，
也不参与任何统计；没有把真实指标补零。图例 `Last *` 仍表示所选范围内最后一个有效值。

GPU 功率按 Worker Pod 合计，利用率按卡平均；这些是硬件采集数据，不是 Omni 推理耗时。
没有相应 exporter 或标签时允许显示 No data，不能据此推断业务空闲。

仅在访问者可信的隔离研发环境建议开启匿名 Viewer；匿名访问、管理员入口和数据源范围说明见
[性能观测访问建议](../../../docs/video-gateway/performance-collection.md#研发环境的匿名只读访问)。

## 本地检查

```bash
python3 -m pytest deploy/observability/video-gateway/tests -q
```

检查 JSON 结构、环境脱敏、变量、跨页链接、Prometheus 数据源及关键图表是否保留。
另外提供实际 VictoriaMetrics 引擎回归：

```bash
VICTORIAMETRICS_BINARY=/path/to/victoria-metrics-prod \
  python3 -m pytest --confcutdir=deploy/observability/video-gateway/tests \
  deploy/observability/video-gateway/tests -q
```

该回归仅启动本机回环地址上的临时实例，导入脱敏后的缺点样本，验证旧查询可复现
450/1530 秒假尖峰、仅用 `increase` 时仍可复现 execution 的 1800 秒假尖峰，
完整快照修正后异常位置留空、恢复采样后为 117 秒。真实 480/2400 秒请求仍能显示，空窗口不补零。
回归覆盖两张面板全部 6 条 P95 和 10 条平均值查询，包括多实例样本加权、总量与计数采样不齐、
正常采样下平均值不变、持续抓取但停止请求时曲线留空、恢复请求后重新显示，以及真实大值保留；
Worker 和其它 Gateway 图使用缺点模式构造对照，
不代表已在这些图上确认过线上异常。
未提供二进制时，该引擎回归会跳过；普通 JSON 检查仍会执行。已验证 VictoriaMetrics v1.122.0。

若有保存的 Grafana 12.2.1 前端资产，可另外设置 `GRAFANA_BUNDLE_DIR` 并安装 Node，
运行同一 pytest 命令以加载实际前端 join transformer，验证缺失区间变为空值、辅助时间轴隐藏，
且保留原系列名称。未提供该目录时，仅跳过此项前端回归。

这些离线检查不代替目标环境的 PromQL 查询验收；导入后核对 Namespace、Pool/workflow、
Worker 数量、执行/预取占用及资源图标签。此次目录迁入不会变更集群部署或 Grafana 实例。

---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Video Gateway 性能采集
subtitle: HTTP 拉取、日志发现、动态配置、去重续采及 PostgreSQL 存储
---

性能采集器支持读取 Gateway 的终态性能日志，或通过显式地址拉取 Gateway 的短期内存记录，写入独立 PostgreSQL，供 Grafana 按请求参数筛选、
聚合耗时和查询任务明细。每个大区可部署一套，覆盖一个或多个 namespace，无需为每个模型单独部署。
源码、SQL、部署模板、Grafana JSON 和测试集中在 `deploy/observability/video-performance/`。
服务级 Prometheus 指标及日志开关见[可观测性与指标](observability.md)。

## 数据来源

### HTTP 定时拉取

**Experimental.** Gateway 的 `performance_cache` 默认开启，滚动保留最近 300 秒（5 分钟）的终态性能快照。
采集器运行 `http_pull.py`，按手动配置的每个实例地址拉取；不读取 Kubernetes 日志，也不需要 Pod 发现 RBAC。
这不实现实例自动发现。普通业务 LB 不能保证覆盖全部 Gateway 副本，多个副本必须分别提供可访问地址。

Gateway 配置示例：

```yaml
performance_cache:
  enabled: true
  retention_s: 300
  max_records: 50000
  max_bytes: 67108864
  token_env: DINGO_VIDEO_PERFORMANCE_TOKEN
```

不配置此节或省略 `enabled` 时默认为 true；可用 `enabled: false` 显式关闭。保留时长允许 1–86400 秒。条数上限允许 1–100000，字节上限允许 1–134217728，
单条编码后的事件最多 256 KiB。容量上限按序列化记录计，不等同于进程 RSS。
默认每个 Gateway 进程最多缓存 50000 条、64 MiB，按实际记录占用，不预分配全部容量。
条数或容量先达到上限时，会提前淘汰最旧记录，不能保证每条记录都保留满 5 分钟。
显式配置的旧值仍然生效，升级镜像不会覆盖 YAML 中的设置。
Token 从指定环境变量读取；未配置或为空时 Gateway 正常启动并继续缓存，但接口拒绝访问，返回 401。
配置非空 Token 时要求至少 16 个无空白 ASCII 字符，不合法则拒绝启动；使用独立随机凭据，不能使用 Kubernetes Token。
缓存与 `DINGO_VIDEO_PERFORMANCE_RECORDS` 日志开关独立。不重新读取媒体或查询 etcd；保留终态发生时间，不用采集时间替代。

接口为 `GET /internal/v1/performance/records`，使用 `Authorization: Bearer <token>`，响应禁止缓存。
仅允许 `cursor` 和 `limit` 参数；`limit` 默认 100、最多 500，单页事件编码总量约最多 1 MiB。
每实例认证后的查询限速为每秒 20 次，超限返回 429 和 `Retry-After: 1`；未认证返回 401，关闭返回 404，
错误参数返回 400，不支持的方法返回 405，不转发给上游。

响应包含 `schema_version=1`、`instance_id`、`retention_s`、`oldest_seq`、`latest_seq`、
`next_seq`、`next_cursor`、`has_more`、`gap_detected`、`instance_changed`、`source` 和 `records`。
每条记录形如 `{"seq": 1, "event": {"log_type": "video_task_lifecycle", "performance": {}}}`；event 保留有效的任务身份、终态和性能摘要，
不包含普通日志的任意 reason/error 文本。source 的 namespace/Pod UID 等通过 `POD_NAMESPACE`、`POD_NAME`、`POD_UID` 环境变量注入；
`GATEWAY_IMAGE` 可提供来源镜像信息。缺少 namespace 时，采集器使用经过认证的配置目标身份；明确返回不同 namespace 或部署 ID 时拒绝该响应。

首次不传游标，返回现存窗口内的记录。游标绑定实例和序号，分页保留快照上界，不会因持续新增任务一直无法完成。
读取不删除记录；超过保留期、容量淘汰和进程重启都会造成潜在漏采。`gap_detected` 表示可检测的缓存缺口，
`instance_changed` 表示游标实例已变化；都不能恢复已丢失数据，实例重启也不能推算精确漏采条数。

采集器 JSON 参考 `http-pull.example.json`：

```json
{
  "targets": [{
    "name": "gateway-a",
    "cluster_id": "example-region",
    "namespace": "example-video",
    "deployment_id": "example-deployment",
    "url": "https://gateway-a.example.invalid",
    "token_env": "VIDEO_PERFORMANCE_TARGET_A_TOKEN",
    "poll_interval_s": 5,
    "timeout_s": 3
  }]
}
```

`url` 为 Gateway 基地址，可包含已有入口前缀，不含查询参数或凭据。默认验证 HTTPS 证书，不跟随重定向；
隔离测试 HTTP 必须显式设置 `allow_insecure_http: true`。Socket 超时和循环读取预算均使用 `timeout_s`；
单次阻塞读可能使总耗时超过预算，不将其描述为严格端到端截止时间。响应最多 2 MiB。
每目标周期默认 5 秒，允许 5–60 秒；使用 10 或 15 秒时应留足低于 Gateway 保留时长的故障恢复余量。

目标配置动态重读，错误时保留上次有效配置；移除目标不删除历史数据。凭据环境变量变更需要按部署方式重载进程。
采集器不加载 Kubernetes 凭据，默认最多并发拉取 4 个目标，可通过 `max_workers` 调整至 1–16；坏目标独立退避。
数据与 HTTP 游标同事务提交，失败不推进游标；相同业务记录重复拉取不新增行，冲突不按接收时间覆盖，
坏事件和冲突仅将目标、实例、序号及原因写入隔离表，不保存未过滤的原始载荷。

`render_http.py` 生成一个包含 PG 与 HTTP 采集器两个容器的 Deployment、数据库 Service 和 ConfigMap，不生成 ServiceAccount 或任何 RBAC，
不自动 apply。通过 `<name>-auth` 提供数据库写入/只读密码，通过 `<name>-tokens` 提供目标 Token，Secret 仅放在采集器所在 namespace。
采集器通过 `PGHOST=127.0.0.1` 访问同 Pod 数据库，只读挂载共享数据卷并用 `DATA_PATH` 检查所在文件系统空间。
PG 和采集器共用 Pod 生命周期；更换任一容器镜像导致 Pod 重建时，emptyDir 中的数据会丢失，必须先备份再升级。

旧 v2 若将 PG 与采集器放在同一 emptyDir Pod，禁止直接重建后再取历史数据。
先暂停旧采集、导出校验、恢复到新的同 Pod 实例、切换 Grafana 并验收，再删除旧组合服务及专属日志读取权限。
日志模式与 HTTP 模式使用同一数据库的单采集器锁，切换前需要停止旧写入者。

### Kubernetes 日志拉取

Gateway 设置 `DINGO_VIDEO_PERFORMANCE_RECORDS=1`，保留 INFO 生命周期日志。
采集器只接受 `log_type=video_task_lifecycle`、event 为 completed/failed/cancelled、且含有
有效 performance 摘要的记录。当前接受的摘要版本为 `schema_version=1`。

模型阶段仍需 Worker 返回有效的 `model_execution`；MiniMax-H3 Worker 需要启用
`--enable-diffusion-pipeline-profiler`。仅启用 Gateway 日志不会恢复缺失的模型计时。
采集器不读取 Worker 日志、不调用推理接口、不查询 etcd、不检查视频文件。
不同模型的阶段名按摘要存储，未知/缺失参数保留 NULL；无须把每个模型名写进发现规则。

## 自动发现与动态 namespace

namespace 可以包含多个名字，空列表表示暂停发现和读取。环境相关的默认采集列表以部署配置为准，
本文统一使用 `["example-video"]` 作为脱敏示例；部署前替换为实际目标。
采集列表只决定日志来源，不决定 PostgreSQL/采集器 Pod 部署在哪个 namespace。
部署 namespace 和稳定的大区标识通过 renderer 的 `--namespace`、`--cluster-id` 显式指定。

采集器列出目标 namespace 的 Pod，从容器 command/args 中识别 `dingo.video_gateway`，
取得实际容器名和重启次数。新增、重建 Pod 自动加入；不依赖固定 Pod 名、模型名或 role 标签，
兼容不同平台的组件标签前缀。若启动命令隐藏在脚本内，可用 Pod annotation
`dingo.datacanvas.com/performance-container` 显式指定容器。

ConfigMap 以整个目录挂载，不使用 subPath；采集器每轮重读 JSON。投射存在传播延迟，
以数据库 `performance_collector_health.details.namespaces` 为实际生效列表。
配置错误保留上次有效值并记录异常；启动时无法读取有效配置则保持暂停。

停止某 namespace 的采集不会删除数据或游标。恢复后尝试续采，能补多少取决于现存 Kubernetes 日志。
跨 namespace 的 `pods get/list`、`pods/log get` 权限需要事先通过 Role/RoleBinding 绑定给采集器账号；
动态添加 namespace 不会自动扩权。删除采集项也不会自动撤销已授予的 RBAC。

## 配置参考

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `namespaces` | 以部署配置为准 | 采集目标，本文示例为 `["example-video"]`；最多 50 个有效名字，去重后使用 |
| `poll_interval_s` | 15 | 每轮处理完成后的等待秒数，允许 5–3600 |
| `initial_lookback_s` | 86400 | 新来源首次回看窗口，允许 60–604800 秒 |
| `max_read_bytes` | 4194304 | 每个来源单批读取预算，允许 1KiB–64MiB |
| `max_line_bytes` | 262144 | 单行预算，允许 1KiB–1MiB，不能超过读取预算 |
| `storage_budget_bytes` | 10737418240 | 数据目录预算，默认 10Gi，renderer 用它设置 emptyDir 上限 |
| `storage_reserve_bytes` | 536870912 | 额外预留空间，默认 512Mi，必须小于预算 |
| `warning_fraction` | 0.8 | 占用估算达到该比例时报告 warning |
| `pause_fraction` | 0.9 | 占用估算达到该比例时暂停新日志入库，必须大于 warning 且小于 1 |

修改热配置中的空间预算不会修改现有 Pod 的 emptyDir 上限。扩大存储需要重新渲染部署，
这通常重建 Pod；使用 emptyDir 时先导出需要保留的数据。

环境变量：

| 变量 | 用途 |
| --- | --- |
| `CLUSTER_ID` | 必填，稳定的大区/集群身份；整套服务重启时保持不变 |
| `DB_PASSWORD` | 必填，采集写入数据库的凭据，从 Secret 注入 |
| `READ_PASSWORD` | 必填，首次创建 Grafana SELECT-only 用户的凭据，从 Secret 注入 |
| `CONFIG_PATH` | 默认 `/opt/config/config.json` |
| `DATA_PATH` | 默认 `/var/lib/postgresql/data`，只用于 statvfs 获取所在文件系统可用空间 |
| `PGHOST` / `PGPORT` | 默认 localhost / 5432，PostgreSQL 与采集器同 Pod |

## 去重和恢复

业务主键为 `(cluster_id, namespace, deployment, task_id, attempt)`，同名跨 namespace 数据不会互相覆盖。
同一记录重复读取不增加行数；同一主键较新的终态日志更新已有行，因此数据库行数不是日志事件数。
排队即取消的任务允许 attempt=0。当前保存每个有终态日志的 attempt，不记录全部中间状态时间线。

游标绑定 `(cluster_id, namespace, pod_uid, container, restart)`，保存 Kubernetes 日志时间戳及
该时间戳已处理行数，支持纳秒和时区。从最后提交时刻重读，跳过已读前缀。
记录写入和游标推进同事务；数据库失败则回滚并重读。数据库 advisory lock 限制单采集器运行。
日志读取后再次核对 UID/restartCount，防止重启期间提交错误代际的游标。

客户端按前缀分批读取，不使用 Kubernetes limitBytes 截取日志尾部。一次读取最多预算加一条完整行；
超长/不完整行、时间倒退或重复前缀耗尽预算而没有进展时报告来源错误，不静默跳过任意日志。
容器重启后尝试读取最近一次 previous 日志，无法恢复任意历史重启。

Pod 删除、日志轮转或长时间停采仍可能漏样，采集器无法精确证明每一次缺失。
这套链路用于性能分析，不承诺完整审计。旧版数据库无 schema 版本时拒绝启动，避免误执行隐式迁移。

## 数据表与字段

`video_performance` 保留原性能面板的列结构。重要字段如下：

| 字段 | 来源和含义 |
| --- | --- |
| `cluster_id` / `namespace` | 采集器所在大区标识 / 日志来源 Pod 的 Kubernetes namespace |
| `deployment` / `task_id` / `attempt` | Gateway lifecycle 摘要中的部署 ID、任务 ID 和 attempt |
| `completed_at` | HTTP 模式使用快照中的任务终态时间；日志模式使用日志 timestamp_ms，接近日志发布时刻。均不是采集入库时间 |
| `model` / `pool` / `status` | 日志中的模型、pool_id、completed/failed/cancelled |
| `configuration_revision` | Gateway 任务摘要中的配置版本 |
| `environment_revision` | 当前为 unknown；没有把全 namespace Worker 配置混合成一个环境版本 |
| `schema_id` / `stages` | 模型阶段 schema 和阶段秒数 JSON；未提供时为 unavailable / 空对象 |
| `seconds` / `width` / `height` / `fps` / `steps` / `sound` | 输出请求参数，不假定等于解码后真实视频参数 |
| `input_type` / `image_count` / `video_count` / `audio_count` | 引用类型组合和各类型数量 |
| `image_pixels` / `image_bytes` | 已知引用图片总像素 / 上传总字节 |
| `video_seconds` / `video_bytes` / `video_frames` | 已知参考视频总时长 / 上传字节 / 实际帧数；未知值不推算 |
| `audio_seconds` / `audio_bytes` | 已知参考音频总时长 / 上传字节 |
| `e2e_s` | 服务端创建到终态的总时间，包含重试和等待 |
| `queue_s` / `worker_queue_s` / `execution_s` / `finalize_s` | 当前/最终 attempt 的 Gateway 排队、Worker 等待、Worker 执行和 Gateway 结果处理 |
| `details` | 来源 Pod UID、容器、restart、Gateway 镜像、worker_id，以及经过字段白名单过滤的 references 元数据 |

不把来源 Gateway 镜像称为 Worker 执行环境。提示词、URL、文件路径、媒体内容和整个容器 env 不入库。
计时口径见[API 说明](video-gateway-api.md)，模型父子阶段不能累加，缺失耗时不能补零。

| 表或视图 | 作用 |
| --- | --- |
| `video_performance_stages` | 将 stages JSON 展开为 stage / stage_s，支持按阶段聚合 |
| `collector_sources` | 各 Pod UID/容器代际的游标、最近错误、写入数、坏记录数及截批标识 |
| `performance_collector_health` | 最新轮询、最近事件、错误、配置、来源列表和空间状态 |
| `performance_environments` | 兼容已有面板的预留表，当前不采集 Worker 配置快照 |
| `performance_http_sources` | HTTP 目标身份绑定、实例游标、最近成功/错误、插入数、缺口和实例变化次数 |
| `performance_http_rejections` | HTTP 坏事件或冲突的目标、实例、序号及原因，不保存原始数据 |
| `collector_schema` | 当前存储结构版本为 2，与日志 schema_version=1 是不同版本号 |

Grafana 使用 `grafanareader`，仅授予这些表/视图的 SELECT 和 schema USAGE，设置 10 秒 statement_timeout。
初始化同时撤销独立性能库中 PUBLIC/只读账号的 schema CREATE 和数据库 CREATE/TEMPORARY 权限，
避免旧 PostgreSQL 默认权限允许只读账号建表。该初始化仅用于独立性能数据库，不用于共享业务库。
写入账号不交给 Grafana。数据库 Secret 不进入源码或 dashboard JSON。

## 空间保护与保留

当前示例使用 10Gi emptyDir，不申请 PVC。每轮计算数据库大小、表及索引大小、WAL 大小和样本数，
并读取所在文件系统可用空间。预算占用估算为数据库＋WAL＋预留量；达到 warning/pause 阈值，
或文件系统可用空间低于预留量时报告状态/暂停。暂停时不读新来源、不推进日志游标。

不自动按天数、namespace、空间阈值删除历史。暂停期间日志仍可能轮转，因此空间告警应及时处理。
预算估算不是硬隔离磁盘，无法保证防止 WAL 突增或节点磁盘压力驱逐。
emptyDir.sizeLimit 不是独立磁盘预留，df 显示节点容量也不是 Pod 配额。

采集器容器单独重启而 Pod 不变时数据库/游标保留；整个 Pod 被删除、替换或驱逐会丢库。
需长期保存时应使用持久存储或外部 PostgreSQL，并明确迁移和备份流程。

研发验证以 49 条历史实际摘要扩展到 5 万条，表及全部索引约 75MiB，增量约 1.54KiB/条；
按相同结构 1 万条/天约 15MiB/天。这只说明当前数据量级，不包含 WAL、更新膨胀和临时查询空间，
也不承诺其它模型或引用规模的记录大小。

## Grafana 对接与验证边界

导入组件目录中的 `grafana/performance.json`、`performance-advanced.json`。
面板 Data source 直接选择 PostgreSQL 数据源，例如 video-performance-example-region。
变量没有写死区域 UID，也不要求数据源采用固定名称。两个性能页可独立部署，
不依赖 Prometheus 或 Gateway/Worker 运行面板；常用页、高级页和明细链接保留所选数据源。

阶段和筛选候选从数据库取得。已知 schema 的阶段说明来自面板目录的 stage_metadata.json，
新增 schema 没有展示说明时仍可显示其原始阶段名；新增说明后重新生成 JSON。
两张面板保留 namespace 筛选，明细查询有行数上限。当前 environment 筛选不代表准确 Worker 硬件归因。

### 研发环境的匿名只读访问

在网络隔离、访问者可信的研发环境，可开启 Grafana 匿名 Viewer，方便直接查看性能面板；
生产环境或公网可达的服务不建议开启，应保留登录或接入统一认证。
配置 Grafana 容器环境变量，`GF_AUTH_ANONYMOUS_ORG_NAME` 必须与面板所属组织名称一致：

```yaml
env:
  - name: GF_AUTH_ANONYMOUS_ENABLED
    value: "true"
  - name: GF_AUTH_ANONYMOUS_ORG_NAME
    value: Main Org.
  - name: GF_AUTH_ANONYMOUS_ORG_ROLE
    value: Viewer
  - name: GF_USERS_VIEWERS_CAN_EDIT
    value: "false"
  - name: GF_AUTH_DISABLE_LOGIN_FORM
    value: "false"
  - name: GF_AUTH_BASIC_ENABLED
    value: "true"
```

匿名访问者可以查看和筛选，管理操作仍需登录；不要关闭管理员登录入口或把匿名角色设为 Editor/Admin。
继续使用只读 PostgreSQL 账号，不向 Grafana 提供写库凭据。环境变量修改需要重启 Grafana，
不需要重启采集器或 PostgreSQL；若 Grafana 自身使用 emptyDir，应先备份未纳入 provisioning 的配置。
验证无登录状态下能打开两张面板、查询性能数据，并确认保存面板及管理接口拒绝匿名请求。
设置 `GF_AUTH_ANONYMOUS_ENABLED=false` 并重启 Grafana，可关闭匿名访问。

> [!WARNING]
> 任何能访问 Grafana 地址的人都能取得匿名角色授权范围内的数据，可能不止面板展示的结果。
> Namespace、模型等筛选条件不是权限边界。限制入口网络范围，不要将此方式当作跨租户隔离。
> 具体机制见 [Grafana 匿名认证](https://grafana.com/docs/grafana/latest/setup-grafana/configure-access/configure-authentication/anonymous-auth/)
> 和 [Grafana 数据源访问边界](https://grafana.com/docs/grafana/latest/setup-grafana/configure-security/)。

### 采集验证边界

原采集器已验证自动发现、日志读取、历史摘要回放、去重、事务回滚、namespace 暂停恢复及单容器重启保留。
HTTP 模式已验证真实 FL2VA 请求的终态缓存、自动拉取、入库及面板查询，
并验证整数 Worker ID 和字符串 Worker ID 的兼容性；Ref2VA 的真实生成验证仍受资源条件限制。
源码迁入本身不切换运行实例或 Grafana 数据源。
新的小型采集器镜像和 PostgreSQL 16 部署模板需另行构建/部署验证；已有集群验证使用的是旧 PostgreSQL 11。

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Video Performance Collection

通过日志或只读 HTTP 接口采集 Video Gateway 终态性能记录，保存至 PostgreSQL，供 Grafana 按请求参数分析。
机制和字段见[性能采集文档](../../../docs/video-gateway/performance-collection.md)，
Prometheus 指标和日志开关见[Gateway 可观测性](../../../docs/video-gateway/observability.md)。

## 目录

```text
collector.py / schema.sql      采集、游标和数据库结构
http_pull.py                  按显式实例地址定时 HTTP 拉取，不使用 Kubernetes 账号
http-pull.example.json        HTTP 目标、认证引用、采集周期配置
render_http.py                PG 和 HTTP 采集器同 Pod 的部署清单，不生成 RBAC
config.example.json           namespace、采集周期和空间预算配置
Dockerfile / requirements.txt 独立 Python 镜像，不依赖 Dynamo/vLLM
render.py / prepare.py         渲染清单 / 创建一次性凭据
kubernetes/                   部署、配置、namespace RBAC 和 Service 模板
grafana/                      两张性能面板、生成脚本及数据源示例
tests/                        采集和部署渲染的本地测试
```

历史验证快照、回放脚本和具体集群资源名保留在测试项目，不复制进组件。

## 准备

HTTP 模式使用 Gateway `performance_cache`，默认开启，不依赖性能日志开关。默认缓存 300 秒（5 分钟），每进程最多 50000 条、64 MiB，拉取周期仍为 5 秒，
但缓存过期、容量淘汰和进程重启可能漏样。它不自动发现 Gateway，多副本不能仅配置一个普通 LB。
开启和接口协议见[HTTP 定时拉取](../../../docs/video-gateway/performance-collection.md#http-定时拉取)。
未配置 Token 时只缓存、不允许拉取，不影响业务启动；需要禁用时设置 `performance_cache.enabled: false`。
以下原有准备和 `render.py` 部署流程适用于 Kubernetes 日志模式。

Gateway 开启 `DINGO_VIDEO_PERFORMANCE_RECORDS=1`，保留 INFO 日志；模型阶段还需要 Worker profiler。
数据库/采集器可以放在现有的观测 namespace，目标 namespace 列表单独配置。
本文的 example-observability、example-video、example-region 分别表示观测服务 namespace、
日志来源 namespace 和大区/集群标识，均为脱敏示例。本组件不创建 namespace；操作前替换为实际值。

在仓库根目录安装开发/渲染依赖并构建镜像：

```bash
python3 -m pip install -r deploy/observability/video-performance/requirements-dev.txt
docker build -t video-performance:local deploy/observability/video-performance
```

实际部署时把镜像推到可拉取的仓库，并通过 `--collector-image` 传入地址。
该 Dockerfile 不属于推理镜像的构建流程，没有修改原有 Gateway/Worker 镜像依赖。
模板默认使用 postgres:16，可通过 `--postgres-image` 指定受维护镜像及 digest。
源码迁入阶段没有构建、推送或上线这些镜像。

## 渲染和部署

HTTP 模式单独渲染：

```bash
python3 deploy/observability/video-performance/render_http.py \
  --namespace example-observability --collector-image video-performance:local \
  --config deploy/observability/video-performance/http-pull.example.json
```

输出仅含 hosting namespace 的一个 Deployment、数据库 Service 和配置 ConfigMap，不 apply、不创建 namespace 或 RBAC。
部署前替换示例目标地址和来源身份，准备 `<name>-auth` 的数据库凭据及 `<name>-tokens` 的目标凭据。
PG 与采集器作为两个容器放在同一个 Pod，采集器通过 localhost 访问数据库，并只读挂载数据卷以检查空间。
两者共用 Pod 生命周期；只重启容器不清空 emptyDir，但修改镜像等导致 Pod 重建时会丢库，升级前必须备份。
切换旧组合 Pod 前必须先备份、校验、恢复并切换 Grafana；不能直接 apply 覆盖旧库或提前删除旧服务。

下文保留原日志模式的部署方式。

按部署环境准备自己的配置文件，环境默认值以实际部署配置为准。
例如准备 `/tmp/video-performance-config.json`，显式指定日志来源：

```json
{
  "namespaces": ["example-video"]
}
```

renderer 创建 hosting namespace 的
ServiceAccount/Deployment/Service/ConfigMap，以及每个采集 namespace 的只读 Role/RoleBinding。
跨 namespace RBAC 需要对应管理权限；它不使用 ClusterRole，也不会自动创建 namespace。

```bash
python3 deploy/observability/video-performance/render.py \
  --namespace example-observability --cluster-id example-region \
  --collector-image video-performance:local \
  --config /tmp/video-performance-config.json \
  > /tmp/video-performance.yaml
kubectl -n example-observability apply --dry-run=server -f /tmp/video-performance.yaml
python3 deploy/observability/video-performance/prepare.py --namespace example-observability
kubectl -n example-observability apply -f /tmp/video-performance.yaml
```

清单可能包含目标 namespace 的 RBAC，操作前核对全部对象的 namespace。
`--name` 可用于独立验证实例，默认 dingo-video-performance；prepare.py 要使用相同 name。
测试时显式指定获准的日志来源和部署 namespace，通过 `--queue-name` 指定现有弹性队列，
并核对该环境对应的节点亲和性与污点容忍配置。不指定节点、不申请 GPU、不创建 Namespace。

prepare.py 只在 Secret 不存在时创建两份随机凭据，已有密码保持不变，不打印密码。
集群 RBAC 错误不会被当成“Secret 不存在”。Grafana 使用 read-password 对应的 grafanareader。

## 修改采集目标

修改 `<name>-config` 的 data.config.json，保留完整 JSON。
挂载整个 ConfigMap 目录，每轮重读；等投射传播后查看数据库 health 确认实际生效列表。
删除 namespace 或改为 [] 都不删除数据/游标。增加 namespace 需要先补 Role/RoleBinding；
不要通过重建 Deployment 更新采集列表。

namespace 停采不会自动撤销其 RBAC，清理账号时应同时清理所有曾绑定的 namespace 中的 RoleBinding/Role。
删除 Deployment/Pod 会丢失当前 emptyDir 数据，cleanup 不能以宽泛 namespace/label 批量删除其它部署。

## 空间和健康

默认 emptyDir 10Gi，PostgreSQL 请求 2Gi ephemeral-storage，限制为预算加 2Gi；
collector 另有 1Gi limit。两容器各请求 1 CPU，均不申请 GPU。
统计 DB、WAL 和 512Mi 余量，80% warning、90% 暂停，不自动淘汰历史。
Pod 重建会丢数据；当前方案用于研发验证，长期保存需持久存储或外部数据库。

```bash
kubectl -n example-observability logs deployment/dingo-video-performance -c collector --tail=3
kubectl -n example-observability exec deployment/dingo-video-performance -c postgres -- \
  psql -U postgres -d performance -c 'SELECT * FROM performance_collector_health;'
```

Running 不代替采集健康检查。注意 last_error、details.namespaces、details.storage.paused/warning
以及 collector_sources 的 cursor_time/last_error/truncated。
存储结构版本为 2，旧无版本数据库不自动迁移；不要将此 schema 覆盖旧数据库。

## Grafana

1. 创建 PostgreSQL 数据源，使用 grafanareader。示例为 grafana/datasource.example.yaml；
   HOST 和 READ_PASSWORD 从部署环境提供，示例 sslmode=disable 用于内部验证，正式环境按连接要求配置 TLS。
2. 给 PostgreSQL 数据源设置易识别的名称，例如 video-performance-example-region，无需匹配 Prometheus 名称。
3. 导入 grafana/performance.json 和 performance-advanced.json，顶部 Data source 直接选择 PostgreSQL 数据源。
   两个性能页可独立部署，不依赖 Prometheus 数据源或 Gateway/Worker 运行面板；跨页跳转保留数据源选择。

已有 Gateway/Worker Prometheus 看板无需数据库，SQL 性能页是单独的一组页面。
数据库保存性能样本、采集游标和健康状态，不保存视频内容。
environment_revision 当前为 unknown，准确 Worker 配置归因尚未采集。

隔离的研发环境可开启匿名 Viewer，免登录查看性能页，编辑和管理仍需登录。
配置、回退和安全边界见[匿名只读访问](../../../docs/video-gateway/performance-collection.md#研发环境的匿名只读访问)。
匿名用户可查询其组织内数据源授权范围内的数据，筛选条件不是访问控制；不建议用于公网或生产环境。

重新生成面板：

```bash
python3 deploy/observability/video-performance/grafana/build_dashboards.py \
  > deploy/observability/video-performance/grafana/performance.json
python3 deploy/observability/video-performance/grafana/build_dashboards.py --advanced \
  > deploy/observability/video-performance/grafana/performance-advanced.json
```

## 本地验证

```bash
python3 -m pytest deploy/observability/video-performance/tests -q
```

采集器代码和 SQL 迁自已验证版本；部署渲染、镜像和 Grafana 数据源配置为源码中的通用版本，
需要构建和部署联调后才能声称新模板已线上验证。没有自动部署或创建数据库的测试脚本。

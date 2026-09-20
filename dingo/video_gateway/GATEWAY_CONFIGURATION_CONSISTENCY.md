# Video Gateway 多副本配置一致性约束

本文说明同一负载均衡入口后、共同服务同一组任务和 Worker pool 的多个
Video Gateway Pod，哪些配置可以长期不同，哪些只允许在滚动升级期间短暂不同，
以及哪些配置必须始终一致或保持协议兼容。

本文中的“相同”不是指 YAML 文本逐字相同。例如，每个 Gateway 可以配置不同的
etcd endpoint 顺序；只要它们最终连接的是同一个 etcd 集群和同一个逻辑 key 空间，
就满足一致性要求。

## 基本原则

通常应由同一个 Deployment、ConfigMap 和镜像版本生成所有 Gateway Pod，以减少人为
配置漂移。但是，多副本安全性不依赖所有本机性能参数完全相同。

判断一个参数属于哪一类时，使用以下规则：

1. 只限制当前 Gateway 自身资源消耗的参数，可以长期不同。
2. 改变请求准入、等待时间、重试或清理策略，但不改变共享数据含义的参数，可以在
   滚动升级期间短暂不同。
3. 改变 etcd key 空间、任务状态含义、Worker slot 身份、fencing、artifact 定位或
   Worker wire protocol 的参数，必须始终一致或向前、向后兼容。

CAS、lease、attempt、owner generation 和 execution token fencing 是防止重复终态、
错误释放 slot 和旧结果覆盖新结果的安全边界。调度策略短暂不同不应绕过这些边界。

## A. 可以在不同 Gateway Pod 之间长期不一致

这些参数仅控制单个 Gateway 的资源占用、检查频率或非共享连接行为。不同 Pod 可以
根据自身 CPU、内存和运维需求独立设置。

| 配置 | 允许差异 | 影响 |
|---|---|---|
| `scheduling.finalization_concurrency` | 可以长期不同 | 每个 Gateway 同时执行多少个本地后处理任务 |
| `scheduling.finalization_pending_limit` | 可以长期不同 | 每个 Gateway 接受多少个本地待后处理任务 |
| `scheduling.finalization_retry_delay_s` | 可以长期不同 | 当前 Gateway 后处理暂态失败后的本地退避速度 |
| `scheduling.discovery_interval_s` | 可以长期不同 | 当前 Gateway 刷新 Worker 视图的频率 |
| `scheduling.dispatch_interval_s` | 可以长期不同 | 当前 Gateway 尝试派发任务的频率 |
| `runtime.discovery_watchdog.interval_s` | 可以长期不同 | 当前 Pod 的 watchdog 检查频率 |
| `runtime.discovery_watchdog.mismatch_grace_s` | 可以长期不同 | 当前 Pod 对 discovery 短暂不一致的容忍时间 |
| `task_store.request_timeout_s` | 可以长期不同 | 当前 Gateway 单次 etcd 请求的客户端超时 |
| `task_store.watch_response_timeout_s` | 可以长期不同 | 当前 Gateway watch 长连接的客户端超时 |
| etcd endpoint 顺序或首选 endpoint | 可以长期不同 | 必须仍指向同一个 etcd 集群 |
| 日志级别、日志格式、指标抓取配置 | 可以长期不同 | 只影响当前 Pod 的可观测性 |
| Pod CPU、内存、线程池等资源配置 | 可以长期不同 | 只影响当前 Pod 的吞吐与延迟 |

`http.host` 和 `http.port` 也可以因 Pod 网络布局而不同，但 Kubernetes Service 的
`targetPort` 和健康检查必须能够正确访问每个 Pod。

## B. 允许在滚动升级期间短暂不一致

这些参数不改变共享记录和 fencing 的含义，但会改变外部行为或任务策略。正常情况下
仍建议所有副本最终收敛到相同值。滚动升级期间允许新旧版本短暂共存。

| 配置 | 短暂不一致时可能出现的现象 |
|---|---|
| `DINGO_VIDEO_WORKER_RETRY_ONCE` | Worker 故障后，有的任务重试，有的任务直接失败 |
| `DINGO_VIDEO_RETRY_BUDGET` | 不同 Gateway 对共享重试额度是否已满采用不同阈值 |
| `DINGO_VIDEO_RETRY_WAIT_TIMEOUT_S` | 重试任务的最长等待时间不同 |
| `DINGO_VIDEO_RETRY_FAILED_INSTANCE_BACKOFF_S` | 不同 Gateway 对刚失败 Worker 的避让时长不同 |
| `scheduling.queue_limit` | 相同负载下，请求打到一个 Gateway 被接收，打到另一个返回 429 |
| `scheduling.accept_without_workers` | 无已注册 Worker 时，不同 Gateway 的准入决定不同 |
| `scheduling.execution_timeout_s` | 新旧 Gateway 对执行超时的判断时间不同 |
| `scheduling.abort_grace_s` | 取消或超时后等待 Worker 确认的时长不同 |
| `scheduling.finalization_timeout_s` | 后处理被认定为超时的时间不同 |
| `scheduling.finalization_max_retries` | 后处理失败后的尝试次数不同 |
| `scheduling.worker_capacity` | 新旧 Gateway 使用 Worker 并发容量的上限不同 |
| `scheduling.worker_prefetch_capacity` | 新旧 Gateway 是否使用 Worker 预取额度不同 |
| `scheduling.early_release_slot` | 新旧 Gateway 释放执行 slot 的时点不同 |
| `http.max_body_bytes` 及 `media.*` 限制 | 同一个输入可能被一个 Gateway 接收、被另一个拒绝 |
| `http.sync_timeout_s` | 同步请求等待时长不同；异步任务状态不受影响 |
| `http.async_submit_status_code` | 提交成功时可能短暂混用 HTTP 200 和 202 |
| `http.default_model`、`pools[*].served_models` | 新增或移除模型期间，部分 Pod 可能暂时不识别该模型 |
| `lifecycle.*` 保留及扫描参数 | 任务和垃圾制品的回收时间可能短暂不同 |
| `artifact_store.hard_min_free_bytes`、`soft_min_free_bytes` | 不同 Gateway 可能做出不同磁盘准入决定 |

这一类差异的接受边界如下：

- 允许行为和性能短暂不一致，但不能绕过共享 lease、CAS 和 fencing。
- 旧 Gateway 创建的任务被新 Gateway 接管时，以接管者当前策略为准。当前系统不承诺
  “任务创建时的策略永久固定”。如果未来需要该语义，应把策略快照写入任务记录。
- 滚动升级完成后必须检查所有 Ready Pod 已使用目标配置，不应长期保留无意的漂移。
- 调整 `worker_capacity` 或 prefetch 时，Gateway 仍必须以 Worker 实际声明容量为上限；
  不允许通过配置绕过容量握手。
- 改动 media 上限时，不得超过代码中的协议 hard cap。

这些差异本身不会使同一任务产生两个有效成功终态，不会破坏 etcd，也不会允许旧
attempt 覆盖新 attempt。Worker 失联且实际仍在计算时，重试可能造成短暂的重复物理
计算，但只有通过当前 fencing 校验的执行结果可以成为有效结果。

## C. 必须始终一致或保持协议兼容

以下配置定义了多个 Gateway 共同操作的数据和协议。服务同一个逻辑部署的 Pod 不应
在这些项目上使用互不兼容的值。

| 配置或协议 | 要求 | 不一致的后果 |
|---|---|---|
| `schema_version` | 必须为当前代码支持的同一配置 schema | Pod 可能无法启动或错误解释配置 |
| `deployment_id` | 同一逻辑部署必须相同 | 形成相互不可见的任务和 artifact 空间 |
| `task_store.kind` | 必须使用兼容实现 | Memory store 不能替代生产 etcd 的共享语义 |
| `task_store.prefix` | 必须相同 | Gateway 会看到不同的任务、队列、lease 和 retry ledger |
| etcd 集群身份 | 必须指向同一集群 | 即使 prefix 相同也会形成 split-brain |
| etcd task、index、lease、retry ledger 的 key/value schema | 新旧代码必须双向兼容 | 可能无法接管任务或正确维护计数器 |
| 任务状态机和 CAS 前置条件 | 新旧代码必须兼容 | 可能错误拒绝合法转换；不得放宽安全条件 |
| attempt、owner generation、execution token 语义 | 必须兼容 | 这是阻止旧执行写回和错释放 slot 的 fencing 边界 |
| Worker slot key 和 slot 编号规则 | 必须兼容 | 可能对同一执行资源形成不同身份认知 |
| `artifact_store.kind` 及逻辑存储 | 必须指向同一份共享制品存储 | 接管任务的 Gateway 可能找不到输入或结果 |
| `artifact_store.root` 对应的逻辑路径布局 | 所有 Pod 和 Worker 必须能解析到同一制品 | 可能出现结果存在但查询、下载或清理失败 |
| `runtime.request_plane`、`event_plane` | 必须与 Worker 和新旧 Gateway 协议兼容 | 请求无法发送、结果事件无法接收 |
| `runtime.discovery_backend` | 同一 pool 必须发现同一组 Worker 身份 | Gateway 的 Worker 视图可能分裂 |
| `pool_id` 的含义 | 同一个 ID 必须始终代表同一个逻辑池 | 队列、slot 和重试账本可能被不同业务混用 |
| `backend_model`、`backend_target` | 同一 pool 内必须保持语义一致 | 任务可能发送到错误模型或错误 endpoint |
| `execution_mode` | 新旧 Gateway 和 Worker 必须兼容 | 生命周期、等待和取消协议可能不匹配 |
| adapter 名称、workflow、协议版本和 wire-format options | 新旧两端必须兼容 | 输入或 Worker 返回值可能被错误解释 |

如果需要对这一类配置做不兼容变更，应采用新的 `deployment_id`、`pool_id` 或独立
task-store prefix 做蓝绿部署，而不是让不兼容版本同时操作同一个共享空间。确认旧任务
已排空后，再切换流量并清理旧空间。

## `configuration_revision` 的作用与限制

当前每个任务记录 `configuration_revision`，用于阻止配置语义不同的 Gateway 盲目恢复
排队任务。当前 revision 覆盖：

- `pool_id`、served models；
- backend model 和 target；
- execution mode；
- adapter workflow、协议版本及 options；
- `worker_capacity`；
- `execution_timeout_s`。

它是一道恢复保护，不是完整的集群配置协调器。当前重试环境变量、prefetch、
`early_release_slot` 和多数本机参数不在 revision 中。因此：

- 不应把 revision 相同理解为所有 Gateway 参数逐项相同；
- 不应仅依赖 revision 发现运维配置漂移；
- 共享协议的不兼容升级仍应使用蓝绿部署；
- 普通滚动升级结束后仍应核对 Pod 镜像、ConfigMap hash 和关键环境变量。

## 运维检查清单

滚动升级前：

1. 判断改动属于 A、B、C 哪一类。
2. C 类不兼容改动改用蓝绿部署，不进行原地混跑。
3. 确认新旧版本使用相同 etcd 集群、prefix、deployment ID 和共享 artifact store。
4. 确认 Worker wire protocol 与新旧 Gateway 均兼容。

滚动升级期间：

1. 允许 B 类行为短暂不同，并关注 429、失败、重试和超时指标。
2. 不单独修改某个 Pod 来绕过 Worker 声明容量或 fencing。
3. 若出现任务恢复失败，首先比较镜像、配置 revision、pool 定义和 etcd prefix。

滚动升级后：

1. 确认所有 Ready Gateway 使用目标镜像和 ConfigMap hash。
2. 确认 B 类参数已经收敛；刻意保留的差异应有运维记录。
3. 确认 queue、retry credits、Worker leases 和 finalization backlog 能正常回落。
4. 至少执行一次提交、查询、下载、删除以及 Worker 故障重试的冒烟验证。


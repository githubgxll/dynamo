---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Video Gateway 与 Omni Worker 简明机制
subtitle: 服务组件、任务状态、执行与预取槽位、请求流程
---

Video Gateway 负责接收、排队、调度和交付视频任务；Omni Worker 负责运行模型并生成视频。下面以 **detached 模式、1 个执行槽位 + 1 个预取名额、开启槽位提前释放**为例，说明两者如何配合。预取和提前释放需要显式开启，并非默认行为。

## 各个服务组件做什么

| 组件 | 主要功能 |
|---|---|
| 负载均衡／Ingress | 提供统一入口，把请求分给可用的 Gateway 副本。 |
| Video Gateway | 校验请求，按 `model` 找到对应 Worker Pool；管理任务排队、状态、取消，调度 Worker，校验并发布结果。 |
| Omni Worker | 接受任务并控制执行数量；准备输入、调用 vLLM-Omni 模型、编码视频，把执行状态和结果写入共享存储。 |
| etcd | 保存任务状态、队列和槽位占用；协调多个 Gateway，避免重复抢占同一任务或槽位。 |
| 共享制品存储 | 保存上传素材、Worker 执行状态和生成的 MP4，让不同 Gateway／Worker 访问同一份文件。 |
| SeedVR2 服务（可选） | 在复合 Worker 内接续 H3 Base 视频做超分，再交付最终视频。 |
| Prometheus／Grafana | 收集和展示队列、执行、预取、结果处理及容量指标。 |

Gateway 内部通过 **Dynamo Runtime** 发现 Worker 并向选定实例发送请求。Runtime 嵌入各进程，不是额外部署的一台中央路由服务。

**多模型：** 一个 Gateway 可以配置多个 Worker Pool，根据请求的 `model` 路由到不同模型别名或工作流，各池独立扩容。当前 Gateway adapter 支持 MiniMax-H3；新增模型家族还需接入适配器。

**高可用：** 多个 Gateway 同时服务，共享 etcd 和制品存储；一个副本退出后，其他副本可查询任务、接管符合条件的任务并提供下载。etcd 和共享文件系统也需要自身的可用性保障。

完整拓扑见[独立部署架构图](../assets/img/video-gateway-omni-deployment.svg)。

## 视频任务有哪些主要状态

正常链路的内部状态为：

```text
queued → dispatching → in_progress → finalizing → completed
 排队       下发           执行链路        结果处理       可下载
```

| 内部状态 | 含义 | 客户端看到的状态 |
|---|---|---|
| `queued` | 已持久化，等待 Gateway 分配 Worker。 | `queued` |
| `dispatching` | 已预留 Worker 槽位，正在准备和下发请求。 | `in_progress` |
| `in_progress` | 任务进入执行链路，可能在 Worker 预取等待，也可能正在生成、编码或写结果。 | `in_progress` |
| `finalizing` | Worker 已产出结果，Gateway 正在校验和发布。 | `in_progress` |
| `completed` | 最终视频已发布，可以下载。 | `completed` |

其他主要转变：

- 排队、下发、执行或结果处理失败，进入 `failed`。
- 活跃任务收到取消请求并完成相应处理后，进入 `cancelled`。取消成功不代表 GPU 计算立即停止。
- 完成、失败或取消后的任务，保留期结束或主动清理后可进入 `expired`。
- 启用有限重试策略时，符合条件的 Worker 失联任务可以重新排队，以新的执行身份再尝试。

**Worker 完成与公开任务完成不是同一时刻。** Worker 的 `completed` 表示执行结果已落盘；Gateway 的 `completed` 表示结果已完成交付处理。客户端看到 `in_progress` 时，不能据此判断 GPU 一定正在计算。

## Worker 槽位与预取如何配合

### 执行容量与接纳容量

“槽位”是任务容量名额，不是一张 GPU。一份任务可能使用多张 GPU。

| 概念 | 作用 |
|---|---|
| 执行容量 N | Worker 最多同时运行多少个任务执行链路；包括模型调用及相应输出处理。 |
| 预取容量 P | 执行容量之外，额外提前接收多少个等待任务；可以提前准备输入，当前 P 为 0 或 1。 |
| 接纳容量 N+P | Worker 可以接收的任务总数；多 Gateway 通过 etcd 槽位预留共同遵守这个上限。 |

Gateway 先与 Worker 协商容量，再预留名额；Worker 内部用执行许可决定哪些任务真正开始执行。预取任务拿到许可后转为执行，**不是固定把某个 slot 编号绑定为“GPU 槽”或“预取槽”**。

例如 N=1、P=1：最多接受两个任务，其中一个执行，一个等待；第二个任务不能因为已经被接收就同时运行模型。未开启预取时默认 N=1、P=0。

### 三个任务的交替过程

假设 A、B、C 发往同一个 Worker：

| 时刻 | 执行位置 | 预取等待位置 | Gateway 侧 |
|---|---|---|---|
| A 执行期间 | A | B 已接受，可准备输入 | C 仍在队列 |
| A 结果已落盘，Worker 执行许可释放 | B 可以开始执行 | 暂时空出 | A 等待或正在结果交接；C 要等 Gateway 释放预留名额 |
| Gateway 持久化 A 的结果交接并提前释放名额后 | B | C 可以被派发并接受 | A 处于 `finalizing` |
| A 结果发布后 | B 或后续任务 | 后续任务 | A 变为 `completed`，可下载 |

这里有两个释放时点：**Worker 释放内部执行许可**，让 B 开始；**Gateway 提交持久化结果交接并释放 etcd 预留名额**，让 C 被接纳。二者配合，才能让下一任务执行与上一任务的 Gateway 结果处理重叠。

预取的作用是减少任务切换等待；提前释放的作用是减少结果处理对后续派发的阻塞。两者都不会自动增加模型并发数或保证 GPU 吞吐提升。

## 一条请求如何执行

以异步接口为例：

1. **提交。** 客户端向 `POST /v1/videos` 发送 model、生成参数和素材；负载均衡选择一个 Gateway。
2. **创建任务。** Gateway 校验请求和素材，按 model 选 pool，将输入写入共享存储，并在 etcd 中创建 `queued` 任务；返回 task ID。
3. **选择 Worker。** Gateway 找到能力匹配且有接纳名额的 Worker，在 etcd 中原子预留任务与槽位，进入 `dispatching`，再向该实例下发请求。
4. **接受和读取素材。** Worker 校验共享任务目录并持久化接收状态，后台执行不再依赖最初的提交连接。使用共享文件输入时，Worker 根据请求中的文件描述，从共享目录读取输入清单和素材，校验后暂存到本地；这可与等待执行许可重叠。若使用 inline 输入，素材由 Gateway 编码进请求传给 Worker。
5. **生成视频。** 输入准备和执行许可都就绪后，Worker 解码素材、构造模型输入，调用 vLLM-Omni，编码并将 MP4 写入共享目录。若启用超分，同一次执行继续完成 SeedVR2 处理后才返回最终结果。
6. **交接结果。** Worker 发布持久化完成状态和文件描述符。Gateway 将结果引用写入 etcd，任务转为 `finalizing`；启用提前释放时，此处释放该任务的预留名额。
7. **发布结果。** Gateway 按结果描述符从共享目录读取 Worker 生成的 MP4，校验文件及媒体信息，完成必要的后处理，发布最终 MP4，将任务改为 `completed`。
8. **查询和下载。** 客户端用 task ID 查询状态，完成后从 `/v1/videos/{task_id}/content` 下载。任一连接同一任务存储和制品存储的 Gateway 副本均可处理。

同步接口 `/v1/videos/sync` 复用上述任务流程，只是 HTTP 请求等待结果后直接返回 MP4。更多配置、恢复与协议细节见[详细设计文档](detailed-design.zh-CN.md)。

# Worker 租约时间常量耦合说明

本文记录 Worker 执行租约相关的时间常量及其联动约束。这些值当前固定且相互匹配，
不计划调整；本文的目的不是推动修改，而是防止未来对其中某一个值做孤立改动。

## 常量清单

| 常量 | 当前值 | 位置 | 作用 |
|---|---|---|---|
| `_GATEWAY_OWNER_TTL_S` | 15s | `dispatcher.py` | Gateway owner 租约 TTL；keepalive 每 TTL/3=5s；连续失败累计满 TTL 触发 fatal restart |
| `_WORKER_LEASE_HEARTBEAT_INTERVAL_S` | 5s | `dispatcher.py` | Worker 执行租约的心跳间隔 |
| 心跳失败阈值 | 2 次 | `dispatcher.py` `_heartbeat` | 连续失败 2 次判 `_WorkerLeaseLost`，停止本地执行 |
| `owner_expires_at_ms` 记账 | `now + 15_000` | `dispatcher.py` reserve/recovery 两处；`task_store.py` 两个 `heartbeat_lease` | lease 记录上的逻辑过期时间（纯记账字段） |
| `execution_lease_ttl_s` | 15（下限 5） | `task_store.py` `EtcdTaskStore.__init__` | 原生 etcd 租约 TTL；heartbeat key 挂在它上面。当前未暴露为 Gateway 配置项，只能改代码 |
| `_DETACHED_WORKER_STALE_S` | 20s | `dispatcher.py` | Worker 状态文件陈旧判定（Worker 每 5s 刷新，容许丢 4 拍） |
| `scheduling.abort_grace_s` | 15s | `config.py` | 取消确认宽限；quarantine 复用时点 = 任务 deadline + 该宽限 |

## 必须保持的不变量

1. **止损先于复用**：心跳间隔（5s）× 失败阈值（2）= 10s，必须小于原生租约 TTL（15s）。
   当前余量 5s。含义：本地执行必须在 slot 可能被他人复用之前停止。TTL 降到 10s 及以下
   会出现"租约已过期被复用、本 Gateway 仍在心跳重试"的窗口，同一 slot 可能双重占用。
2. **keepalive 节奏装进 owner TTL**：Gateway owner 每 TTL/3（5s）keepalive，单次 etcd
   请求超时（默认 5s）与该节奏叠加后仍须在 TTL 内留有余量。
3. **HA 接管时延下限 = TTL**：`claim_orphaned_active` 要求 heartbeat key 已随原生租约
   自然消失才允许接管。TTL 调大则故障接管和 owner fatal restart 都变慢，恢复 SLA 变差；
   调小则先破坏不变量 1。
4. **相邻计时器一并复核**：Worker 状态文件心跳（5s/陈旧 20s）与 `abort_grace_s`（15s）
   不由 TTL 推导，但调整 TTL 或心跳参数时应一并复核。

## 修改 `execution_lease_ttl_s` 时的联动清单

- 调小（如 5s）：先破坏不变量 1（止损 10s > 租约 5s），产生 slot 双卖窗口；同时
  `dispatcher.py` 与 `task_store.py` 中硬编码的 `owner_expires_at_ms = now + 15_000`
  记账与真实租约脱节。注意 `claim_orphaned_active` 使用的是该参数本身，而两个
  `heartbeat_lease` 实现和 dispatcher 的 reserve/recovery 路径是硬编码 15s，不会随
  参数自动变化。
- 调大（如 60s）：不变量 1 更安全，但 HA 接管与 owner fatal restart 时延同步变大。
- 任一方向都必须同步复核：心跳间隔与失败阈值、owner keepalive、四处
  `owner_expires_at_ms` 写入点、Worker 状态文件陈旧阈值、取消宽限。

## 当前决策

2026-09-21 确认：上述值相互匹配且已经过故障注入验证，保持现状不做参数化推导。
如确需调整，按上节清单整体联动修改并重新做 Worker 失联、HA 接管故障注入验证。

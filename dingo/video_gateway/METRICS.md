# Multi-slot capacity and statistics

## Task-level performance analysis

`DINGO_VIDEO_PERFORMANCE_RECORDS=1` adds a versioned `performance` object to
existing completed/failed/cancelled lifecycle log records. It defaults off.
The summary contains generation parameters, reference counts, known reference
dimensions and upload sizes, current-attempt model timings and common task
timings. Prompt text, file paths and media contents are omitted. No additional
file probing or etcd lookup is performed. Missing sizes/frame counts stay null.
Reference FPS multiplied by duration is not reported as an exact frame count.

The test monitoring project collects these logs into a separate SQL store for
Grafana parameter filters. These task dimensions are not Prometheus labels.
The collector deduplicates `(deployment,task_id,attempt)`; collection gaps and
ephemeral-store loss prevent treating this as a complete audit record.

## Model-stage timings

`dingo_video_model_stage_duration_seconds` is a Histogram with bounded
`pool`, `schema` and `stage` labels. It aggregates the existing
`metrics.model_execution` payload when a successful result handoff transitions
the current attempt from `in_progress` to `finalizing`. Subsequent finalization
does not observe the sample again. Missing/invalid stages are omitted, not zero.
There is no task-ID label, additional etcd query, or historical-task scan.
These are Worker timings reported by Gateway, not Gateway processing times;
parent and child stages may overlap and must not be summed.

Across Gateway replicas, average latency is
`sum by(pool,schema,stage)(rate(dingo_video_model_stage_duration_seconds_sum[5m])) /
sum by(pool,schema,stage)(rate(dingo_video_model_stage_duration_seconds_count[5m]))`.
P95 uses `histogram_quantile(0.95, sum by(pool,schema,stage,le)
(rate(dingo_video_model_stage_duration_seconds_bucket[5m])))`.
Samples are recorded at handoff time, not stage start time. Counters are local
and restart with the Gateway; unobserved events during crashes may be lost.
No retrospective backfill or exact audit-count guarantee is provided.

All metrics below use the `dingo_video_` prefix and a bounded `pool` label.

| Metric suffix | Meaning |
| --- | --- |
| workers | Discovered physical Worker registrations, not Pods or slots |
| worker_busy | Distinct registered Workers with a non-quarantined lease |
| worker_execution_capacity | Negotiated execution concurrency, clamped to Gateway configuration |
| worker_prefetch_capacity | Negotiated additional prefetch admission capacity, not compute capacity |
| worker_admission_capacity | Execution plus prefetch capacity |
| worker_slots_busy | Non-quarantined leases mapped to current admission slots |
| worker_slots_quarantined | Quarantined leases mapped to current admission slots |
| worker_slots_free | Unleased admission slots; zero if discovery/lease-watch view is unhealthy |
| worker_unmapped_leases | Leases outside current slot map, e.g. disappeared registrations or unknown/reduced capacity |
| worker_capacity_view_healthy | Whether local discovery and the shared lease view are healthy |
| worker_execution_capacity_configured | Per-Worker execution ceiling, not pool total |
| worker_prefetch_capacity_configured | Per-Worker prefetch ceiling |
| early_release_slot_enabled | Local Gateway feature flag |
| finalization_pending_local | This Gateway's tracked finalizers, including waiting/running work |
| finalization_concurrency_configured | Local finalizer concurrency limit |
| finalization_pending_limit_configured | Local finalizer pending limit |

`worker_busy` previously counted leases: for a multi-slot Worker this overstated
the number of physical busy Workers. Consumers needing lease occupancy should
switch to `worker_slots_busy`. Single-slot registered-Worker behavior is unchanged.
Disappeared registrations are excluded from physical busy count; their leases
remain visible in `worker_unmapped_leases` until recovered.

Example: one Worker with N=2/P=1 has execution capacity 2 and admission capacity 3.
With two active leases, physical busy=1, slot busy=2 and slot free=1. This does NOT
prove that two model computations are currently running: a lease covers Worker
queuing, execution and output publication. Slot IDs are interchangeable; the
highest slot is not a permanently designated prefetch task. Similarly, task
`in_progress` and its `execution_started` lifecycle boundary are Gateway execution
states, not GPU kernel start measurements. Exact engine-running/queued counts
require Worker-side telemetry; do not infer them from admission leases.

Early result handoff releases the lease before Gateway finalization. Historical
task Worker assignments must not count as current occupancy. Shared task counts
by status still include finalizing tasks; finalization_pending_local is a local
scheduler/backpressure view, not a replacement for that shared task count.

Capacity snapshots reuse discovery capability caches and lease snapshots; scraping
does not call every Worker. Unknown multi-slot capacity contributes zero until
probed. Legacy fallback contributes at most one slot. A fully busy model remains
available for queuing; a model with no negotiated slots is unavailable.

In HA, each Gateway exposes a view of the SAME shared pool and task counts.
Do not sum these gauges across Gateway replicas: deduplicate by deployment/pool
(e.g. max for a dashboard, checking view health and agreement). Local finalizer
and process-memory metrics are per replica. Free slots are observational, not an
admission guarantee: quarantine, memory, retries and finalization backpressure
may still block dispatch; etcd reservation CAS remains authoritative.

Compare configured gauges across Gateways to detect configuration skew. Prefetch
and early-release flags intentionally retain the existing task configuration
fingerprint; changing it would fail queued tasks during recovery. This is not
permission to mix incompatible binaries: upgrade all Gateways before enabling
handoff, and drain durable handoffs before downgrading to older code.

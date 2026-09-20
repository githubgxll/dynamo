# Detached Worker capacity and continuous execution

These are opt-in deployment settings, not public HTTP request parameters.
They do not claim that a model gains throughput from GPU concurrency.

## Enable

Upgrade all Gateways to a version that understands durable result handoff before
enabling it. Upgrade Workers before requesting multiple execution/prefetch slots.
Gateway and Worker must share the same artifact filesystem and task root.

For one executing request and one prefetched request per physical Worker:

```yaml
# Within a pool using execution_mode: detached
scheduling:
  worker_capacity: 1
  worker_prefetch_capacity: 1
  early_release_slot: true
  finalization_concurrency: 4
  finalization_pending_limit: 64
  finalization_timeout_s: 60
  finalization_max_retries: 2
  finalization_retry_delay_s: 0.1
```

Start the Omni Worker with `--max-num-seqs 1` and
`--detached-video-prefetch-capacity 1`, plus its existing detached task-root
arguments. Set `DINGO_VIDEO_BINARY_RESULTS=1` and `DINGO_VIDEO_INLINE_RESULT=1`.
The first enables durable MP4 descriptors; the second avoids an extra response
JSONL file. Neither enables an arbitrary-path public upload/download API.

For MiniMax-H3 with N > 1, set matching `worker_capacity` / `--max-num-seqs`, and
enable `--step-execution`. The current integration rejects Cache-DiT with step
execution. The extra prefetch slot does not increase engine concurrency: N=2/P=1
means at most two executions and one waiting task, not three model executions.
Verify memory capacity and model behavior separately before increasing N.

Gateway negotiates capacity per physical registration and uses separate etcd
slot leases. Failed/legacy capability probes never infer N slots from config:
fallback is at most one. Two Gateways share reservations; their capacity gauges
must not be summed. See [metrics](METRICS.md).

## Result boundary and failures

Worker writes and synchronizes the MP4 before publishing a terminal descriptor.
Gateway then atomically persists the current attempt's descriptor, enters
`finalizing`, releases that task's slot and model-retry credit, and wakes dispatch.
Independent result processing subsequently publishes `completed`.

A transient finalization error retries only result processing, with a persisted
retry count and deadline. Permanent validation failure or timeout fails that
task; it does not run the model again or release a reused slot. Cancellation
targets the task/attempt/token. A replacement Gateway can claim a durable
handoff without a live Worker. Before the handoff, Worker loss still follows
the configured one-retry policy; parameter/OOM/ordinary execution errors are
not automatically retried.

Finalizer concurrency and pending limits are per Gateway, not cluster totals.
When its pending threshold is reached, that Gateway pauses new dispatch; work
already admitted can still finish. Memory admission remains independently
bounded. Size queue/retry/memory budgets for N+P admissions and pending results.
These controls cannot make permanently blocked filesystem calls cancellable.

## Cancellation boundary

Video task cancellation is a logical result and scheduling operation. After a
DELETE request, Gateway signals the detached Worker and waits up to
`abort_grace_s` for its durable terminal status. A confirmed `cancelled` status
releases the task's lease immediately. If the Worker does not confirm, Gateway
quarantines the old slot through the task deadline plus the abort grace. A
restarted Worker registers with a new instance ID and is not blocked by the old
instance's quarantine.

Engine compute interruption is backend-dependent. In particular, a vLLM-Omni
orchestrator can acknowledge an abort and unwind the outer request while an
already-running diffusion forward continues to its next interruptible boundary.
DingoRouter still prevents the cancelled attempt from publishing a result. With
engine concurrency one, a following task remains serialized inside vLLM-Omni
until that forward returns, even though its Gateway lifecycle has entered
`in_progress`.

This behavior is an accepted integration boundary, not a task-state correctness
failure. `execution_started` means that the detached Worker admitted the task;
it is not proof that a GPU kernel started. `worker_queue_wait_s` measures the
detached Worker's admission queue and does not include an opaque queue inside
the engine. Immediate reclamation of in-flight diffusion compute requires the
engine to implement interruption within its execution loop. Do not use a
successful DELETE response or a `cancelled` task status as proof that GPU use
has already fallen to zero.

## Disable or roll back

Drain accepted work and durable `finalizing` handoffs before downgrading to code
that does not understand this protocol. Disabling prefetch/early release does
not intentionally change the task configuration fingerprint, but that is not
permission to mix incompatible Gateway binaries. Keep all Gateways' pool
configuration aligned and check both shared ledgers and local finalizer gauges.

The default is N=1/P=0 with early release off. The native frame converter is a
separate optimization: `DINGO_VIDEO_FRAME_CONVERSION_WORKERS` defaults to 8,
accepts integers 1–16, and 1 selects the legacy conversion path. It affects CPU
frame conversion only, not model execution slots or generated media parameters.

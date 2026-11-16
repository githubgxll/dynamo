# Finite Worker task runtime

Video generation, standalone SR and Context-IR share the dispatch and execution
control components in this package. Model execution and public API records are
provided by business adapters.

## Gateway integration

Implement `DispatchAdapter` and use `DispatchCoordinator` with an
`ExecutionSupervisor`:

| Adapter method | Responsibility |
|---|---|
| `candidates()` | Return eligible queue entries and physical slot choices in business-policy order. A candidate does not own capacity. |
| `key(candidate)` | Stable local task key, shared by normal dispatch and takeover. Include a business prefix when IDs may overlap. |
| `claim(candidate)` | Acquire preparation budget and atomically claim or adopt the persisted execution. Return `DispatchClaim(record)` only after confirmation. An empty decision means blocked; `retry_now=True` permits another turn after a CAS race. |
| `committed(candidate, record)` | Synchronous cache and telemetry updates after confirmed ownership. Must never call the model. |
| `release_preparation(candidate)` | Release only this candidate's local preparation allocation when no execution was acquired. Does not delete physical slots. |
| `execute(record)` | Track the owned attempt through the existing protocol, handle its business results and settle its resources. |

The coordinator owns select/claim/launch, prevents concurrent local claims for the
same task, and retains confirmed claims when local task creation fails. Its
`claim_one()` entry also supports orphaned-attempt adoption. The supervisor owns
background controllers independently of request connections, retrieves escaped
exceptions, and prevents an old completion callback from deleting a new
controller with the same key. Native completion evidence remains the business
adapter's responsibility.

Recovery scans must check `coordinator.manages(key)`, which includes in-flight
claims, unconfirmed claims, confirmed-but-not-launched work and running
controllers. Submit adoption through `claim_one()` as well. Checking only
running controllers and launching directly can bypass confirmation of a lost
claim/adoption reply.

`VideoSchedulingAdapter` serves both H3 and SR, preserving retry priority,
instance exclusions, memory budget, prefetch and early release. The compiler's
`CompilerSchedulingAdapter` preserves capability, route and physical-slot rules.
Business-specific protocol observers and result handling remain in their
respective adapters. The common coordinator contains no model-name branches.

## Storage and unknown outcomes

Existing key layouts and codecs stay readable. Runtime v2 snapshots and CAS
builders in `repository.py` / `transactions.py` apply to v2 slot records; video's
legacy lease and handoff records keep their own atomic write builders. Do not
map a legacy video lease into a v2 physical slot merely to reuse a transaction.

Both use `execute_fenced_transaction()`:

1. Execute one immutable compare/write plan.
2. On a lost reply, invoke the adapter's `confirm()` using a consistent MVCC
   snapshot and the original execution identity.
3. If still unknown, raise `UnconfirmedTransaction`. The coordinator retains its
   `resolve()` operation and preparation budget without starting a model.
4. Resolution first reads back, and may resend the **same revision-fenced CAS**.
   It never generates another execution token, attempt or native lease. A
   confirmed conflict releases preparation; confirmed ownership launches the
   original attempt.

For state mutations outside claim, callers must reconcile
`UnconfirmedTransaction` before treating the mutation as failed or releasing
business artifacts. `confirm_record_write()` checks a business receipt and
optional dedicated identity witnesses in one snapshot. Shared counters are
updated atomically but are unsuitable receipts because another task can change
them immediately after commit.

The exact structured etcd rejection `code=5` / `etcdserver: requested lease not
found` on a KV mutation is a definite non-commit, represented by
`EtcdLeaseNotFound`. It is a CAS conflict, not an unknown network outcome. A
fresh queued reservation can obtain a new lease without consuming a model
retry. For an earlier unknown mutation, reconciliation still checks the
original identity before resolving it; arbitrary HTTP 404s remain unknown.

Pipeline parent admission uses `TaskStore.start_pipeline_parent()`. Etcd counts
active status indexes at one MVCC revision and compares a per-pool start fence
in the same transaction that changes the parent status and dequeues it. This
prevents two Gateways from both passing a count-then-start check. Existing
active parents are counted without initializing a new counter, and terminal
transitions require no additional quota cleanup. Every Gateway that admits
parents must use this entry; an older count-then-start instance does not honor
the new fence during a mixed-version rollout.

A publicly cancelled task can still occupy physical capacity. Owner lease
expiration, an HTTP disconnect or a failed status read does not prove native
computation stopped. Preserve backend-specific completion, cancellation and
recovery policies, including compiler usage receipts and video handoffs.

## Verification

Video dispatch memory is estimated by `memory_budget.task_memory_bytes()` at
submission and claim/takeover. A task's persisted `shared_inputs` marker means
the descriptor fits in fixed task overhead; original reference bytes incur no
Base64 reservation. Legacy inline tasks keep two encoded input copies, and SR
adapters already report zero encoded source bytes. Old shared tasks carrying
larger persisted estimates are corrected when claimed without rewriting task
identity or input files. ACK-time shrink and terminal release retain the result
reserve. This process-local dispatch budget is separate from queue limits,
Worker RAM/GPU memory and Context-IR artifact storage quotas.

`test_gateway_coordinator.py` exercises the shared lifecycle with video, SR,
compiler and a separate CPU example policy. `test_fenced_mutation_recovery.py`
executes actual video/compiler transaction dictionaries with lost replies,
read-back outages, cancellation and takeover. Existing protocol, queue,
handoff, retry and public API tests remain part of `tests/video_gateway`.
`test_gateway_hardening.py` covers the actual compiler recovery scan, expired
lease rejection against an isolated real etcd, and concurrent parent admission
for memory, fake-etcd and real-etcd stores.

The companion Worker execution lifecycle lives in `dingo/worker/execution.py`.
This Gateway runtime does not change LLM streaming Workers or require migrating
video v1 to the compiler v2 wire protocol.


## Unified generation queues

Set `pipeline: {sr_pool: <sr-pool-id>}` on a detached H3 pool to route high-resolution requests through independent SR. Ordinary requests and base-generation children share that pool's queue and worker slots. Admission atomically persists the public parent and first queued child, charging one queue credit. Parents are orchestration records, never worker executions; unified pools reject `max_active_parents`.

Internal children carry `_pipeline_child` and are excluded from public submission, terminal and duration counts. Parents aggregate stage observations. Use `dingo_video_request_stage_active` for deduplicated user-request occupancy; `dingo_video_tasks` counts raw records. Use `dingo_video_pipeline_stage_duration_seconds{phase="base"|"sr"}` for multi-stage details, not as an additional public request count.

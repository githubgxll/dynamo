"""Small CAS building blocks shared by video and compiler task stores.

Resource comparisons and writes are appended to the SAME transaction by callers.
These helpers never retry an execution against a newer task revision.
"""
import json

from dingo.video_gateway.errors import EtcdLeaseNotFound, StoreUnavailable


class UnconfirmedTransaction(StoreUnavailable):
    """Keep the original mutation and identities for later reconciliation.

    resolve() returns the committed record or None after a definitive conflict.
    If still unknown it raises this exception again. It never allocates another
    token, native lease, attempt, or retry credit.
    """

    def __init__(self, resolve):
        super().__init__("task transaction outcome requires reconciliation")
        self.resolve = resolve


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sequence_token(sequence):
    if sequence <= 0 or sequence >= 10**20:
        raise ValueError("task created_seq is outside the supported range")
    return f"{sequence:020d}"


def compare_snapshot(client, key, value):
    if value is None:
        return [client.compare_version(key, 0)]
    return [
        client.compare_mod(key, value.mod_revision),
        client.compare_value(key, value.value),
    ]


async def read_counter(client, key, *, maximum=None):
    value = await client.get(key)
    count = int(value.value) if value is not None else 0
    if count < 0 or (maximum is not None and count >= maximum):
        raise RuntimeError("task counter outside supported range")
    return count, value


async def get_record(client, key, decoder):
    value = await client.get(key)
    return decoder(value) if value is not None else None


async def get_idempotent(client, key, digest, get_task):
    from dingo.video_gateway.errors import GatewayError

    value = await client.get(key)
    if value is None:
        return None
    index = json.loads(value.value)
    if index["request_digest"] != digest:
        raise GatewayError(
            409,
            "idempotency_conflict",
            "Idempotency-Key was already used with another request",
        )
    task = await get_task(index["task_id"])
    if task is None:
        raise RuntimeError("idempotency index points to a missing task")
    return task


def admission_transaction(
    client,
    *,
    task_key,
    task_json,
    sequence_key,
    sequence,
    sequence_value,
    queue_counter_key,
    queue_count,
    queue_counter_value,
    queue_writes,
    index_writes=(),
    idempotency_key=None,
    task_id=None,
    request_digest=None,
    compares=(),
    writes=(),
):
    compare = [
        client.compare_version(task_key, 0),
        *compare_snapshot(client, sequence_key, sequence_value),
        *compare_snapshot(client, queue_counter_key, queue_counter_value),
        *compares,
    ]
    success = [
        client.put(task_key, encode(task_json)),
        client.put(sequence_key, str(sequence + 1)),
        client.put(queue_counter_key, str(queue_count + 1)),
        *queue_writes,
        *index_writes,
        *writes,
    ]
    if idempotency_key is not None:
        compare.append(client.compare_version(idempotency_key, 0))
        success.append(
            client.put(
                idempotency_key,
                encode({"task_id": task_id, "request_digest": request_digest}),
            )
        )
    return compare, success


def _guard_records(client, records):
    """Compare both value and revision; reject aliasing keys before sending."""
    keys = [record.key for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("runtime transaction contains aliased keys")
    return [
        comparison
        for record in records
        for comparison in (
            client.compare_mod(record.key, record.revision),
            client.compare_value(record.key, record.value),
        )
    ]


def _fence_json(fence):
    from dataclasses import asdict

    return encode(asdict(fence))


def claim_transaction(
    client, *, task, worker, owner, route, queue, queued, slot_id, fence, task_json
):
    """Build one atomic dequeue/occupy/attempt transaction, without executing.

    The repository adapter checks task state, deadlines, route eligibility and
    worker health before calling. Those observations MUST be in the snapshots
    guarded here. Occupancy is deliberately not attached to a native lease.
    """
    from dingo.common.task_protocol import integer
    from dingo.video_gateway.task_runtime.repository import RuntimeKeys

    if fence.task != task.ref or fence.worker != worker.ref:
        raise ValueError("claim fence does not match snapshots")
    integer(slot_id, "slot_id", minimum=0, maximum=worker.capacity - 1)
    if queue.value != task.ref.task_id.encode():
        raise ValueError("queue points to a different task")
    try:
        count = int(queued.value)
    except ValueError as exc:
        raise ValueError("invalid queue counter") from exc
    if count <= 0:
        raise ValueError("queue counter underflow")
    keys = RuntimeKeys(task.ref.deployment_id)
    slot_key = keys.slot(worker.ref, slot_id)
    attempt_key = keys.attempt(fence)
    active_key = keys.active(task.ref)
    records = (task.record, worker.record, owner.record, route, queue, queued)
    if {slot_key, attempt_key, active_key} & {r.key for r in records}:
        raise ValueError("claim control keys overlap source records")
    compare = _guard_records(client, records)
    compare.extend(
        client.compare_version(k, 0) for k in (slot_key, attempt_key, active_key)
    )
    identity = _fence_json(fence)
    attempt = {
        "fence": json.loads(identity),
        "owner_generation": owner.generation,
        "owner_key": owner.record.key,
        "slot_id": slot_id,
        "state": "reserved",
    }
    writes = [
        client.put(task.record.key, encode(task_json)),
        client.delete(queue.key),
        client.put(queued.key, str(count - 1)),
        client.put(slot_key, identity),
        client.put(active_key, identity),
        client.put(attempt_key, encode(attempt)),
    ]
    return compare, writes


def settle_transaction(client, *, attempt, outcome, task_json):
    """Release only this attempt's slot after validated physical completion.

    Calling again with the same snapshots conflicts without mutating anything.
    The repository resolves that conflict by reading the terminal attempt; it
    must not repeat result publication or usage accounting.
    """
    from dataclasses import asdict

    from dingo.video_gateway.task_runtime.repository import RuntimeKeys

    keys = RuntimeKeys(attempt.fence.task.deployment_id)
    stored = json.loads(attempt.record.value)
    identity = _fence_json(attempt.fence).encode()
    if (
        attempt.record.key != keys.attempt(attempt.fence)
        or attempt.active.key != keys.active(attempt.fence.task)
        or attempt.slot.key != keys.slot(attempt.fence.worker, stored["slot_id"])
        or attempt.active.value != identity
        or attempt.slot.value != identity
        or stored["fence"] != json.loads(identity)
        or stored["owner_key"] != attempt.owner.record.key
        or stored["owner_generation"] != attempt.owner.generation
    ):
        raise ValueError("settlement fence or owner mismatch")
    current_task = json.loads(attempt.task.record.value)
    if (
        current_task.get("status") == "cancelled"
        and task_json.get("status") != "cancelled"
    ):
        raise ValueError("settlement cannot undo public cancellation")
    # ValidatedOutcome requires a physical terminal state. Do not offer a
    # generic release_slot helper that could bypass this requirement.
    if outcome.state not in {"succeeded", "failed", "cancelled"}:
        raise ValueError("physical execution is not terminal")
    stored.update(
        state=outcome.state,
        result=asdict(outcome.result) if outcome.result is not None else None,
        usage=json.loads(outcome.usage_json),
    )
    records = (
        attempt.task.record,
        attempt.record,
        attempt.slot,
        attempt.active,
        attempt.owner.record,
    )
    return _guard_records(client, records), [
        client.put(attempt.task.record.key, encode(task_json)),
        client.put(attempt.record.key, encode(stored)),
        client.delete(attempt.slot.key),
        client.delete(attempt.active.key),
    ]


async def execute_runtime_transaction(client, compare, writes):
    """Execute once; transport uncertainty MUST NOT become a negative ACK.

    No retry or read-back is hidden here. Repository adapters reconcile UNKNOWN
    using the expected fence before they may dispatch or delete input files.
    """
    from dingo.video_gateway.errors import StoreUnavailable
    from dingo.video_gateway.task_runtime.repository import CasResult, CasStatus

    try:
        applied, revision = await client.txn(compare, writes)
    except EtcdLeaseNotFound:
        # The server rejected this entire atomic request. When reconciling an
        # older uncertain request, execute_fenced_transaction still reads its
        # identity witnesses before concluding that it did not commit.
        return CasResult(CasStatus.CONFLICT, 0)
    except (StoreUnavailable, TimeoutError, ConnectionError):
        return CasResult(CasStatus.UNKNOWN)
    return CasResult(CasStatus.APPLIED if applied else CasStatus.CONFLICT, revision)


async def execute_fenced_transaction(client, compare, writes, *, committed, confirm):
    """Execute once and read back an ambiguous result through business witnesses.

    The callback checks one consistent etcd snapshot against the original
    execution identity, not just a terminal state. On uncertainty the coordinator
    retains this exact mutation; resolution first reads back, then may resend the
    identical revision-fenced CAS. A negative CAS plus failed confirmation is a
    definitive conflict. Network errors alone never mean not accepted.
    """
    from dingo.video_gateway.task_runtime.repository import CasStatus

    async def readback():
        try:
            return await confirm()
        except (StoreUnavailable, TimeoutError, ConnectionError):
            raise UnconfirmedTransaction(resolve) from None

    async def resolve():
        found = await readback()
        if found is not None:
            return found
        return await execute(reconciling=True)

    async def execute(*, reconciling=False):
        result = await execute_runtime_transaction(client, compare, writes)
        if result.status is CasStatus.APPLIED:
            return committed(result.revision)
        if result.status is CasStatus.CONFLICT and not reconciling:
            return None
        found = await readback()
        if found is not None:
            return found
        if result.status is CasStatus.CONFLICT:
            return None
        raise UnconfirmedTransaction(resolve)

    return await execute()


async def confirm_record_write(client, key, expected, decode, *, witnesses=()):
    """Confirm business data and identity witnesses in one MVCC snapshot.

    Only dedicated identity/result records belong in witnesses. Shared counters
    may legitimately change after commit and must not be used as a receipt.
    """
    values, _ = await client.get_many([key, *(k for k, _ in witnesses)])
    value = values[0]
    raw = expected.encode() if isinstance(expected, str) else expected
    if value is None or value.value != raw:
        return None
    for current, (_, wanted) in zip(values[1:], witnesses):
        wanted = wanted.encode() if isinstance(wanted, str) else wanted
        if wanted is None:
            if current is not None:
                return None
        elif current is None or current.value != wanted:
            return None
    return decode(value)

"""Small CAS building blocks shared by video and compiler task stores.

Resource comparisons and writes are appended to the SAME transaction by callers.
These helpers never retry an execution against a newer task revision.
"""
import json


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sequence_token(sequence):
    if sequence <= 0 or sequence >= 10**20:
        raise ValueError("task created_seq is outside the supported range")
    return f"{sequence:020d}"


def compare_snapshot(client, key, value):
    if value is None:
        return [client.compare_version(key, 0)]
    return [client.compare_mod(key, value.mod_revision),
            client.compare_value(key, value.value)]


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
        raise GatewayError(409, "idempotency_conflict",
                           "Idempotency-Key was already used with another request")
    task = await get_task(index["task_id"])
    if task is None:
        raise RuntimeError("idempotency index points to a missing task")
    return task


def admission_transaction(client, *, task_key, task_json, sequence_key,
                          sequence, sequence_value, queue_counter_key,
                          queue_count, queue_counter_value, queue_writes,
                          index_writes=(), idempotency_key=None, task_id=None,
                          request_digest=None, compares=(), writes=()):
    compare = [client.compare_version(task_key, 0),
               *compare_snapshot(client, sequence_key, sequence_value),
               *compare_snapshot(client, queue_counter_key, queue_counter_value),
               *compares]
    success = [client.put(task_key, encode(task_json)),
               client.put(sequence_key, str(sequence + 1)),
               client.put(queue_counter_key, str(queue_count + 1)),
               *queue_writes, *index_writes, *writes]
    if idempotency_key is not None:
        compare.append(client.compare_version(idempotency_key, 0))
        success.append(client.put(idempotency_key, encode(
            {"task_id": task_id, "request_digest": request_digest})))
    return compare, success

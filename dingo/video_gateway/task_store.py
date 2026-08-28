# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent task, FIFO queue, idempotency and Worker lease operations."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from dingo.video_gateway.errors import GatewayError, StoreConflict
from dingo.video_gateway.etcd_http import EtcdHttpClient, EtcdValue
from dingo.video_gateway.models import (
    ACTIVE_STATUSES,
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    StoredTask,
    TaskError,
    TaskStatus,
    VideoTask,
    WorkerLease,
    now_ms,
)

_INDEX_SCHEMA_VERSION = 3
_SEQUENCE_WIDTH = 20


@dataclass(frozen=True, slots=True)
class LeaseWatchEvent:
    revision: int
    worker_key: str | None
    lease: WorkerLease | None
    created: bool = False


@dataclass(frozen=True, slots=True)
class TaskWatchEvent:
    revision: int
    task_id: str | None
    deleted: bool = False
    created: bool = False


def worker_key(backend_target: str, instance_id: int | str) -> str:
    return hashlib.sha256(
        backend_target.encode() + b"\0" + str(instance_id).encode()
    ).hexdigest()


def _clone_task(task: VideoTask) -> VideoTask:
    return VideoTask.from_dict(task.to_dict())


def _apply_patch(task: VideoTask, patch: Mapping[str, Any]) -> VideoTask:
    result = _clone_task(task)
    for key, value in patch.items():
        if not hasattr(result, key):
            raise ValueError(f"unknown task field {key!r}")
        if key == "status" and not isinstance(value, TaskStatus):
            value = TaskStatus(value)
        if key == "error" and isinstance(value, Mapping):
            value = TaskError(**value)
        setattr(result, key, value)
    if (
        result.status != task.status
        and result.status not in ALLOWED_TRANSITIONS[task.status]
    ):
        raise StoreConflict(
            f"illegal task transition {task.status.value} -> {result.status.value}"
        )
    return result


class TaskStore(ABC):
    @abstractmethod
    async def health(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def prepare(self) -> None:
        """Prepare optional indexes before the Gateway becomes ready."""

        return None

    @property
    def lease_watch_supported(self) -> bool:
        return False

    @property
    def task_watch_supported(self) -> bool:
        return False

    @property
    def gateway_owner_supported(self) -> bool:
        return False

    async def lease_snapshot(
        self, pool_id: str
    ) -> tuple[dict[str, WorkerLease], int]:
        leases = await self.list_leases(pool_id)
        return {lease.worker_key: lease for lease in leases}, 0

    async def watch_leases(
        self, pool_id: str, *, start_revision: int
    ) -> AsyncIterator[LeaseWatchEvent]:
        del pool_id, start_revision
        raise NotImplementedError("this TaskStore does not support lease watches")

    async def task_watch_revision(self) -> int:
        raise NotImplementedError("this TaskStore does not support task watches")

    async def watch_tasks(
        self, *, start_revision: int
    ) -> AsyncIterator[TaskWatchEvent]:
        del start_revision
        raise NotImplementedError("this TaskStore does not support task watches")

    async def register_gateway(self, gateway_id: str, *, ttl_s: int) -> int:
        del gateway_id, ttl_s
        raise NotImplementedError("this TaskStore does not support Gateway owners")

    async def keepalive_gateway(self, lease_id: int) -> None:
        del lease_id
        raise NotImplementedError("this TaskStore does not support Gateway owners")

    async def unregister_gateway(self, lease_id: int) -> None:
        del lease_id
        raise NotImplementedError("this TaskStore does not support Gateway owners")

    async def iter_orphaned_active_tasks(self) -> AsyncIterator[StoredTask]:
        raise NotImplementedError("this TaskStore does not support Gateway owners")

    async def claim_orphaned_active(
        self, stored: StoredTask, *, new_owner_generation: str
    ) -> StoredTask | None:
        del stored, new_owner_generation
        raise NotImplementedError("this TaskStore does not support Gateway owners")

    @abstractmethod
    async def create_task(
        self,
        task: VideoTask,
        *,
        principal_hash: str,
        idempotency_hash: str | None,
        queue_limit: int,
    ) -> tuple[StoredTask, bool]: ...

    @abstractmethod
    async def get_task(self, task_id: str) -> StoredTask | None: ...

    @abstractmethod
    async def get_idempotent(
        self, principal_hash: str, idempotency_hash: str
    ) -> StoredTask | None: ...

    @abstractmethod
    async def list_tasks(
        self,
        *,
        pool_id: str | None = None,
        status: TaskStatus | None = None,
        after: str | None = None,
        limit: int = 100,
        descending: bool = False,
    ) -> list[StoredTask]: ...

    @abstractmethod
    async def list_all_tasks(self) -> list[StoredTask]: ...

    @abstractmethod
    async def list_queued(
        self, pool_id: str, *, limit: int = 100
    ) -> list[StoredTask]: ...

    @abstractmethod
    async def queue_depth(self, pool_id: str) -> int: ...

    @abstractmethod
    async def task_counts(self, pool_id: str) -> Mapping[TaskStatus, int]: ...

    @abstractmethod
    async def list_due_tasks(
        self, before_ms: int, *, limit: int = 256
    ) -> list[StoredTask]: ...

    @abstractmethod
    async def reserve(
        self,
        stored: StoredTask,
        lease: WorkerLease,
        *,
        deadline_at_ms: int,
    ) -> StoredTask | None: ...

    @abstractmethod
    async def transition(
        self,
        task_id: str,
        *,
        expected: Iterable[TaskStatus],
        patch: Mapping[str, Any],
        expected_revision: int | None = None,
        release_lease: bool = False,
        quarantine_until_ms: int | None = None,
    ) -> StoredTask: ...

    @abstractmethod
    async def request_cancel(
        self, task_id: str, *, terminal_expires_at_ms: int | None = None
    ) -> StoredTask: ...

    @abstractmethod
    async def list_leases(self, pool_id: str) -> list[WorkerLease]: ...

    @abstractmethod
    async def release_lease(self, pool_id: str, worker_key_value: str) -> None: ...

    @abstractmethod
    async def heartbeat_lease(
        self,
        pool_id: str,
        worker_key_value: str,
        task_id: str,
        lease_id: int | None = None,
    ) -> None: ...

    @abstractmethod
    async def delete_expired(self, stored: StoredTask) -> bool: ...

    @abstractmethod
    async def reconcile_pool(self, pool_id: str) -> None: ...


class MemoryTaskStore(TaskStore):
    """Deterministic in-process implementation used by core unit tests."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[str, tuple[VideoTask, int]] = {}
        self._queue: dict[str, list[str]] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, str]] = {}
        self._leases: dict[tuple[str, str], WorkerLease] = {}
        self._revision = 0
        self._created_seq = 0
        self._task_watchers: set[asyncio.Queue[TaskWatchEvent]] = set()

    def _next_revision(self) -> int:
        self._revision += 1
        return self._revision

    def _publish_task_event(
        self, task_id: str, revision: int, *, deleted: bool = False
    ) -> None:
        event = TaskWatchEvent(revision, task_id, deleted=deleted)
        for watcher in self._task_watchers:
            watcher.put_nowait(event)

    @property
    def task_watch_supported(self) -> bool:
        return True

    async def task_watch_revision(self) -> int:
        async with self._lock:
            return self._revision

    async def watch_tasks(
        self, *, start_revision: int
    ) -> AsyncIterator[TaskWatchEvent]:
        queue: asyncio.Queue[TaskWatchEvent] = asyncio.Queue()
        async with self._lock:
            self._task_watchers.add(queue)
            revision = self._revision
        try:
            yield TaskWatchEvent(revision, None, created=True)
            while True:
                event = await queue.get()
                if event.revision >= start_revision:
                    yield event
        finally:
            async with self._lock:
                self._task_watchers.discard(queue)

    def _stored(self, task_id: str) -> StoredTask | None:
        item = self._tasks.get(task_id)
        if item is None:
            return None
        return StoredTask(_clone_task(item[0]), item[1])

    async def health(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def create_task(
        self,
        task: VideoTask,
        *,
        principal_hash: str,
        idempotency_hash: str | None,
        queue_limit: int,
    ) -> tuple[StoredTask, bool]:
        task = _apply_patch(
            task,
            {
                "principal_hash": principal_hash,
                "idempotency_hash": idempotency_hash,
            },
        )
        async with self._lock:
            if idempotency_hash is not None:
                existing = self._idempotency.get((principal_hash, idempotency_hash))
                if existing is not None:
                    task_id, digest = existing
                    if digest != task.request_digest:
                        raise GatewayError(
                            409,
                            "idempotency_conflict",
                            "Idempotency-Key was already used with another request",
                        )
                    stored = self._stored(task_id)
                    if stored is None:
                        raise RuntimeError("idempotency index points to a missing task")
                    return stored, False
            queue = self._queue.setdefault(task.pool_id, [])
            if len(queue) >= queue_limit:
                raise GatewayError(429, "queue_full", "video queue is full")
            if task.id in self._tasks:
                raise StoreConflict(f"task already exists: {task.id}")
            self._created_seq += 1
            task = _apply_patch(task, {"created_seq": self._created_seq})
            revision = self._next_revision()
            self._tasks[task.id] = (_clone_task(task), revision)
            self._publish_task_event(task.id, revision)
            queue.append(task.id)
            if idempotency_hash is not None:
                self._idempotency[(principal_hash, idempotency_hash)] = (
                    task.id,
                    task.request_digest,
                )
            stored = self._stored(task.id)
            assert stored is not None
            return stored, True

    async def get_task(self, task_id: str) -> StoredTask | None:
        async with self._lock:
            return self._stored(task_id)

    async def get_idempotent(
        self, principal_hash: str, idempotency_hash: str
    ) -> StoredTask | None:
        async with self._lock:
            existing = self._idempotency.get((principal_hash, idempotency_hash))
            return self._stored(existing[0]) if existing is not None else None

    async def list_tasks(
        self,
        *,
        pool_id: str | None = None,
        status: TaskStatus | None = None,
        after: str | None = None,
        limit: int = 100,
        descending: bool = False,
    ) -> list[StoredTask]:
        async with self._lock:
            ordered_ids = [
                task_id
                for task_id, _item in sorted(
                    self._tasks.items(),
                    key=lambda item: (item[1][0].created_seq, item[0]),
                )
            ]
            values = [self._stored(task_id) for task_id in ordered_ids]
            filtered = [value for value in values if value is not None]
            if pool_id is not None:
                filtered = [
                    value for value in filtered if value.task.pool_id == pool_id
                ]
            if status is not None:
                filtered = [value for value in filtered if value.task.status == status]
            if after is not None:
                after_item = self._tasks.get(after)
                if after_item is None:
                    return []
                after_sequence = after_item[0].created_seq
                filtered = [
                    value
                    for value in filtered
                    if (
                        value.task.created_seq < after_sequence
                        if descending
                        else value.task.created_seq > after_sequence
                    )
                ]
            if descending:
                filtered.reverse()
            return filtered[:limit]

    async def list_all_tasks(self) -> list[StoredTask]:
        async with self._lock:
            ordered_ids = [
                task_id
                for task_id, _item in sorted(
                    self._tasks.items(),
                    key=lambda item: (item[1][0].created_seq, item[0]),
                )
            ]
            values = [self._stored(task_id) for task_id in ordered_ids]
            return [value for value in values if value is not None]

    async def list_queued(self, pool_id: str, *, limit: int = 100) -> list[StoredTask]:
        async with self._lock:
            result: list[StoredTask] = []
            for task_id in self._queue.get(pool_id, []):
                stored = self._stored(task_id)
                if stored is not None and stored.task.status == TaskStatus.QUEUED:
                    result.append(stored)
                    if len(result) >= limit:
                        break
            return result

    async def queue_depth(self, pool_id: str) -> int:
        async with self._lock:
            return len(self._queue.get(pool_id, []))

    async def task_counts(self, pool_id: str) -> Mapping[TaskStatus, int]:
        async with self._lock:
            counts = {status: 0 for status in TaskStatus}
            for task, _revision in self._tasks.values():
                if task.pool_id == pool_id:
                    counts[task.status] += 1
            return counts

    async def list_due_tasks(
        self, before_ms: int, *, limit: int = 256
    ) -> list[StoredTask]:
        if limit <= 0:
            return []
        async with self._lock:
            values = [
                self._stored(task_id)
                for task_id, (task, _revision) in sorted(
                    self._tasks.items(),
                    key=lambda item: (
                        item[1][0].expires_at_ms,
                        item[1][0].created_seq,
                        item[0],
                    ),
                )
                if task.expires_at_ms <= before_ms
            ]
            return [value for value in values if value is not None][:limit]

    async def reserve(
        self,
        stored: StoredTask,
        lease: WorkerLease,
        *,
        deadline_at_ms: int,
    ) -> StoredTask | None:
        async with self._lock:
            current = self._tasks.get(stored.task.id)
            if (
                current is None
                or current[1] != stored.revision
                or current[0].status != TaskStatus.QUEUED
            ):
                return None
            lease_key = (lease.pool_id, lease.worker_key)
            existing_lease = self._leases.get(lease_key)
            if existing_lease is not None:
                if (
                    existing_lease.reuse_after_ms is None
                    or existing_lease.reuse_after_ms > now_ms()
                ):
                    return None
                del self._leases[lease_key]
            queue = self._queue.setdefault(stored.task.pool_id, [])
            if stored.task.id not in queue:
                return None
            queue.remove(stored.task.id)
            updated = _apply_patch(
                current[0],
                {
                    "status": TaskStatus.DISPATCHING,
                    "attempt": current[0].attempt + 1,
                    "worker_instance_id": lease.worker_instance_id,
                    "worker_key": lease.worker_key,
                    "owner_generation": lease.owner_generation,
                    "execution_token": lease.execution_token,
                    "assigned_at_ms": now_ms(),
                    "deadline_at_ms": deadline_at_ms,
                },
            )
            revision = self._next_revision()
            self._tasks[stored.task.id] = (updated, revision)
            self._publish_task_event(stored.task.id, revision)
            self._leases[lease_key] = copy.deepcopy(lease)
            return StoredTask(_clone_task(updated), revision)

    async def transition(
        self,
        task_id: str,
        *,
        expected: Iterable[TaskStatus],
        patch: Mapping[str, Any],
        expected_revision: int | None = None,
        release_lease: bool = False,
        quarantine_until_ms: int | None = None,
    ) -> StoredTask:
        expected_set = set(expected)
        async with self._lock:
            current = self._tasks.get(task_id)
            if current is None:
                raise KeyError(task_id)
            if expected_revision is not None and current[1] != expected_revision:
                raise StoreConflict("task revision changed")
            if current[0].status not in expected_set:
                raise StoreConflict(
                    f"task status is {current[0].status.value}, expected "
                    + ", ".join(sorted(status.value for status in expected_set))
                )
            updated = _apply_patch(current[0], patch)
            if (
                current[0].status == TaskStatus.QUEUED
                and updated.status != TaskStatus.QUEUED
            ):
                queue = self._queue.setdefault(current[0].pool_id, [])
                if task_id in queue:
                    queue.remove(task_id)
            if release_lease and current[0].worker_key:
                key = (current[0].pool_id, current[0].worker_key)
                lease = self._leases.get(key)
                owns_lease = lease is not None and lease.task_id == task_id
                if owns_lease and quarantine_until_ms is not None:
                    assert lease is not None
                    lease.state = "quarantined"
                    lease.reuse_after_ms = quarantine_until_ms
                    lease.heartbeat_at_ms = now_ms()
                elif owns_lease:
                    self._leases.pop(key, None)
                elif lease is None and quarantine_until_ms is not None:
                    if current[0].worker_instance_id is None:
                        raise StoreConflict(
                            "cannot quarantine a task without a Worker instance"
                        )
                    self._leases[key] = WorkerLease(
                        pool_id=current[0].pool_id,
                        worker_key=current[0].worker_key,
                        worker_instance_id=current[0].worker_instance_id,
                        backend_target=current[0].backend_target,
                        task_id=current[0].id,
                        owner_generation=current[0].owner_generation or "unknown",
                        execution_token=current[0].execution_token,
                        state="quarantined",
                        heartbeat_at_ms=now_ms(),
                        reuse_after_ms=quarantine_until_ms,
                    )
            revision = self._next_revision()
            self._tasks[task_id] = (updated, revision)
            self._publish_task_event(task_id, revision)
            return StoredTask(_clone_task(updated), revision)

    async def request_cancel(
        self, task_id: str, *, terminal_expires_at_ms: int | None = None
    ) -> StoredTask:
        for _ in range(8):
            stored = await self.get_task(task_id)
            if stored is None:
                raise KeyError(task_id)
            if stored.task.status == TaskStatus.QUEUED:
                patch = {
                    "status": TaskStatus.CANCELLED,
                    "cancel_requested_at_ms": now_ms(),
                    "completed_at_ms": now_ms(),
                    "expires_at_ms": terminal_expires_at_ms
                    or now_ms() + 60 * 60 * 1000,
                    "error": TaskError(
                        code="cancelled", message="video task was cancelled"
                    ),
                }
                expected = {TaskStatus.QUEUED}
            elif stored.task.status in ACTIVE_STATUSES:
                patch = {"cancel_requested_at_ms": now_ms()}
                expected = ACTIVE_STATUSES
            else:
                return stored
            try:
                return await self.transition(
                    task_id,
                    expected=expected,
                    expected_revision=stored.revision,
                    patch=patch,
                )
            except StoreConflict:
                continue
        raise StoreConflict("cancel lost repeated in-memory CAS races")

    async def list_leases(self, pool_id: str) -> list[WorkerLease]:
        async with self._lock:
            return [
                copy.deepcopy(lease)
                for (lease_pool, _), lease in self._leases.items()
                if lease_pool == pool_id
            ]

    async def release_lease(self, pool_id: str, worker_key_value: str) -> None:
        async with self._lock:
            self._leases.pop((pool_id, worker_key_value), None)

    async def heartbeat_lease(
        self,
        pool_id: str,
        worker_key_value: str,
        task_id: str,
        lease_id: int | None = None,
    ) -> None:
        del lease_id
        async with self._lock:
            lease = self._leases.get((pool_id, worker_key_value))
            if lease is None or lease.task_id != task_id:
                raise StoreConflict("Worker execution lease ownership was lost")
            lease.heartbeat_at_ms = now_ms()
            lease.owner_expires_at_ms = now_ms() + 15_000

    async def delete_expired(self, stored: StoredTask) -> bool:
        async with self._lock:
            current = self._tasks.get(stored.task.id)
            if (
                current is None
                or current[1] != stored.revision
                or current[0].status != TaskStatus.EXPIRED
            ):
                return False
            del self._tasks[stored.task.id]
            for key, value in list(self._idempotency.items()):
                if value[0] == stored.task.id:
                    del self._idempotency[key]
            revision = self._next_revision()
            self._publish_task_event(stored.task.id, revision, deleted=True)
            return True

    async def reconcile_pool(self, pool_id: str) -> None:
        async with self._lock:
            queued = [
                task_id
                for task_id, (task, _revision) in sorted(
                    self._tasks.items(),
                    key=lambda item: (item[1][0].created_seq, item[0]),
                )
                if task.pool_id == pool_id and task.status == TaskStatus.QUEUED
            ]
            self._queue[pool_id] = queued


class EtcdTaskStore(TaskStore):
    """etcd-backed multi-Gateway task state with leases and optimistic CAS."""

    def __init__(
        self,
        client: EtcdHttpClient,
        *,
        prefix: str,
        deployment_id: str,
        execution_lease_ttl_s: int = 15,
    ) -> None:
        if execution_lease_ttl_s < 5:
            raise ValueError("execution lease TTL must be at least 5 seconds")
        self.client = client
        self.root = f"{prefix.rstrip('/')}/deployments/{deployment_id}"
        self.execution_lease_ttl_s = execution_lease_ttl_s

    @property
    def lease_watch_supported(self) -> bool:
        return True

    @property
    def task_watch_supported(self) -> bool:
        return True

    @property
    def gateway_owner_supported(self) -> bool:
        return True

    def _task_key(self, task_id: str) -> str:
        return f"{self.root}/tasks/{task_id}"

    def _queue_key(self, pool_id: str, task_id: str) -> str:
        return f"{self.root}/pools/{pool_id}/queue/{task_id}"

    def _queue_prefix(self, pool_id: str) -> str:
        return f"{self.root}/pools/{pool_id}/queue/"

    def _ordered_queue_key(self, task: VideoTask) -> str:
        return (
            f"{self.root}/indexes/queue/{task.pool_id}/"
            f"{self._sequence_token(task.created_seq)}/{task.id}"
        )

    def _ordered_queue_prefix(self, pool_id: str) -> str:
        return f"{self.root}/indexes/queue/{pool_id}/"

    def _counter_key(self, pool_id: str) -> str:
        return f"{self.root}/pools/{pool_id}/counters/queued"

    def _lease_key(self, pool_id: str, worker_key_value: str) -> str:
        return f"{self.root}/pools/{pool_id}/worker-leases/{worker_key_value}"

    def _lease_prefix(self, pool_id: str) -> str:
        return f"{self.root}/pools/{pool_id}/worker-leases/"

    def _lease_heartbeat_key(self, pool_id: str, worker_key_value: str) -> str:
        return f"{self.root}/pools/{pool_id}/worker-heartbeats/{worker_key_value}"

    def _gateway_key(self, gateway_id: str) -> str:
        return f"{self.root}/gateways/{gateway_id}"

    def _owner_task_key(self, gateway_id: str, task_id: str) -> str:
        return f"{self.root}/gateway-owners/{gateway_id}/tasks/{task_id}"

    def _owner_task_prefix(self) -> str:
        return f"{self.root}/gateway-owners/"

    def _idempotency_key(self, principal_hash: str, key_hash: str) -> str:
        return f"{self.root}/idempotency/{principal_hash}/{key_hash}"

    def _meta_key(self, name: str) -> str:
        return f"{self.root}/meta/{name}"

    def _sequence_key(self) -> str:
        return self._meta_key("task-sequence")

    def _index_schema_key(self) -> str:
        return self._meta_key("task-index-schema")

    @staticmethod
    def _sequence_token(sequence: int) -> str:
        if sequence <= 0 or sequence >= 10**_SEQUENCE_WIDTH:
            raise ValueError("task created_seq is outside the supported range")
        return f"{sequence:0{_SEQUENCE_WIDTH}d}"

    def _task_index_prefix(
        self,
        *,
        pool_id: str | None,
        status: TaskStatus | None,
    ) -> str:
        pool = pool_id if pool_id is not None else "_all"
        status_value = status.value if status is not None else "_all"
        return f"{self.root}/indexes/tasks/{pool}/{status_value}/"

    def _task_index_key(
        self,
        task: VideoTask,
        *,
        pool_id: str | None,
        status: TaskStatus | None,
    ) -> str:
        return (
            self._task_index_prefix(pool_id=pool_id, status=status)
            + f"{self._sequence_token(task.created_seq)}/{task.id}"
        )

    def _all_task_index_keys(self, task: VideoTask) -> tuple[str, ...]:
        return (
            self._task_index_key(task, pool_id=None, status=None),
            self._task_index_key(task, pool_id=None, status=task.status),
            self._task_index_key(task, pool_id=task.pool_id, status=None),
            self._task_index_key(
                task,
                pool_id=task.pool_id,
                status=task.status,
            ),
        )

    def _status_task_index_keys(
        self, task: VideoTask, status: TaskStatus
    ) -> tuple[str, str]:
        return (
            self._task_index_key(task, pool_id=None, status=status),
            self._task_index_key(task, pool_id=task.pool_id, status=status),
        )

    def _expiry_index_key(self, task: VideoTask) -> str:
        return (
            f"{self.root}/indexes/expiry/{task.expires_at_ms:013d}/"
            f"{self._sequence_token(task.created_seq)}/{task.id}"
        )

    @staticmethod
    def _encode(value: Any) -> str:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @staticmethod
    def _task(value: EtcdValue) -> StoredTask:
        return StoredTask(
            VideoTask.from_dict(json.loads(value.value)), value.mod_revision
        )

    async def health(self) -> None:
        await self.client.range(self.root, prefix=True, limit=1, keys_only=True)

    async def close(self) -> None:
        await self.client.close()

    async def prepare(self) -> None:
        """Backfill v2 task indexes before this Gateway becomes ready."""

        marker_key = self._index_schema_key()
        marker = await self.client.get(marker_key)
        if marker is not None:
            try:
                marker_version = int(marker.value.decode())
            except (UnicodeDecodeError, ValueError) as exc:
                raise RuntimeError("invalid task index schema marker") from exc
            if marker_version == _INDEX_SCHEMA_VERSION:
                return
            if not 1 <= marker_version < _INDEX_SCHEMA_VERSION:
                raise RuntimeError("unsupported task index schema marker")

        # The Gateway calls prepare before it accepts traffic. Multiple new
        # Gateways may run this deterministic migration concurrently; every
        # task write is protected by its own mod revision and is idempotent.
        for _ in range(8):
            values, _snapshot_revision = await self.client.range_all(
                f"{self.root}/tasks/",
                prefix=True,
            )
            tasks = [self._task(value) for value in values]
            tasks.sort(key=lambda item: (item.task.created_at_ms, item.task.id))
            sequence, sequence_value = await self._sequence()
            next_sequence = max(
                [sequence, *(item.task.created_seq for item in tasks)]
            )
            conflicted = False
            for stored in tasks:
                task = stored.task
                task_changed = False
                if task.created_seq <= 0:
                    next_sequence += 1
                    task = _apply_patch(task, {"created_seq": next_sequence})
                    task_changed = True
                success = [
                    *[
                        self.client.put(key, task.id)
                        for key in self._all_task_index_keys(task)
                    ],
                    self.client.put(self._expiry_index_key(task), task.id),
                ]
                if task_changed:
                    success.insert(
                        0,
                        self.client.put(
                            self._task_key(task.id), self._encode(task.to_dict())
                        ),
                    )
                if task.status == TaskStatus.QUEUED:
                    success.append(
                        self.client.put(self._ordered_queue_key(task), task.id)
                    )
                if task.status in ACTIVE_STATUSES and task.owner_generation:
                    success.append(
                        self.client.put(
                            self._owner_task_key(task.owner_generation, task.id),
                            task.id,
                        )
                    )
                succeeded, _revision = await self.client.txn(
                    [
                        self.client.compare_mod(
                            self._task_key(task.id), stored.revision
                        )
                    ],
                    success,
                )
                if not succeeded:
                    conflicted = True
                    break
            if conflicted:
                continue

            current_sequence, current_sequence_value = await self._sequence()
            target_sequence = max(current_sequence, next_sequence)
            compare = self._counter_compare(
                self._sequence_key(), current_sequence_value
            )
            success = [
                self.client.put(self._sequence_key(), str(target_sequence))
            ]
            if (
                current_sequence == target_sequence
                and current_sequence_value is not None
            ):
                success = []
            if success:
                succeeded, _revision = await self.client.txn(compare, success)
                if not succeeded:
                    continue

            marker_compare = (
                [self.client.compare_version(marker_key, 0)]
                if marker is None
                else [self.client.compare_mod(marker_key, marker.mod_revision)]
            )
            succeeded, _revision = await self.client.txn(
                marker_compare,
                [self.client.put(marker_key, str(_INDEX_SCHEMA_VERSION))],
            )
            if succeeded:
                return
            marker = await self.client.get(marker_key)
            if marker is not None and marker.value == str(
                _INDEX_SCHEMA_VERSION
            ).encode():
                return
        raise StoreConflict("unable to prepare task indexes after repeated races")

    async def _counter(self, pool_id: str) -> tuple[int, EtcdValue | None]:
        value = await self.client.get(self._counter_key(pool_id))
        if value is None:
            return 0, None
        try:
            count = int(value.value.decode())
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError(f"invalid queue counter for pool {pool_id}") from exc
        if count < 0:
            raise RuntimeError(f"negative queue counter for pool {pool_id}")
        return count, value

    async def _sequence(self) -> tuple[int, EtcdValue | None]:
        key = self._sequence_key()
        value = await self.client.get(key)
        if value is None:
            return 0, None
        try:
            sequence = int(value.value.decode())
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("invalid task sequence counter") from exc
        if sequence < 0 or sequence >= 10**_SEQUENCE_WIDTH - 1:
            raise RuntimeError("task sequence counter is outside the supported range")
        return sequence, value

    def _counter_compare(self, key: str, value: EtcdValue | None) -> list[dict]:
        if value is None:
            return [self.client.compare_version(key, 0)]
        return [
            self.client.compare_mod(key, value.mod_revision),
            self.client.compare_value(key, value.value),
        ]

    async def create_task(
        self,
        task: VideoTask,
        *,
        principal_hash: str,
        idempotency_hash: str | None,
        queue_limit: int,
    ) -> tuple[StoredTask, bool]:
        task = _apply_patch(
            task,
            {
                "principal_hash": principal_hash,
                "idempotency_hash": idempotency_hash,
            },
        )
        task_key = self._task_key(task.id)
        queue_key = self._queue_key(task.pool_id, task.id)
        counter_key = self._counter_key(task.pool_id)
        idem_key = (
            self._idempotency_key(principal_hash, idempotency_hash)
            if idempotency_hash is not None
            else None
        )
        for _ in range(8):
            if idem_key is not None:
                existing = await self.client.get(idem_key)
                if existing is not None:
                    index = json.loads(existing.value)
                    if index["request_digest"] != task.request_digest:
                        raise GatewayError(
                            409,
                            "idempotency_conflict",
                            "Idempotency-Key was already used with another request",
                        )
                    stored = await self.get_task(index["task_id"])
                    if stored is None:
                        raise RuntimeError("idempotency index points to a missing task")
                    return stored, False

            count, counter = await self._counter(task.pool_id)
            sequence, sequence_value = await self._sequence()
            if count >= queue_limit:
                raise GatewayError(429, "queue_full", "video queue is full")
            assigned = _apply_patch(task, {"created_seq": sequence + 1})
            compare = [
                self.client.compare_version(task_key, 0),
                self.client.compare_version(queue_key, 0),
                *self._counter_compare(counter_key, counter),
                *self._counter_compare(self._sequence_key(), sequence_value),
            ]
            success = [
                self.client.put(task_key, self._encode(assigned.to_dict())),
                self.client.put(queue_key, str(assigned.queued_at_ms)),
                self.client.put(counter_key, str(count + 1)),
                self.client.put(self._sequence_key(), str(assigned.created_seq)),
                self.client.put(self._ordered_queue_key(assigned), assigned.id),
                *[
                    self.client.put(key, assigned.id)
                    for key in self._all_task_index_keys(assigned)
                ],
                self.client.put(self._expiry_index_key(assigned), assigned.id),
            ]
            if idem_key is not None:
                compare.append(self.client.compare_version(idem_key, 0))
                success.append(
                    self.client.put(
                        idem_key,
                        self._encode(
                            {"task_id": task.id, "request_digest": task.request_digest}
                        ),
                    )
                )
            succeeded, revision = await self.client.txn(compare, success)
            if succeeded:
                return StoredTask(assigned, revision), True
        raise StoreConflict("unable to create task after repeated etcd CAS conflicts")

    async def get_task(self, task_id: str) -> StoredTask | None:
        value = await self.client.get(self._task_key(task_id))
        return self._task(value) if value is not None else None

    async def get_idempotent(
        self, principal_hash: str, idempotency_hash: str
    ) -> StoredTask | None:
        value = await self.client.get(
            self._idempotency_key(principal_hash, idempotency_hash)
        )
        if value is None:
            return None
        try:
            task_id = json.loads(value.value)["task_id"]
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("invalid idempotency index record") from exc
        stored = await self.get_task(task_id)
        if stored is None:
            raise RuntimeError("idempotency index points to a missing task")
        return stored

    async def _list_indexed(
        self,
        prefix: str,
        *,
        after_task: VideoTask | None,
        limit: int,
        descending: bool,
        pool_id: str | None = None,
        status: TaskStatus | None = None,
    ) -> list[StoredTask]:
        if limit <= 0:
            return []
        if limit > 10_000:
            raise ValueError("task list limit must not exceed 10000")
        range_end = self.client.prefix_end(prefix)
        cursor = prefix.encode()
        if after_task is not None:
            after_key = (
                prefix
                + f"{self._sequence_token(after_task.created_seq)}/"
                + after_task.id
            ).encode()
            if descending:
                range_end = after_key
            else:
                cursor = after_key + b"\0"

        snapshot_revision = 0
        result: list[StoredTask] = []
        while len(result) < limit and cursor < range_end:
            page = await self.client.range_page(
                cursor,
                range_end=range_end,
                limit=min(128, limit - len(result)),
                keys_only=False,
                descending=descending,
                revision=snapshot_revision,
            )
            if snapshot_revision == 0:
                snapshot_revision = page.revision
                if snapshot_revision <= 0:
                    raise RuntimeError("task index response omitted its revision")
            if not page.values:
                break
            task_ids = [value.value.decode() for value in page.values]
            values, _revision = await self.client.get_many(
                [self._task_key(task_id) for task_id in task_ids],
                revision=snapshot_revision,
            )
            for value in values:
                if value is None:
                    raise RuntimeError("task index points to a missing task")
                stored = self._task(value)
                if pool_id is not None and stored.task.pool_id != pool_id:
                    raise RuntimeError("task pool index is inconsistent")
                if status is not None and stored.task.status != status:
                    raise RuntimeError("task status index is inconsistent")
                result.append(stored)
            last_key = page.values[-1].key.encode()
            if descending:
                if last_key >= range_end:
                    raise RuntimeError("descending task index cursor did not advance")
                range_end = last_key
            else:
                next_cursor = last_key + b"\0"
                if next_cursor <= cursor:
                    raise RuntimeError("ascending task index cursor did not advance")
                cursor = next_cursor
            if not page.more:
                break
        return result[:limit]

    async def list_tasks(
        self,
        *,
        pool_id: str | None = None,
        status: TaskStatus | None = None,
        after: str | None = None,
        limit: int = 100,
        descending: bool = False,
    ) -> list[StoredTask]:
        after_task: VideoTask | None = None
        if after is not None:
            stored = await self.get_task(after)
            if stored is None:
                return []
            after_task = stored.task
        prefix = self._task_index_prefix(pool_id=pool_id, status=status)
        return await self._list_indexed(
            prefix,
            after_task=after_task,
            limit=limit,
            descending=descending,
            pool_id=pool_id,
            status=status,
        )

    async def list_all_tasks(self) -> list[StoredTask]:
        values, _revision = await self.client.range_all(
            f"{self.root}/tasks/",
            prefix=True,
        )
        return [self._task(value) for value in values]

    async def list_queued(self, pool_id: str, *, limit: int = 100) -> list[StoredTask]:
        return await self._list_indexed(
            self._ordered_queue_prefix(pool_id),
            after_task=None,
            limit=limit,
            descending=False,
            pool_id=pool_id,
            status=TaskStatus.QUEUED,
        )

    async def queue_depth(self, pool_id: str) -> int:
        count, _ = await self._counter(pool_id)
        return count

    async def task_counts(self, pool_id: str) -> Mapping[TaskStatus, int]:
        values = await asyncio.gather(
            *(
                self.client.count_prefix(
                    self._task_index_prefix(pool_id=pool_id, status=status)
                )
                for status in TaskStatus
            )
        )
        return dict(zip(TaskStatus, values, strict=True))

    async def list_due_tasks(
        self, before_ms: int, *, limit: int = 256
    ) -> list[StoredTask]:
        if limit <= 0:
            return []
        if limit > 10_000:
            raise ValueError("due task list limit must not exceed 10000")
        prefix = f"{self.root}/indexes/expiry/"
        range_end = (prefix + f"{before_ms + 1:013d}/").encode()
        page = await self.client.range_page(
            prefix,
            range_end=range_end,
            limit=limit,
        )
        if not page.values:
            return []
        task_ids = [value.value.decode() for value in page.values]
        values, _revision = await self.client.get_many(
            [self._task_key(task_id) for task_id in task_ids],
            revision=page.revision,
        )
        result: list[StoredTask] = []
        for value in values:
            if value is None:
                raise RuntimeError("expiry index points to a missing task")
            stored = self._task(value)
            if stored.task.expires_at_ms > before_ms:
                raise RuntimeError("expiry index is inconsistent")
            result.append(stored)
        return result

    async def reserve(
        self,
        stored: StoredTask,
        lease: WorkerLease,
        *,
        deadline_at_ms: int,
    ) -> StoredTask | None:
        task_key = self._task_key(stored.task.id)
        queue_key = self._queue_key(stored.task.pool_id, stored.task.id)
        counter_key = self._counter_key(stored.task.pool_id)
        lease_key = self._lease_key(lease.pool_id, lease.worker_key)
        heartbeat_key = self._lease_heartbeat_key(lease.pool_id, lease.worker_key)
        count, counter = await self._counter(stored.task.pool_id)
        if count <= 0:
            return None
        native_lease = await self.client.lease_grant(self.execution_lease_ttl_s)
        lease = copy.deepcopy(lease)
        lease.etcd_lease_id = native_lease.lease_id
        updated = _apply_patch(
            stored.task,
            {
                "status": TaskStatus.DISPATCHING,
                "attempt": stored.task.attempt + 1,
                "worker_instance_id": lease.worker_instance_id,
                "worker_key": lease.worker_key,
                "worker_lease_id": native_lease.lease_id,
                "owner_generation": lease.owner_generation,
                "execution_token": lease.execution_token,
                "assigned_at_ms": now_ms(),
                "deadline_at_ms": deadline_at_ms,
            },
        )
        compare = [
            self.client.compare_mod(task_key, stored.revision),
            self.client.compare_version(queue_key, 0, result="GREATER"),
            self.client.compare_version(lease_key, 0),
            self.client.compare_version(heartbeat_key, 0),
            self.client.compare_version(
                self._gateway_key(lease.owner_generation), 0, result="GREATER"
            ),
            *self._counter_compare(counter_key, counter),
        ]
        success = [
            self.client.put(task_key, self._encode(updated.to_dict())),
            self.client.put(
                lease_key,
                self._encode(lease.to_dict()),
            ),
            self.client.put(
                heartbeat_key,
                self._encode(
                    {
                        "task_id": updated.id,
                        "lease_id": native_lease.lease_id,
                    }
                ),
                lease_id=native_lease.lease_id,
            ),
            self.client.delete(queue_key),
            self.client.delete(self._ordered_queue_key(stored.task)),
            self.client.put(counter_key, str(count - 1)),
            self.client.put(
                self._owner_task_key(lease.owner_generation, updated.id), updated.id
            ),
            *[
                self.client.delete(key)
                for key in self._status_task_index_keys(
                    stored.task, stored.task.status
                )
            ],
            *[
                self.client.put(key, updated.id)
                for key in self._status_task_index_keys(updated, updated.status)
            ],
        ]
        try:
            succeeded, revision = await self.client.txn(compare, success)
        except Exception:
            try:
                await self.client.lease_revoke(native_lease.lease_id)
            except Exception:
                pass
            raise
        if not succeeded:
            try:
                await self.client.lease_revoke(native_lease.lease_id)
            except Exception:
                pass
        return StoredTask(updated, revision) if succeeded else None

    async def transition(
        self,
        task_id: str,
        *,
        expected: Iterable[TaskStatus],
        patch: Mapping[str, Any],
        expected_revision: int | None = None,
        release_lease: bool = False,
        quarantine_until_ms: int | None = None,
    ) -> StoredTask:
        stored = await self.get_task(task_id)
        if stored is None:
            raise KeyError(task_id)
        expected_set = set(expected)
        if stored.task.status not in expected_set:
            raise StoreConflict(
                f"task status is {stored.task.status.value}, expected "
                + ", ".join(sorted(status.value for status in expected_set))
            )
        if expected_revision is not None and stored.revision != expected_revision:
            raise StoreConflict("task revision changed")
        updated = _apply_patch(stored.task, patch)
        task_key = self._task_key(task_id)
        compare: list[dict] = [self.client.compare_mod(task_key, stored.revision)]
        success: list[dict] = [
            self.client.put(task_key, self._encode(updated.to_dict()))
        ]

        if stored.task.status != updated.status:
            success.extend(
                self.client.delete(key)
                for key in self._status_task_index_keys(
                    stored.task, stored.task.status
                )
            )
            success.extend(
                self.client.put(key, updated.id)
                for key in self._status_task_index_keys(updated, updated.status)
            )
        if (
            stored.task.owner_generation
            and updated.status in TERMINAL_STATUSES
            and stored.task.status not in TERMINAL_STATUSES
        ):
            success.append(
                self.client.delete(
                    self._owner_task_key(stored.task.owner_generation, task_id)
                )
            )
        if stored.task.expires_at_ms != updated.expires_at_ms:
            success.extend(
                [
                    self.client.delete(self._expiry_index_key(stored.task)),
                    self.client.put(self._expiry_index_key(updated), updated.id),
                ]
            )

        if (
            stored.task.status == TaskStatus.QUEUED
            and updated.status != TaskStatus.QUEUED
        ):
            queue_key = self._queue_key(stored.task.pool_id, task_id)
            counter_key = self._counter_key(stored.task.pool_id)
            count, counter = await self._counter(stored.task.pool_id)
            if count <= 0:
                raise RuntimeError("queue counter is inconsistent")
            compare.extend(
                [
                    self.client.compare_version(queue_key, 0, result="GREATER"),
                    *self._counter_compare(counter_key, counter),
                ]
            )
            success.extend(
                [
                    self.client.delete(queue_key),
                    self.client.delete(self._ordered_queue_key(stored.task)),
                    self.client.put(counter_key, str(count - 1)),
                ]
            )

        if release_lease and stored.task.worker_key is not None:
            lease_key = self._lease_key(stored.task.pool_id, stored.task.worker_key)
            heartbeat_key = self._lease_heartbeat_key(
                stored.task.pool_id, stored.task.worker_key
            )
            lease_value = await self.client.get(lease_key)
            if lease_value is not None:
                lease = WorkerLease.from_dict(json.loads(lease_value.value))
                if lease.task_id == task_id:
                    compare.append(
                        self.client.compare_mod(lease_key, lease_value.mod_revision)
                    )
                if lease.task_id == task_id and quarantine_until_ms is None:
                    success.extend(
                        [
                            self.client.delete(lease_key),
                            self.client.delete(heartbeat_key),
                        ]
                    )
                elif lease.task_id == task_id:
                    lease.state = "quarantined"
                    lease.reuse_after_ms = quarantine_until_ms
                    lease.heartbeat_at_ms = now_ms()
                    success.append(
                        self.client.put(lease_key, self._encode(lease.to_dict()))
                    )
                    success.append(self.client.delete(heartbeat_key))
            elif quarantine_until_ms is not None:
                if stored.task.worker_instance_id is None:
                    raise StoreConflict(
                        "cannot quarantine a task without a Worker instance"
                    )
                quarantine = WorkerLease(
                    pool_id=stored.task.pool_id,
                    worker_key=stored.task.worker_key,
                    worker_instance_id=stored.task.worker_instance_id,
                    backend_target=stored.task.backend_target,
                    task_id=stored.task.id,
                    owner_generation=stored.task.owner_generation or "unknown",
                    execution_token=stored.task.execution_token,
                    state="quarantined",
                    heartbeat_at_ms=now_ms(),
                    reuse_after_ms=quarantine_until_ms,
                )
                compare.append(self.client.compare_version(lease_key, 0))
                success.extend(
                    [
                        self.client.put(
                            lease_key, self._encode(quarantine.to_dict())
                        ),
                        self.client.delete(heartbeat_key),
                    ]
                )

        succeeded, revision = await self.client.txn(compare, success)
        if not succeeded:
            raise StoreConflict("task transition lost an etcd CAS race")
        return StoredTask(updated, revision)

    async def request_cancel(
        self, task_id: str, *, terminal_expires_at_ms: int | None = None
    ) -> StoredTask:
        for _ in range(8):
            stored = await self.get_task(task_id)
            if stored is None:
                raise KeyError(task_id)
            if stored.task.status == TaskStatus.QUEUED:
                patch = {
                    "status": TaskStatus.CANCELLED,
                    "cancel_requested_at_ms": now_ms(),
                    "completed_at_ms": now_ms(),
                    "expires_at_ms": terminal_expires_at_ms
                    or now_ms() + 60 * 60 * 1000,
                    "error": TaskError(
                        code="cancelled", message="video task was cancelled"
                    ),
                }
                expected: Iterable[TaskStatus] = {TaskStatus.QUEUED}
            elif stored.task.status in ACTIVE_STATUSES:
                patch = {"cancel_requested_at_ms": now_ms()}
                expected = ACTIVE_STATUSES
            else:
                return stored
            try:
                return await self.transition(
                    task_id,
                    expected=expected,
                    expected_revision=stored.revision,
                    patch=patch,
                )
            except StoreConflict:
                continue
        raise StoreConflict("cancel lost repeated etcd CAS races")

    async def list_leases(self, pool_id: str) -> list[WorkerLease]:
        values = await self.client.range(self._lease_prefix(pool_id), prefix=True)
        return [WorkerLease.from_dict(json.loads(value.value)) for value in values]

    async def lease_snapshot(
        self, pool_id: str
    ) -> tuple[dict[str, WorkerLease], int]:
        values, revision = await self.client.range_all(
            self._lease_prefix(pool_id),
            prefix=True,
        )
        leases = [WorkerLease.from_dict(json.loads(value.value)) for value in values]
        return {lease.worker_key: lease for lease in leases}, revision

    async def watch_leases(
        self, pool_id: str, *, start_revision: int
    ) -> AsyncIterator[LeaseWatchEvent]:
        prefix = self._lease_prefix(pool_id)
        async for response in self.client.watch_prefix(
            prefix,
            start_revision=start_revision,
            progress_notify=True,
        ):
            if not response.events:
                yield LeaseWatchEvent(
                    revision=response.revision,
                    worker_key=None,
                    lease=None,
                    created=response.created,
                )
                continue
            for event in response.events:
                key = event.value.key.rsplit("/", 1)[-1]
                lease = None
                if event.event_type != "DELETE":
                    lease = WorkerLease.from_dict(json.loads(event.value.value))
                    if lease.pool_id != pool_id or lease.worker_key != key:
                        raise RuntimeError("Worker lease watch index is inconsistent")
                yield LeaseWatchEvent(
                    revision=max(response.revision, event.value.mod_revision),
                    worker_key=key,
                    lease=lease,
                )

    async def task_watch_revision(self) -> int:
        page = await self.client.range_page(
            f"{self.root}/tasks/",
            prefix=True,
            limit=1,
            keys_only=True,
        )
        if page.revision <= 0:
            raise RuntimeError("task watch snapshot omitted its etcd revision")
        return page.revision

    async def watch_tasks(
        self, *, start_revision: int
    ) -> AsyncIterator[TaskWatchEvent]:
        prefix = f"{self.root}/tasks/"
        async for response in self.client.watch_prefix(
            prefix,
            start_revision=start_revision,
            progress_notify=True,
        ):
            if not response.events:
                yield TaskWatchEvent(
                    revision=response.revision,
                    task_id=None,
                    created=response.created,
                )
                continue
            for event in response.events:
                yield TaskWatchEvent(
                    revision=max(response.revision, event.value.mod_revision),
                    task_id=event.value.key.rsplit("/", 1)[-1],
                    deleted=event.event_type == "DELETE",
                )

    async def register_gateway(self, gateway_id: str, *, ttl_s: int) -> int:
        native_lease = await self.client.lease_grant(ttl_s)
        key = self._gateway_key(gateway_id)
        succeeded, _revision = await self.client.txn(
            [self.client.compare_version(key, 0)],
            [
                self.client.put(
                    key,
                    self._encode(
                        {
                            "gateway_id": gateway_id,
                            "lease_id": native_lease.lease_id,
                            "registered_at_ms": now_ms(),
                        }
                    ),
                    lease_id=native_lease.lease_id,
                )
            ],
        )
        if not succeeded:
            await self.client.lease_revoke(native_lease.lease_id)
            raise StoreConflict(f"Gateway owner already exists: {gateway_id}")
        return native_lease.lease_id

    async def keepalive_gateway(self, lease_id: int) -> None:
        await self.client.lease_keepalive(lease_id)

    async def unregister_gateway(self, lease_id: int) -> None:
        await self.client.lease_revoke(lease_id)

    async def iter_orphaned_active_tasks(self) -> AsyncIterator[StoredTask]:
        prefix = self._owner_task_prefix()
        cursor = prefix.encode()
        range_end = self.client.prefix_end(prefix)
        snapshot_revision = 0
        while cursor < range_end:
            page = await self.client.range_page(
                cursor,
                range_end=range_end,
                limit=128,
                revision=snapshot_revision,
            )
            if snapshot_revision == 0:
                snapshot_revision = page.revision
            if not page.values:
                break
            gateway_ids: list[str] = []
            task_ids: list[str] = []
            for value in page.values:
                relative = value.key.removeprefix(prefix)
                parts = relative.split("/")
                if len(parts) != 3 or parts[1] != "tasks":
                    raise RuntimeError("invalid Gateway owner task index key")
                gateway_ids.append(parts[0])
                task_ids.append(parts[2])
            gateway_values, _revision = await self.client.get_many(
                [self._gateway_key(gateway_id) for gateway_id in gateway_ids],
                revision=snapshot_revision,
            )
            task_values, _revision = await self.client.get_many(
                [self._task_key(task_id) for task_id in task_ids],
                revision=snapshot_revision,
            )
            for gateway_id, task_id, gateway_value, task_value in zip(
                gateway_ids,
                task_ids,
                gateway_values,
                task_values,
                strict=True,
            ):
                if gateway_value is not None:
                    continue
                if task_value is None:
                    continue
                stored = self._task(task_value)
                if (
                    stored.task.status in ACTIVE_STATUSES
                    and stored.task.owner_generation == gateway_id
                    and stored.task.id == task_id
                ):
                    yield stored
            last_key = page.values[-1].key.encode()
            next_cursor = last_key + b"\0"
            if next_cursor <= cursor:
                raise RuntimeError("Gateway owner index cursor did not advance")
            cursor = next_cursor
            if not page.more:
                break

    async def claim_orphaned_active(
        self, stored: StoredTask, *, new_owner_generation: str
    ) -> StoredTask | None:
        task = stored.task
        if (
            task.status not in ACTIVE_STATUSES
            or not task.owner_generation
            or not task.worker_key
            or task.worker_instance_id is None
            or not task.execution_token
        ):
            return None
        native_lease = await self.client.lease_grant(self.execution_lease_ttl_s)
        lease = WorkerLease(
            pool_id=task.pool_id,
            worker_key=task.worker_key,
            worker_instance_id=task.worker_instance_id,
            backend_target=task.backend_target,
            task_id=task.id,
            owner_generation=new_owner_generation,
            execution_token=task.execution_token,
            state="running",
            heartbeat_at_ms=now_ms(),
            owner_expires_at_ms=now_ms() + self.execution_lease_ttl_s * 1000,
            etcd_lease_id=native_lease.lease_id,
        )
        updated = _apply_patch(
            task,
            {
                "owner_generation": new_owner_generation,
                "worker_lease_id": native_lease.lease_id,
            },
        )
        task_key = self._task_key(task.id)
        lease_key = self._lease_key(task.pool_id, task.worker_key)
        heartbeat_key = self._lease_heartbeat_key(task.pool_id, task.worker_key)
        existing_lease_value = await self.client.get(lease_key)
        if existing_lease_value is None:
            lease_compare = self.client.compare_version(lease_key, 0)
        else:
            existing_lease = WorkerLease.from_dict(
                json.loads(existing_lease_value.value)
            )
            if (
                existing_lease.task_id != task.id
                or existing_lease.owner_generation != task.owner_generation
            ):
                try:
                    await self.client.lease_revoke(native_lease.lease_id)
                except Exception:
                    pass
                return None
            lease_compare = self.client.compare_mod(
                lease_key, existing_lease_value.mod_revision
            )
        compare = [
            self.client.compare_mod(task_key, stored.revision),
            self.client.compare_version(
                self._gateway_key(task.owner_generation), 0
            ),
            self.client.compare_version(
                self._gateway_key(new_owner_generation), 0, result="GREATER"
            ),
            lease_compare,
            self.client.compare_version(heartbeat_key, 0),
        ]
        success = [
            self.client.put(task_key, self._encode(updated.to_dict())),
            self.client.put(
                lease_key,
                self._encode(lease.to_dict()),
            ),
            self.client.put(
                heartbeat_key,
                self._encode(
                    {
                        "task_id": task.id,
                        "lease_id": native_lease.lease_id,
                    }
                ),
                lease_id=native_lease.lease_id,
            ),
            self.client.delete(
                self._owner_task_key(task.owner_generation, task.id)
            ),
            self.client.put(
                self._owner_task_key(new_owner_generation, task.id), task.id
            ),
        ]
        try:
            succeeded, revision = await self.client.txn(compare, success)
        except Exception:
            try:
                await self.client.lease_revoke(native_lease.lease_id)
            except Exception:
                pass
            raise
        if not succeeded:
            try:
                await self.client.lease_revoke(native_lease.lease_id)
            except Exception:
                pass
            return None
        return StoredTask(updated, revision)

    async def release_lease(self, pool_id: str, worker_key_value: str) -> None:
        key = self._lease_key(pool_id, worker_key_value)
        heartbeat_key = self._lease_heartbeat_key(pool_id, worker_key_value)
        value = await self.client.get(key)
        if value is None:
            return
        await self.client.txn(
            [self.client.compare_mod(key, value.mod_revision)],
            [self.client.delete(key), self.client.delete(heartbeat_key)],
        )

    async def heartbeat_lease(
        self,
        pool_id: str,
        worker_key_value: str,
        task_id: str,
        lease_id: int | None = None,
    ) -> None:
        key = self._lease_key(pool_id, worker_key_value)
        value = await self.client.get(key)
        if value is None:
            raise StoreConflict("Worker execution lease no longer exists")
        lease = WorkerLease.from_dict(json.loads(value.value))
        if lease.task_id != task_id:
            raise StoreConflict("Worker execution lease belongs to another task")
        if lease_id is not None:
            if lease.etcd_lease_id != lease_id:
                raise StoreConflict("Worker execution lease identity changed")
            # Rolling upgrades may still encounter the legacy representation,
            # where the logical Worker guard itself is attached to the native
            # lease. Keep it alive until that task reaches a terminal state.
            if value.lease == lease_id:
                await self.client.lease_keepalive(lease_id)
                return
            if value.lease != 0:
                raise StoreConflict("Worker execution guard has an invalid lease")
            heartbeat_value = await self.client.get(
                self._lease_heartbeat_key(pool_id, worker_key_value)
            )
            if heartbeat_value is None or heartbeat_value.lease != lease_id:
                raise StoreConflict("Worker execution heartbeat no longer exists")
            try:
                heartbeat = json.loads(heartbeat_value.value)
            except (TypeError, ValueError) as exc:
                raise StoreConflict("Worker execution heartbeat is invalid") from exc
            if (
                heartbeat.get("task_id") != task_id
                or heartbeat.get("lease_id") != lease_id
            ):
                raise StoreConflict("Worker execution heartbeat identity changed")
            await self.client.lease_keepalive(lease_id)
            return
        lease.heartbeat_at_ms = now_ms()
        lease.owner_expires_at_ms = now_ms() + 15_000
        succeeded, _revision = await self.client.txn(
            [self.client.compare_mod(key, value.mod_revision)],
            [self.client.put(key, self._encode(lease.to_dict()))],
        )
        if not succeeded:
            raise StoreConflict("Worker execution lease changed during heartbeat")

    async def delete_expired(self, stored: StoredTask) -> bool:
        if stored.task.status != TaskStatus.EXPIRED:
            return False
        task_key = self._task_key(stored.task.id)
        compare = [self.client.compare_mod(task_key, stored.revision)]
        success = [
            self.client.delete(task_key),
            *[
                self.client.delete(key)
                for key in self._all_task_index_keys(stored.task)
            ],
            self.client.delete(self._expiry_index_key(stored.task)),
        ]
        if stored.task.owner_generation:
            success.append(
                self.client.delete(
                    self._owner_task_key(
                        stored.task.owner_generation, stored.task.id
                    )
                )
            )
        if stored.task.principal_hash and stored.task.idempotency_hash:
            idem_key = self._idempotency_key(
                stored.task.principal_hash, stored.task.idempotency_hash
            )
            value = await self.client.get(idem_key)
            if value is not None:
                try:
                    points_to_task = (
                        json.loads(value.value).get("task_id") == stored.task.id
                    )
                except (
                    AttributeError,
                    TypeError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ):
                    points_to_task = False
                if points_to_task:
                    compare.append(
                        self.client.compare_mod(idem_key, value.mod_revision)
                    )
                    success.append(self.client.delete(idem_key))
        succeeded, _ = await self.client.txn(compare, success)
        return succeeded

    async def reconcile_pool(self, pool_id: str) -> None:
        await self.prepare()
        for _ in range(8):
            index_values, snapshot_revision = await self.client.range_all(
                self._task_index_prefix(
                    pool_id=pool_id, status=TaskStatus.QUEUED
                ),
                prefix=True,
            )
            task_ids = [value.value.decode() for value in index_values]
            task_records: list[EtcdValue | None] = []
            for offset in range(0, len(task_ids), 128):
                values, _revision = await self.client.get_many(
                    [
                        self._task_key(task_id)
                        for task_id in task_ids[offset : offset + 128]
                    ],
                    revision=snapshot_revision,
                )
                task_records.extend(values)
            expected: dict[str, StoredTask] = {}
            for value in task_records:
                if value is None:
                    raise RuntimeError("queued task index points to a missing task")
                stored = self._task(value)
                if (
                    stored.task.pool_id != pool_id
                    or stored.task.status != TaskStatus.QUEUED
                ):
                    raise RuntimeError("queued task index is inconsistent")
                expected[stored.task.id] = stored
            queue_values, _ = await self.client.range_all(
                self._queue_prefix(pool_id),
                prefix=True,
                revision=snapshot_revision,
            )
            existing = {value.key.rsplit("/", 1)[-1]: value for value in queue_values}
            ordered_values, _ = await self.client.range_all(
                self._ordered_queue_prefix(pool_id),
                prefix=True,
                revision=snapshot_revision,
            )
            ordered = {value.value.decode(): value for value in ordered_values}
            conflicted = False
            for task_id in sorted(set(existing) - set(expected)):
                task_record = await self.client.get(self._task_key(task_id))
                if task_record is not None:
                    current = self._task(task_record)
                    if (
                        current.task.pool_id == pool_id
                        and current.task.status == TaskStatus.QUEUED
                    ):
                        succeeded, _ = await self.client.txn(
                            [
                                self.client.compare_mod(
                                    task_record.key, task_record.mod_revision
                                )
                            ],
                            [
                                *[
                                    self.client.put(key, task_id)
                                    for key in self._all_task_index_keys(current.task)
                                ],
                                self.client.put(
                                    self._ordered_queue_key(current.task), task_id
                                ),
                                self.client.put(
                                    self._expiry_index_key(current.task), task_id
                                ),
                            ],
                        )
                        conflicted = True
                        if not succeeded:
                            continue
                        continue
                count, counter = await self._counter(pool_id)
                task_compare = (
                    self.client.compare_mod(task_record.key, task_record.mod_revision)
                    if task_record is not None
                    else self.client.compare_version(self._task_key(task_id), 0)
                )
                succeeded, _ = await self.client.txn(
                    [
                        self.client.compare_mod(
                            existing[task_id].key,
                            existing[task_id].mod_revision,
                        ),
                        task_compare,
                        *self._counter_compare(self._counter_key(pool_id), counter),
                    ],
                    [
                        self.client.delete(existing[task_id].key),
                        self.client.put(
                            self._counter_key(pool_id), str(max(0, count - 1))
                        ),
                    ],
                )
                conflicted = conflicted or not succeeded
            for task_id in sorted(set(expected) - set(existing)):
                count, counter = await self._counter(pool_id)
                succeeded, _ = await self.client.txn(
                    [
                        self.client.compare_mod(
                            self._task_key(task_id), expected[task_id].revision
                        ),
                        self.client.compare_version(
                            self._queue_key(pool_id, task_id), 0
                        ),
                        *self._counter_compare(self._counter_key(pool_id), counter),
                    ],
                    [
                        self.client.put(
                            self._queue_key(pool_id, task_id),
                            str(expected[task_id].task.queued_at_ms),
                        ),
                        self.client.put(self._counter_key(pool_id), str(count + 1)),
                    ],
                )
                conflicted = conflicted or not succeeded
            for task_id in sorted(set(ordered) - set(expected)):
                task_record = await self.client.get(self._task_key(task_id))
                if task_record is not None:
                    current = self._task(task_record)
                    if (
                        current.task.pool_id == pool_id
                        and current.task.status == TaskStatus.QUEUED
                    ):
                        expected_ordered_key = self._ordered_queue_key(current.task)
                        repair = [
                            *[
                                self.client.put(key, task_id)
                                for key in self._all_task_index_keys(current.task)
                            ],
                            self.client.put(expected_ordered_key, task_id),
                            self.client.put(
                                self._expiry_index_key(current.task), task_id
                            ),
                        ]
                        if ordered[task_id].key != expected_ordered_key:
                            repair.append(self.client.delete(ordered[task_id].key))
                        await self.client.txn(
                            [
                                self.client.compare_mod(
                                    task_record.key, task_record.mod_revision
                                ),
                                self.client.compare_mod(
                                    ordered[task_id].key,
                                    ordered[task_id].mod_revision,
                                ),
                            ],
                            repair,
                        )
                        conflicted = True
                        continue
                task_compare = (
                    self.client.compare_mod(task_record.key, task_record.mod_revision)
                    if task_record is not None
                    else self.client.compare_version(self._task_key(task_id), 0)
                )
                succeeded, _ = await self.client.txn(
                    [
                        self.client.compare_mod(
                            ordered[task_id].key,
                            ordered[task_id].mod_revision,
                        ),
                        task_compare,
                    ],
                    [self.client.delete(ordered[task_id].key)],
                )
                conflicted = conflicted or not succeeded
            for task_id in sorted(set(expected) - set(ordered)):
                task = expected[task_id]
                ordered_key = self._ordered_queue_key(task.task)
                succeeded, _ = await self.client.txn(
                    [
                        self.client.compare_mod(
                            self._task_key(task_id), task.revision
                        ),
                        self.client.compare_version(ordered_key, 0),
                    ],
                    [self.client.put(ordered_key, task_id)],
                )
                conflicted = conflicted or not succeeded
            if conflicted:
                continue

            counter_key = self._counter_key(pool_id)
            _count, counter = await self._counter(pool_id)
            current_queue, _ = await self.client.range_all(
                self._queue_prefix(pool_id),
                prefix=True,
                keys_only=True,
            )
            succeeded, _ = await self.client.txn(
                self._counter_compare(counter_key, counter),
                [self.client.put(counter_key, str(len(current_queue)))],
            )
            if succeeded:
                return
        raise StoreConflict(
            "unable to reconcile queue after repeated etcd CAS conflicts"
        )


def terminal_error(code: str, message: str, *, retryable: bool = False) -> TaskError:
    return TaskError(code=code, message=message, retryable=retryable)

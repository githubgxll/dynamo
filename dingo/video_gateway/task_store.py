# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent task, FIFO queue, idempotency and Worker lease operations."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from dingo.video_gateway.errors import GatewayError, StoreConflict
from dingo.video_gateway.etcd_http import EtcdHttpClient, EtcdValue
from dingo.video_gateway.models import (
    ACTIVE_STATUSES,
    ALLOWED_TRANSITIONS,
    StoredTask,
    TaskError,
    TaskStatus,
    VideoTask,
    WorkerLease,
    now_ms,
)


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
    async def list_queued(
        self, pool_id: str, *, limit: int = 100
    ) -> list[StoredTask]: ...

    @abstractmethod
    async def queue_depth(self, pool_id: str) -> int: ...

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
    async def request_cancel(self, task_id: str) -> StoredTask: ...

    @abstractmethod
    async def list_leases(self, pool_id: str) -> list[WorkerLease]: ...

    @abstractmethod
    async def release_lease(self, pool_id: str, worker_key_value: str) -> None: ...

    @abstractmethod
    async def heartbeat_lease(
        self, pool_id: str, worker_key_value: str, task_id: str
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

    def _next_revision(self) -> int:
        self._revision += 1
        return self._revision

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
            revision = self._next_revision()
            self._tasks[task.id] = (_clone_task(task), revision)
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
            values = [self._stored(task_id) for task_id in sorted(self._tasks)]
            filtered = [value for value in values if value is not None]
            if pool_id is not None:
                filtered = [
                    value for value in filtered if value.task.pool_id == pool_id
                ]
            if status is not None:
                filtered = [value for value in filtered if value.task.status == status]
            if after is not None:
                filtered = [
                    value
                    for value in filtered
                    if (value.task.id < after if descending else value.task.id > after)
                ]
            if descending:
                filtered.reverse()
            return filtered[:limit]

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
                    "assigned_at_ms": now_ms(),
                    "deadline_at_ms": deadline_at_ms,
                },
            )
            revision = self._next_revision()
            self._tasks[stored.task.id] = (updated, revision)
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
                if lease is not None and quarantine_until_ms is not None:
                    lease.state = "quarantined"
                    lease.reuse_after_ms = quarantine_until_ms
                    lease.heartbeat_at_ms = now_ms()
                else:
                    self._leases.pop(key, None)
            revision = self._next_revision()
            self._tasks[task_id] = (updated, revision)
            return StoredTask(_clone_task(updated), revision)

    async def request_cancel(self, task_id: str) -> StoredTask:
        for _ in range(8):
            stored = await self.get_task(task_id)
            if stored is None:
                raise KeyError(task_id)
            if stored.task.status == TaskStatus.QUEUED:
                patch = {
                    "status": TaskStatus.CANCELLED,
                    "cancel_requested_at_ms": now_ms(),
                    "completed_at_ms": now_ms(),
                    "expires_at_ms": now_ms() + 60 * 60 * 1000,
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
        self, pool_id: str, worker_key_value: str, task_id: str
    ) -> None:
        async with self._lock:
            lease = self._leases.get((pool_id, worker_key_value))
            if lease is not None and lease.task_id == task_id:
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
            self._next_revision()
            return True

    async def reconcile_pool(self, pool_id: str) -> None:
        async with self._lock:
            queued = sorted(
                task_id
                for task_id, (task, _revision) in self._tasks.items()
                if task.pool_id == pool_id and task.status == TaskStatus.QUEUED
            )
            self._queue[pool_id] = queued


class EtcdTaskStore(TaskStore):
    """etcd-backed single-Gateway task state with optimistic transactions."""

    def __init__(
        self, client: EtcdHttpClient, *, prefix: str, deployment_id: str
    ) -> None:
        self.client = client
        self.root = f"{prefix.rstrip('/')}/deployments/{deployment_id}"

    def _task_key(self, task_id: str) -> str:
        return f"{self.root}/tasks/{task_id}"

    def _queue_key(self, pool_id: str, task_id: str) -> str:
        return f"{self.root}/pools/{pool_id}/queue/{task_id}"

    def _queue_prefix(self, pool_id: str) -> str:
        return f"{self.root}/pools/{pool_id}/queue/"

    def _counter_key(self, pool_id: str) -> str:
        return f"{self.root}/pools/{pool_id}/counters/queued"

    def _lease_key(self, pool_id: str, worker_key_value: str) -> str:
        return f"{self.root}/pools/{pool_id}/worker-leases/{worker_key_value}"

    def _lease_prefix(self, pool_id: str) -> str:
        return f"{self.root}/pools/{pool_id}/worker-leases/"

    def _idempotency_key(self, principal_hash: str, key_hash: str) -> str:
        return f"{self.root}/idempotency/{principal_hash}/{key_hash}"

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
            if count >= queue_limit:
                raise GatewayError(429, "queue_full", "video queue is full")
            compare = [
                self.client.compare_version(task_key, 0),
                self.client.compare_version(queue_key, 0),
                *self._counter_compare(counter_key, counter),
            ]
            success = [
                self.client.put(task_key, self._encode(task.to_dict())),
                self.client.put(queue_key, str(task.queued_at_ms)),
                self.client.put(counter_key, str(count + 1)),
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
                return StoredTask(VideoTask.from_dict(task.to_dict()), revision), True
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

    async def list_tasks(
        self,
        *,
        pool_id: str | None = None,
        status: TaskStatus | None = None,
        after: str | None = None,
        limit: int = 100,
        descending: bool = False,
    ) -> list[StoredTask]:
        values = await self.client.range(f"{self.root}/tasks/", prefix=True)
        tasks = [self._task(value) for value in values]
        if pool_id is not None:
            tasks = [value for value in tasks if value.task.pool_id == pool_id]
        if status is not None:
            tasks = [value for value in tasks if value.task.status == status]
        if after is not None:
            tasks = [
                value
                for value in tasks
                if (value.task.id < after if descending else value.task.id > after)
            ]
        tasks.sort(key=lambda value: value.task.id, reverse=descending)
        return tasks[:limit]

    async def list_queued(self, pool_id: str, *, limit: int = 100) -> list[StoredTask]:
        values = await self.client.range(self._queue_prefix(pool_id), prefix=True)
        result: list[StoredTask] = []
        for value in values:
            task_id = value.key.rsplit("/", 1)[-1]
            task = await self.get_task(task_id)
            if task is not None and task.task.status == TaskStatus.QUEUED:
                result.append(task)
                if len(result) >= limit:
                    break
        return result

    async def queue_depth(self, pool_id: str) -> int:
        count, _ = await self._counter(pool_id)
        return count

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
        count, counter = await self._counter(stored.task.pool_id)
        if count <= 0:
            return None
        updated = _apply_patch(
            stored.task,
            {
                "status": TaskStatus.DISPATCHING,
                "attempt": stored.task.attempt + 1,
                "worker_instance_id": lease.worker_instance_id,
                "worker_key": lease.worker_key,
                "owner_generation": lease.owner_generation,
                "assigned_at_ms": now_ms(),
                "deadline_at_ms": deadline_at_ms,
            },
        )
        compare = [
            self.client.compare_mod(task_key, stored.revision),
            self.client.compare_version(queue_key, 0, result="GREATER"),
            self.client.compare_version(lease_key, 0),
            *self._counter_compare(counter_key, counter),
        ]
        success = [
            self.client.put(task_key, self._encode(updated.to_dict())),
            self.client.put(lease_key, self._encode(lease.to_dict())),
            self.client.delete(queue_key),
            self.client.put(counter_key, str(count - 1)),
        ]
        succeeded, revision = await self.client.txn(compare, success)
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
                    self.client.put(counter_key, str(count - 1)),
                ]
            )

        if release_lease and stored.task.worker_key is not None:
            lease_key = self._lease_key(stored.task.pool_id, stored.task.worker_key)
            lease_value = await self.client.get(lease_key)
            if lease_value is not None:
                compare.append(
                    self.client.compare_mod(lease_key, lease_value.mod_revision)
                )
                if quarantine_until_ms is None:
                    success.append(self.client.delete(lease_key))
                else:
                    lease = WorkerLease.from_dict(json.loads(lease_value.value))
                    lease.state = "quarantined"
                    lease.reuse_after_ms = quarantine_until_ms
                    lease.heartbeat_at_ms = now_ms()
                    success.append(
                        self.client.put(lease_key, self._encode(lease.to_dict()))
                    )

        succeeded, revision = await self.client.txn(compare, success)
        if not succeeded:
            raise StoreConflict("task transition lost an etcd CAS race")
        return StoredTask(updated, revision)

    async def request_cancel(self, task_id: str) -> StoredTask:
        for _ in range(8):
            stored = await self.get_task(task_id)
            if stored is None:
                raise KeyError(task_id)
            if stored.task.status == TaskStatus.QUEUED:
                patch = {
                    "status": TaskStatus.CANCELLED,
                    "cancel_requested_at_ms": now_ms(),
                    "completed_at_ms": now_ms(),
                    "expires_at_ms": now_ms() + 60 * 60 * 1000,
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

    async def release_lease(self, pool_id: str, worker_key_value: str) -> None:
        key = self._lease_key(pool_id, worker_key_value)
        value = await self.client.get(key)
        if value is None:
            return
        await self.client.txn(
            [self.client.compare_mod(key, value.mod_revision)],
            [self.client.delete(key)],
        )

    async def heartbeat_lease(
        self, pool_id: str, worker_key_value: str, task_id: str
    ) -> None:
        key = self._lease_key(pool_id, worker_key_value)
        value = await self.client.get(key)
        if value is None:
            return
        lease = WorkerLease.from_dict(json.loads(value.value))
        if lease.task_id != task_id:
            return
        lease.heartbeat_at_ms = now_ms()
        lease.owner_expires_at_ms = now_ms() + 15_000
        await self.client.txn(
            [self.client.compare_mod(key, value.mod_revision)],
            [self.client.put(key, self._encode(lease.to_dict()))],
        )

    async def delete_expired(self, stored: StoredTask) -> bool:
        if stored.task.status != TaskStatus.EXPIRED:
            return False
        task_key = self._task_key(stored.task.id)
        compare = [self.client.compare_mod(task_key, stored.revision)]
        success = [self.client.delete(task_key)]
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
        for _ in range(8):
            tasks = await self.list_tasks(pool_id=pool_id, limit=10_000)
            expected = {
                stored.task.id: stored
                for stored in tasks
                if stored.task.status == TaskStatus.QUEUED
            }
            queue_values = await self.client.range(
                self._queue_prefix(pool_id), prefix=True
            )
            existing = {value.key.rsplit("/", 1)[-1]: value for value in queue_values}
            counter_key = self._counter_key(pool_id)
            _count, counter = await self._counter(pool_id)
            compare: list[dict] = [
                self.client.compare_mod(self._task_key(task_id), stored.revision)
                for task_id, stored in expected.items()
            ]
            compare.extend(
                self.client.compare_mod(value.key, value.mod_revision)
                for value in existing.values()
            )
            compare.extend(self._counter_compare(counter_key, counter))
            success: list[dict] = []
            for task_id in sorted(set(existing) - set(expected)):
                success.append(self.client.delete(existing[task_id].key))
            for task_id in sorted(set(expected) - set(existing)):
                success.append(
                    self.client.put(
                        self._queue_key(pool_id, task_id),
                        str(expected[task_id].task.queued_at_ms),
                    )
                )
            success.append(self.client.put(counter_key, str(len(expected))))
            succeeded, _ = await self.client.txn(compare, success)
            if succeeded:
                return
        raise StoreConflict(
            "unable to reconcile queue after repeated etcd CAS conflicts"
        )


def terminal_error(code: str, message: str, *, retryable: bool = False) -> TaskError:
    return TaskError(code=code, message=message, retryable=retryable)

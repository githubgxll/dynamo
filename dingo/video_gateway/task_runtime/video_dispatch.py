# SPDX-License-Identifier: Apache-2.0
"""Video/SR queue, budget and lease policy for the shared dispatch coordinator."""
import logging
import secrets
from dataclasses import dataclass

from dingo.video_gateway.memory_budget import task_memory_bytes
from dingo.video_gateway.models import StoredTask, WorkerLease, now_ms
from dingo.video_gateway.task_store import retry_excludes_worker, worker_key

from .coordinator import DispatchClaim


@dataclass
class VideoSelection:
    stored: StoredTask
    slots: list
    retry_ledger: bool
    retry_limit: int
    prepared: bool = False
    lease: WorkerLease | None = None
    adopting: bool = False


class VideoSchedulingAdapter:
    def __init__(self, dispatcher, pool):
        self.dispatcher, self.pool = dispatcher, pool

    async def candidates(self):
        d, pool = self.dispatcher, self.pool
        if (
            sum(
                p == pool.config.pool_id for p in getattr(d, "_finalizing", {}).values()
            )
            >= pool.config.scheduling.finalization_pending_limit
        ):
            await d._clear_budget_waiter(pool)
            return []
        # A healthy lease view can prove that no instance is usable without
        # reading queue indexes, task bodies, or shared retry counters. Keep
        # the later checks and reserve CAS: availability can change while the
        # awaited queue reads are in flight.
        skip_reason = None
        if not pool.instance_ids:
            skip_reason = "no_workers"
        elif d.store.lease_watch_supported and not pool.lease_watch_healthy:
            skip_reason = "lease_watch_unhealthy"
        else:
            occupied = {
                lease.worker_key for lease in await d.pool_leases(pool.config.pool_id)
            }
            if all(
                worker_key(pool.config.backend_target, instance, slot) in occupied
                for instance, slot in d._worker_slots(pool)
            ):
                skip_reason = "no_free_worker"
        if skip_reason is not None:
            await d._clear_budget_waiter(pool)
            d.telemetry.increment(
                "dingo_video_dispatch_skips_total",
                labels={"pool": pool.config.pool_id, "reason": skip_reason},
            )
            return []
        ledger = hasattr(d.store, "retry_budget_used")
        budget_limit = getattr(d, "_retry_budget_limit", 32)
        queued = await d.store.list_queued(
            pool.config.pool_id,
            limit=min(10000, pool.config.scheduling.queue_limit + budget_limit),
        )
        used = 0
        if ledger:
            waiting = await d.store.retry_queue_depth(pool.config.pool_id)
            used = await d.store.retry_budget_used(pool.config.pool_id)
            d.telemetry.set_gauge(
                "dingo_video_retry_waiting_tasks",
                waiting,
                labels={"pool": pool.config.pool_id},
            )
            d.telemetry.set_gauge(
                "dingo_video_retry_credits_used",
                used,
                labels={"pool": pool.config.pool_id},
            )
            d.telemetry.set_gauge(
                "dingo_video_normal_queue_depth",
                max(0, await d.store.queue_depth(pool.config.pool_id) - waiting),
                labels={"pool": pool.config.pool_id},
            )
        if not queued or not pool.instance_ids:
            await d._clear_budget_waiter(pool)
            return []
        if d.store.lease_watch_supported and not pool.lease_watch_healthy:
            await d._clear_budget_waiter(pool)
            return []
        leased = {
            lease.worker_key for lease in await d.pool_leases(pool.config.pool_id)
        }
        selected = None
        available = []
        # Retry first, but do not let a retry that excludes the only available
        # instance block unrelated runnable work behind it.
        for candidate in sorted(queued, key=lambda item: item.task.attempt == 0):
            if (
                candidate.task.normalized_request.get("_pipeline") is not None
                or candidate.task.id in d.running_calls
                or candidate.task.expires_at_ms <= now_ms()
            ):
                continue
            if (
                ledger
                and getattr(d, "_worker_retry_once", False)
                and candidate.task.attempt == 0
                and used >= budget_limit
            ):
                continue
            available = [
                (instance, slot)
                for instance, slot in d._worker_slots(
                    pool, candidate.task.normalized_request
                )
                if worker_key(pool.config.backend_target, instance, slot) not in leased
                and not retry_excludes_worker(
                    candidate.task, pool.config.backend_target, instance
                )
            ]
            if available:
                selected = candidate
                break
        if selected is None:
            await d._clear_budget_waiter(pool)
            return []
        return [VideoSelection(selected, available, ledger, budget_limit)]

    def key(self, candidate):
        return "video-task-" + candidate.stored.task.id

    async def claim(self, candidate):
        d, pool = self.dispatcher, self.pool
        selected = candidate.stored
        available = candidate.slots
        ledger, budget_limit = candidate.retry_ledger, candidate.retry_limit
        task_id = selected.task.id
        if pool.budget_waiter_id not in {None, task_id}:
            await d.memory_budget.cancel_waiter(pool.budget_waiter_id)
            pool.budget_waiter_id = None
        weight_bytes = task_memory_bytes(
            d.config.media,
            selected.task.normalized_request,
            persisted_estimate=selected.task.estimated_payload_bytes,
        )
        if not await d.memory_budget.try_acquire(task_id, weight_bytes):
            pool.budget_waiter_id = task_id
            return DispatchClaim()
        pool.budget_waiter_id = None
        candidate.prepared = True
        if d._draining:
            return DispatchClaim()
        if candidate.adopting:
            from .transactions import UnconfirmedTransaction

            async def observe(operation):
                try:
                    reserved = await operation()
                except UnconfirmedTransaction as unknown:
                    resolver = unknown.resolve
                    async def resolve():
                        return await observe(resolver)
                    raise UnconfirmedTransaction(resolve) from None
                except Exception:
                    d.telemetry.increment("dingo_video_ha_takeovers_total",
                        labels={"pool": pool.config.pool_id, "outcome": "failed"})
                    raise
                if reserved is None:
                    d.telemetry.increment("dingo_video_ha_takeovers_total",
                        labels={"pool": pool.config.pool_id, "outcome": "contended"})
                return reserved

            reserved = await observe(lambda: d.store.claim_orphaned_active(
                selected, new_owner_generation=d.generation))
            return DispatchClaim(reserved)
        index = pool.cursor % len(available)
        instance_id, slot_id = available[index]
        pool.cursor = (index + 1) % len(available)
        key = worker_key(pool.config.backend_target, instance_id, slot_id)
        lease = WorkerLease(
            pool_id=pool.config.pool_id,
            worker_key=key,
            worker_instance_id=instance_id,
            backend_target=pool.config.backend_target,
            task_id=selected.task.id,
            owner_generation=d.generation,
            execution_token=secrets.token_hex(16),
            state="reserved",
            heartbeat_at_ms=now_ms(),
            owner_expires_at_ms=now_ms() + 15_000,
        )
        deadline = now_ms() + int(pool.config.scheduling.execution_timeout_s * 1000)
        retry_options = (
            {
                "reserve_retry": getattr(d, "_worker_retry_once", False)
                and pool.config.execution_mode == "detached",
                "retry_limit": budget_limit,
            }
            if ledger
            else {}
        )
        candidate.lease = lease
        reserved = await d.store.reserve(
            selected, lease, deadline_at_ms=deadline, **retry_options
        )
        if reserved is None:
            return DispatchClaim(retry_now=True)
        return DispatchClaim(reserved)

    def committed(self, candidate, reserved):
        d, pool = self.dispatcher, self.pool
        selected = candidate.stored
        lease = candidate.lease
        if candidate.adopting:
            task = reserved.task
            lease = WorkerLease(
                pool_id=task.pool_id,
                worker_key=task.worker_key,
                worker_instance_id=task.worker_instance_id,
                backend_target=task.backend_target,
                task_id=task.id,
                owner_generation=d.generation,
                execution_token=task.execution_token,
                state="running",
                heartbeat_at_ms=now_ms(),
                owner_expires_at_ms=now_ms() + 15_000,
                etcd_lease_id=task.worker_lease_id,
            )
            d.telemetry.increment(
                "dingo_video_ha_takeovers_total",
                labels={"pool": task.pool_id, "outcome": "claimed"},
            )
            d.telemetry.audit_task(
                "ha_takeover",
                task,
                gateway_generation=d.generation,
                previous_status=selected.task.status.value,
                revision=reserved.revision,
                reason="expired_gateway_owner",
            )
            logging.getLogger(__name__).info(
                "claimed detached task %s from expired Gateway owner", task.id
            )
        else:
            d.telemetry.record_transition(
                "reserved",
                selected.task,
                reserved.task,
                gateway_generation=d.generation,
                revision=reserved.revision,
            )
        if d.store.lease_watch_supported:
            lease.etcd_lease_id = reserved.task.worker_lease_id
            pool.lease_cache[lease.worker_key] = lease

    async def release_preparation(self, candidate):
        if candidate.prepared:
            await self.dispatcher.memory_budget.release(candidate.stored.task.id)
            candidate.prepared = False

    async def execute(self, record):
        await self.dispatcher._run_reserved(self.pool, record)

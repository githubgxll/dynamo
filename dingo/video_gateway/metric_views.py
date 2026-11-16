# SPDX-License-Identifier: Apache-2.0
"""Active user requests, without counting internal pipeline records twice."""
from dingo.video_gateway.models import TaskStatus


async def request_stages(service, counts):
    # Read only nonterminal indexes, never retained result history. Classify
    # records directly: subtracting independent count/child snapshots can double
    # count a child that completes between those reads.
    result = {pool: {"gateway_queue": 0, "worker": 0} for pool in counts}
    seen = set()
    for pool in service.config.pools:
        for status in (TaskStatus.QUEUED, TaskStatus.DISPATCHING,
                       TaskStatus.IN_PROGRESS, TaskStatus.FINALIZING):
            after = None
            while True:
                rows = await service.store.list_tasks(pool_id=pool.pool_id,
                    status=status, after=after, limit=64)
                if not rows:
                    break
                for row in rows:
                    task = row.task
                    if task.id in seen or task.normalized_request.get("_pipeline_child"):
                        continue
                    seen.add(task.id)
                    queueing = task.status == TaskStatus.QUEUED
                    if task.normalized_request.get("_pipeline") is not None and task.status != TaskStatus.FINALIZING:
                        plan = task.normalized_request.get("_pipeline", {})
                        tid = plan.get("children", {}).get(plan.get("phase"))
                        stored = await service.store.get_task(tid) if tid else None
                        child = stored.task if stored else None
                        queueing = queueing or child is None or child.status == TaskStatus.QUEUED or (
                            plan.get("phase") == "base" and child.status == TaskStatus.COMPLETED
                            and "super_resolution" in task.normalized_request)
                    result[pool.pool_id]["gateway_queue" if queueing else "worker"] += 1
                next_id = rows[-1].task.id
                if next_id == after:
                    raise RuntimeError("non-advancing active task cursor")
                after = next_id
    return result

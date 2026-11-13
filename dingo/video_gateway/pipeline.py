# SPDX-License-Identifier: Apache-2.0
"""Durable H3 -> independent SR orchestration, with fenced child admission.

Coordinators may run in multiple Gateways. Every state change is a CAS; creating
a child and linking it to its parent is one TaskStore transaction. No in-memory
coroutine owns the only copy of the execution plan or a completed base result.
"""
import asyncio
import copy
import hashlib
import logging
import os
from pathlib import Path
import uuid

from dingo.common.video_sr_timings import SR_TIMINGS_KEY
from dingo.common.video_timing_schemas import MODEL_EXECUTION_KEY
from dingo.video_gateway.adapters.base import UploadedArtifact
from dingo.video_gateway.adapters.h3_resolution import classify_resolution
from dingo.video_gateway.errors import GatewayError, StoreConflict
from dingo.video_gateway.file_io import run_file_io
from dingo.video_gateway.models import TaskStatus, TaskError, TERMINAL_STATUSES, now_ms

LOG = logging.getLogger(__name__)


def copy_checked(source, destination, size, sha256):
    """Create a private immutable copy; never trust a stale size/hash alone."""
    digest = hashlib.sha256()
    count = 0
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(fd, 'rb') as src, Path(destination).open('xb') as dst:
            if os.fstat(src.fileno()).st_size != size:
                raise ValueError('pipeline source size changed')
            while chunk := src.read(1024 * 1024):
                count += len(chunk)
                if count > size:
                    raise ValueError('pipeline source grew')
                digest.update(chunk)
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if count != size or digest.hexdigest() != sha256:
            raise ValueError('pipeline source checksum changed')
    except BaseException:
        Path(destination).unlink(missing_ok=True)
        raise


class VideoPipelineCoordinator:
    def __init__(self, service):
        self.service = service
        self.store = service.store
        self.artifacts = service.artifacts
        self.config = service.config
        self.dispatcher = service.dispatcher
        self.pools = [p for p in self.config.pools if p.pipeline is not None]

    async def run(self):
        while True:
            try:
                await self.tick()
            except Exception:
                LOG.exception('pipeline scan temporarily unavailable')
            await asyncio.sleep(.5)

    async def tick(self):
        if not self.dispatcher.draining:
            for pool in self.pools:
                for status in (TaskStatus.QUEUED, TaskStatus.DISPATCHING,
                               TaskStatus.IN_PROGRESS, TaskStatus.FINALIZING):
                    after = None
                    while True:
                        tasks = await self.store.list_tasks(pool_id=pool.pool_id,
                            status=status, after=after, limit=64)
                        if not tasks:
                            break
                        for stored in tasks:
                            try:
                                await self.advance(stored, pool)
                            except (StoreConflict, KeyError):
                                pass
                            except GatewayError as exc:
                                # Queue pressure / temporarily absent workers are retryable.
                                LOG.warning('pipeline admission deferred task=%s error=%s', stored.task.id, exc)
                            except Exception:
                                LOG.exception('pipeline reconciliation deferred task=%s', stored.task.id)
                        after = tasks[-1].task.id

    async def transition(self, stored, patch, event):
        updated = await self.store.transition(stored.task.id,
            expected={stored.task.status}, expected_revision=stored.revision, patch=patch)
        self.service.telemetry.record_transition(event, stored.task, updated.task,
            gateway_generation=self.dispatcher.generation, revision=updated.revision)
        return updated

    async def finish(self, stored, status, error=None, *, internal_error=None):
        ttl = (self.config.lifecycle.cancelled_ttl_s if status == TaskStatus.CANCELLED
               else self.config.lifecycle.failed_ttl_s)
        return await self.transition(stored, {'status':status, 'error':error, 'internal_error':internal_error,
            'completed_at_ms':now_ms(), 'expires_at_ms':now_ms()+int(ttl*1000)}, status.value)

    async def advance(self, stored, pool):
        task = stored.task
        state = task.normalized_request['_pipeline']
        if task.status == TaskStatus.QUEUED:
            counts = await self.store.task_counts(pool.pool_id)
            active = sum(counts.get(status, 0) for status in
                         (TaskStatus.DISPATCHING, TaskStatus.IN_PROGRESS, TaskStatus.FINALIZING))
            if active >= pool.pipeline['max_active_parents']:
                return  # Bound parent fan-out while downstream queues are full.
            return await self.transition(stored, {'status':TaskStatus.DISPATCHING,
                'started_at_ms':now_ms(), 'deadline_at_ms':now_ms()+int(pool.scheduling.execution_timeout_s*1000),
                'queue_wait_s':max(0, (now_ms()-task.queued_at_ms)/1000)}, 'pipeline_started')
        if task.status == TaskStatus.DISPATCHING:
            return await self.transition(stored, {'status':TaskStatus.IN_PROGRESS}, 'pipeline_started')

        children = {}
        for phase, child_id in state['children'].items():
            child = await self.store.get_task(child_id)
            if child is None:
                return await self.finish(stored, TaskStatus.FAILED,
                    TaskError('pipeline_child_missing', 'persisted stage task is missing'))
            children[phase] = child
        timed_out = task.deadline_at_ms is not None and task.deadline_at_ms <= now_ms()
        if task.cancel_requested_at_ms is not None or timed_out:
            active = [c for c in children.values() if c.task.status not in TERMINAL_STATUSES]
            for child in active:
                await self.dispatcher.cancel(child.task.id)
            if active:
                return
            status = TaskStatus.CANCELLED if task.cancel_requested_at_ms is not None else TaskStatus.FAILED
            return await self.finish(stored, status, TaskError(
                'cancelled' if status == TaskStatus.CANCELLED else 'pipeline_timeout',
                'video task was cancelled' if status == TaskStatus.CANCELLED else 'pipeline execution deadline exceeded'))

        # Refuse semantic drift across a restart. Existing children are never regenerated.
        for phase in ('base', 'sr'):
            actual = self.config.pools_by_id.get(state[phase+'_pool'])
            if actual is None or actual.configuration_revision != state[phase+'_revision']:
                for child in children.values():
                    if child.task.status not in TERMINAL_STATUSES:
                        await self.dispatcher.cancel(child.task.id)
                        return
                return await self.finish(stored, TaskStatus.FAILED,
                    TaskError('configuration_changed', 'pipeline stage configuration changed'))
        phase = state['phase']
        if task.status == TaskStatus.FINALIZING:
            return await self.publish(stored, children[phase].task)
        child = children.get(phase)
        if child is None:
            return await self.create_child(stored, phase, children)
        if child.task.status not in TERMINAL_STATUSES:
            return
        if child.task.status != TaskStatus.COMPLETED:
            return await self.finish(stored, TaskStatus.FAILED,
                child.task.error or TaskError('pipeline_stage_failed', phase+' stage failed'),
                internal_error=child.task.internal_error)
        if phase == 'base' and 'super_resolution' in task.normalized_request:
            state = copy.deepcopy(state)
            state['phase'] = 'sr'
            return await self.transition(stored, {'normalized_request':{
                **task.normalized_request, '_pipeline':state}}, 'pipeline_sr_queued')

        base = children['base'].task
        normalized = dict(task.normalized_request)
        for key, source in ((MODEL_EXECUTION_KEY, base), (SR_TIMINGS_KEY, child.task)):
            if key in source.normalized_request:
                normalized[key] = source.normalized_request[key]
        normalized['_pipeline_metrics'] = {
            'base_generation_s':base.inference_time_s,
            'base_queue_wait_s':base.queue_wait_s,
            'super_resolution_s':child.task.inference_time_s if phase == 'sr' else None,
            'sr_queue_wait_s':child.task.queue_wait_s if phase == 'sr' else None,
        }
        stages = {'base_generation': base.inference_time_s or 0}
        if phase == 'sr':
            stages['super_resolution'] = child.task.inference_time_s or 0
        return await self.transition(stored, {'status':TaskStatus.FINALIZING,
            'normalized_request':normalized,
            'inference_time_s':sum(stages.values()), 'stage_durations':stages}, 'finalization_started')

    async def create_child(self, parent, phase, children):
        task = parent.task
        state = task.normalized_request['_pipeline']
        pool = self.config.pools_by_id[state[phase+'_pool']]
        # Avoid reading/copying files on every tick while a stage is unavailable.
        if not self.dispatcher.has_workers(pool.pool_id):
            return
        if await self.store.queue_depth(pool.pool_id) >= pool.scheduling.queue_limit:
            return
        root = await self.artifacts.create_upload()
        uploads = []
        prepared = None
        try:
            if phase == 'base':
                manifest = await self.artifacts.read_json(Path(task.input_manifest_path))
                parent_root = await self.artifacts.resolve_task_root(task.deployment_id, task.pool_id, task.id)
                for entry in manifest:
                    source = self.artifacts.result_path(parent_root / entry['path'])
                    destination = root / 'inputs' / (str(entry['ordinal'])+'-'+Path(entry['path']).name)
                    await run_file_io(copy_checked, source, destination, entry['bytes'], entry['sha256'])
                    uploads.append(UploadedArtifact(entry['field_name'],entry['ordinal'],entry['filename'],
                        entry['content_type'],destination,entry['bytes'],entry['sha256']))
                prepared = copy.deepcopy(task.normalized_request)
                prepared.pop('_pipeline', None)
                sr = prepared.pop('super_resolution', None)
                if sr:
                    prepared.update(width=sr['base_width'],height=sr['base_height'])
                prepared.update(public_model=pool.served_models[0], backend_model=pool.backend_model)
                # Transport is selected by the actual base pool, not the parent.
                prepared.pop('shared_inputs', None)
                if pool.adapter.options.get('reference_transport') == 'shared_files_v1':
                    prepared['shared_inputs']={'storage_id':pool.adapter.options['shared_storage_id']}
                fields = {'model':[pool.served_models[0]]}
            else:
                base = children['base'].task
                source = self.artifacts.result_path(base.result_path)
                destination = root / 'inputs' / 'base.mp4'
                await run_file_io(copy_checked, source, destination, base.result_bytes, base.result_sha256)
                uploads=[UploadedArtifact('input_references',0,'base.mp4','video/mp4',destination,
                    base.result_bytes,base.result_sha256)]
                tier = classify_resolution(task.normalized_request['width'], task.normalized_request['height'], super_resolution=True)
                fields={'model':[pool.served_models[0]],'operation':['sr_only'],'resolution':[tier]}
                # The SR adapter must agree with the already admitted parent canvas.
                normalized = await run_file_io(self.service.adapters[pool.pool_id].normalize_request,
                                              fields, uploads, pool.served_models[0])
                if (normalized['width'],normalized['height']) != (task.normalized_request['width'],task.normalized_request['height']):
                    raise ValueError('SR source selects a different canvas than the admitted parent')
        except Exception as exc:
            await self.artifacts.discard(root)
            return await self.finish(parent, TaskStatus.FAILED,
                TaskError('pipeline_input_invalid', str(exc)))
        # TaskStore atomically checks this parent revision and cancellation bit,
        # admits the child, and records its ID in the parent. Losers cannot launch work.
        await self.service.submit(fields=fields,uploads=uploads,upload_root=root,
            delivery_mode='async',idempotency_key=phase,principal='pipeline:'+task.id,
            pipeline_parent=parent,pipeline_phase=phase,prepared_request=prepared)

    async def publish(self, stored, child):
        task = stored.task
        root = await self.artifacts.resolve_task_root(task.deployment_id,task.pool_id,task.id)
        destination = root / ('result-pipeline-'+uuid.uuid4().hex+'.mp4')
        try:
            source = self.artifacts.result_path(child.result_path)
            await run_file_io(copy_checked,source,destination,child.result_bytes,child.result_sha256)
            updated = await self.transition(stored, {'status':TaskStatus.COMPLETED,
                'result_path':str(destination),'result_bytes':child.result_bytes,'result_sha256':child.result_sha256,
                'completed_at_ms':now_ms(),'expires_at_ms':now_ms()+int(self.config.lifecycle.completed_ttl_s*1000),
                'normalized_request':{**task.normalized_request,'_result_media':child.normalized_request.get('_result_media',{})}},
                'completed')
            return updated
        except StoreConflict:
            destination.unlink(missing_ok=True)
            raise
        except Exception:
            # Preserve a candidate after an ambiguous store write; next tick
            # reads the durable task before deciding whether another copy is needed.
            raise

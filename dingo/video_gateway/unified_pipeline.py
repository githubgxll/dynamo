# SPDX-License-Identifier: Apache-2.0
"""Atomic first-stage admission into the public generation pool's only queue."""
import copy
import hashlib
from dataclasses import replace
from pathlib import Path

from dingo.video_gateway.errors import GatewayError, StoreConflict
from dingo.video_gateway.file_io import run_file_io
from dingo.video_gateway.ids import new_video_id
from dingo.video_gateway.models import TaskStatus, now_ms
from dingo.video_gateway.pipeline import copy_checked


async def admit_base(service, parent, pool):
    root = await service.artifacts.create_upload()
    child_root = None
    child_id = new_video_id()
    try:
        manifest = await service.artifacts.read_json(Path(parent.input_manifest_path))
        parent_root = await service.artifacts.resolve_task_root(parent.deployment_id, pool.pool_id, parent.id)
        for entry in manifest:
            destination = root / entry['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            await run_file_io(copy_checked, service.artifacts.result_path(parent_root / entry['path']),
                              destination, entry['bytes'], entry['sha256'])
        normalized = copy.deepcopy(parent.normalized_request)
        normalized.pop('_pipeline')
        sr = normalized.pop('super_resolution')
        normalized.update(width=sr['base_width'], height=sr['base_height'],
                          _pipeline_child={'parent_id':parent.id,'phase':'base'})
        child_root = await service.artifacts.commit_upload(root, parent.deployment_id, pool.pool_id, child_id,
            artifact_manifest={'schema_version':1,'task_id':child_id,'deployment_id':parent.deployment_id,
                'pool_id':pool.pool_id,'created_at_ms':parent.created_at_ms,'expires_at_ms':parent.expires_at_ms})
        await service.artifacts.write_json(child_root/'request.json', normalized)
        await service.artifacts.write_json(child_root/'input-manifest.json', manifest)
        stamp = now_ms()
        child = replace(parent, id=child_id, status=TaskStatus.QUEUED, created_at_ms=stamp,
            queued_at_ms=stamp, started_at_ms=None, deadline_at_ms=None, queue_wait_s=None,
            idempotency_hash=None, request_digest='sha256:'+hashlib.sha256((parent.request_digest+':base').encode()).hexdigest(),
            request_path=str(child_root/'request.json'), input_manifest_path=str(child_root/'input-manifest.json'),
            normalized_request=normalized, configuration_revision=pool.request_configuration_revision(normalized))
        stored, created = await service.store.create_task(child,
            principal_hash=parent.principal_hash, idempotency_hash=parent.idempotency_hash,
            queue_limit=pool.scheduling.queue_limit, initial_parent=parent)
        if not created:
            await service.artifacts.discard(child_root)
        return stored, created
    except (GatewayError, StoreConflict):
        await service.artifacts.discard(child_root or root)
        raise
    except Exception:
        # An interrupted etcd reply may have committed both records. Never
        # destroy inputs on an unknown admission; managed orphan GC owns them.
        raise

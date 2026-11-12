# SPDX-License-Identifier: Apache-2.0
"""SR-only handler: shared source video replaces the H3 generation stage."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

from dingo.common.video_sr_only import SOURCE_KEY, validate_sr_only
from dingo.vllm.omni.h3_shared_inputs import _open_beneath
from dingo.vllm.omni.seedvr2_handler import SeedVR2Handler
from dingo.vllm.omni.seedvr2_execution import read_journal, UnsafeRecovery
from dingo.vllm.omni import seedvr2_media as media


def stage_source(attempt_root, descriptor, storage_id):
    if set(descriptor) != {'schema_version','storage_id','manifest_sha256','media','entry'} or type(descriptor['schema_version']) is not int or descriptor['schema_version'] != 1 or descriptor['storage_id'] != storage_id:
        raise ValueError('invalid SR source storage/schema')
    entry = descriptor['entry']
    keys = ('ordinal','path','content_type','bytes','sha256')
    if not isinstance(entry, dict) or set(entry) != set(keys):
        raise ValueError('invalid SR source entry')
    relative = entry['path']
    if not isinstance(relative, str) or len(relative) > 512 or '\\' in relative:
        raise ValueError('invalid source path')
    parts = relative.split('/')
    if len(parts) != 2 or parts[0] != 'inputs' or parts[1] in ('', '.', '..'):
        raise ValueError('source path must stay in task inputs')
    if entry['content_type'] != 'video/mp4' or type(entry['bytes']) is not int or not 0 < entry['bytes'] <= 128 * 1024**2:
        raise ValueError('invalid source media/size')
    task_root = Path(attempt_root).parent.parent
    if Path(attempt_root).parent.name != 'attempts':
        raise ValueError('invalid task attempt directory')
    root_fd = os.open(task_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    destination = Path(attempt_root) / 'source.mp4'
    try:
        with os.fdopen(_open_beneath(root_fd, ['input-manifest.json']), 'rb') as f:
            raw = f.read(256 * 1024 + 1)
        if len(raw) > 256 * 1024 or hashlib.sha256(raw).hexdigest() != descriptor['manifest_sha256']:
            raise ValueError('SR source manifest mismatch')
        manifest = json.loads(raw)
        if not isinstance(manifest, list) or len(manifest) != 1 or {k:manifest[0][k] for k in keys} != entry:
            raise ValueError('SR source differs from persisted upload')
        digest, size = hashlib.sha256(), 0
        with os.fdopen(_open_beneath(root_fd, parts), 'rb') as src, destination.open('xb') as dst:
            if os.fstat(src.fileno()).st_size != entry['bytes']:
                raise ValueError('SR source size mismatch')
            while data := src.read(1024 * 1024):
                size += len(data)
                if size > entry['bytes']:
                    raise ValueError('SR source grew during read')
                digest.update(data);dst.write(data)
            dst.flush();os.fsync(dst.fileno())
        if size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError('SR source checksum mismatch')
        return destination
    finally:
        os.close(root_fd)


class SROnlyHandler(SeedVR2Handler):
    def __init__(self, composite, *, storage_id):
        super().__init__(None, composite.execution, engine_identity=composite.engine_identity,
                         healthy=composite.healthy, profile=composite.sr_profile,
                         measurement_identity=composite.measurement_identity)
        # The transient composite has no base model and is not used to execute.
        self.execution.quarantine = self.quarantine
        self.storage_id = storage_id

    def request_plan(self, payload):
        return validate_sr_only(payload)

    async def _base(self, payload, root, fingerprint, context):
        started = time.monotonic()
        for journal in root.parent.glob('*/sr-journal.json'):
            if journal.parent == root:
                continue
            if journal.parent.is_symlink():
                raise UnsafeRecovery('symlink in SR attempt directory')
            previous = read_journal(journal)
            if previous and previous.get('phase') not in {'completed','failed','cancelled'}:
                raise UnsafeRecovery('previous SR attempt has unresolved GPU work')
        source = await asyncio.to_thread(stage_source, root, payload[SOURCE_KEY], self.storage_id)
        info = await media.probe(source)
        expected = payload[SOURCE_KEY]['media']
        if not info['copy_compatible'] or any(info[k] != expected[k] for k in ('width','height','frames')) or bool(info['audio']) != expected['audio']:
            raise ValueError('SR source metadata changed or is unsupported')
        if info['audio'] and info['audio']['codec'] != 'aac':
            raise ValueError('SR source audio must be AAC')
        if context.is_stopped():
            raise asyncio.CancelledError
        return source, {'sr_source_load_s':time.monotonic()-started}, None

"""Compiler layout atop existing safe filesystem and atomic JSON operations."""
import hashlib
import json
import os
import re
from pathlib import Path
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.file_io import run_file_io


class ContextIRArtifacts(FileArtifactStore):
    def __init__(self, root, deployment_id, result_max_bytes):
        super().__init__(Path(root) / "_context_ir" / deployment_id)
        self.deployment_id = deployment_id
        self.result_max_bytes = result_max_bytes

    def task_root(self, deployment_id, pool_id, task_id):
        if not re.fullmatch(r"context-ir-[0-9A-HJKMNP-TV-Z]{26}", task_id):
            raise ValueError("invalid ContextIR task ID")
        return self._symlink_free(self.root / "tasks" / task_id)

    def path(self, relative):
        if Path(relative).is_absolute():
            raise ValueError("ContextIR paths must be relative")
        return self._symlink_free(self.root / relative)

    def relative(self, path):
        return str(self._symlink_free(path).relative_to(self.root))

    async def write_json(self, path, value):
        self._symlink_free(path)
        await super().write_json(path, value)
        def sync_parent():
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        await run_file_io(sync_parent)

    async def read_bounded(self, relative, limit, digest=None):
        def read():
            path = self.path(relative)
            with path.open("rb") as stream:
                raw = stream.read(limit + 1)
            if len(raw) > limit:
                raise ValueError("ContextIR artifact exceeds size limit")
            if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("ContextIR artifact checksum mismatch")
            return json.loads(raw)
        return await run_file_io(read)

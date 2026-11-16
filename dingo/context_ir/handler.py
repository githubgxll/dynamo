# SPDX-License-Identifier: Apache-2.0
"""ContextIR semantic adapter; runtime scheduling remains compiler-independent."""
import asyncio
import hashlib
import logging
import threading
import time
from pathlib import Path

from dingo.common.task_protocol import relative_path, resolve_artifact
from dingo.video_gateway.context_ir.protocol import prepare, validate_result
from dingo.video_gateway.context_ir.validation import request_digest
from dingo.worker.runtime_io import atomic_json, read_json


class CallJournal:
    def __init__(self, root, cancelled, deadline_at_ms):
        self.root, self.cancelled, self.deadline = Path(root), cancelled, deadline_at_ms
        self.lock, self.sequence = threading.Lock(), 0

    def check_cancelled(self):
        from h3ir.engine import CompilationCancelled

        if (self.root.parent / "cancel.json").exists():
            self.cancelled.set()
        if self.cancelled.is_set() or time.time() * 1000 >= self.deadline:
            raise CompilationCancelled("compiler cancelled or deadline exceeded")

    def call_started(self, metadata=None):
        self.check_cancelled()
        with self.lock:
            self.sequence += 1
            call_id = f"{self.sequence:06d}"
            atomic_json(self.root / "intents" / f"{call_id}.json", {"call_id": call_id, "input": metadata or {}})
            return call_id

    def call_reported(self, call_id, usage):
        atomic_json(
            self.root / "receipts" / f"{call_id}.json",
            {"call_id": call_id, "usage": usage},
        )

    def stage_read(self, name, fingerprint):
        path = self.root / "stages" / (name + ".json")
        if not path.exists():
            return None
        value = read_json(path)
        return value["result"] if value.get("input_sha256") == fingerprint else None

    def stage_write(self, name, fingerprint, result):
        atomic_json(self.root / "stages" / (name + ".json"),
                    {"input_sha256": fingerprint, "result": result})

    def snapshot(self):
        from h3ir.usage import Usage

        aggregate = Usage()
        for intent in sorted((self.root / "intents").glob("*.json")):
            receipt = self.root / "receipts" / intent.name
            aggregate.add(read_json(receipt).get("usage") if receipt.exists() else None)
        return aggregate.snapshot()


class ContextIRHandler:
    def __init__(self, artifact_root, *, storage_id, prepare_parallelism=2):
        self.root = Path(artifact_root).resolve()
        self.storage_id = storage_id
        self.prepare_slots = asyncio.Semaphore(prepare_parallelism)

    async def execute(self, request, attempt_root, cancelled):
        descriptor = request.payload
        if descriptor.storage_id != self.storage_id:
            raise ValueError("shared storage identity mismatch")
        async with self.prepare_slots:
            payload, sources, expected, normalized = await asyncio.to_thread(
                self._prepare, descriptor, attempt_root
            )
        journal = CallJournal(attempt_root / "usage", cancelled, request.deadline_at_ms)
        from h3ir.engine import CompilationCancelled, compile_request

        start = time.monotonic()
        try:
            compiled = await asyncio.to_thread(
                compile_request,
                payload,
                sources=sources,
                hooks=journal,
                lease_token="worker-"
                + hashlib.sha256(str(attempt_root).encode()).hexdigest(),
            )
            result = validate_result(
                normalized, expected, compiled.status_code, compiled.body
            )
            return {
                "result": result,
                "usage": journal.snapshot(),
                "metrics": {"compile_time_s": time.monotonic() - start,
                            **({"intent_resolution_s": compiled.body["ir"]["provenance"]["intent_resolution"]["seconds"]}
                               if normalized.get("input_resolution") else {})},
            }
        except CompilationCancelled:
            return {"cancelled": True, "usage": journal.snapshot()}
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "compiler failed type=%s", type(exc).__name__, exc_info=True
            )
            # Qwen transport timeout may leave a request executing. The manager
            # holds admission shut until its dedicated-engine idle proof succeeds.
            return {
                "failed": True,
                "error_code": "context_ir_compilation_failed",
                "usage": journal.snapshot(),
                "settled": False,
                "diagnostic_type": type(exc).__name__,
            }

    def _prepare(self, descriptor, attempt_root):
        request_path = resolve_artifact(self.root, descriptor.request_path)
        manifest_path = resolve_artifact(self.root, descriptor.manifest_path)
        route_path = resolve_artifact(self.root, descriptor.route_path)
        for path, digest in (
            (manifest_path, descriptor.manifest_sha256),
            (route_path, descriptor.route_sha256),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("compiler descriptor hash mismatch")
        normalized = read_json(request_path, 512 * 1024)
        if request_digest(normalized) != descriptor.request_digest:
            raise ValueError("compiler request digest mismatch")
        if read_json(manifest_path) != normalized["references"]:
            raise ValueError("compiler manifest does not match request")
        for entry in normalized["references"]:
            relative_path(entry["path"])
            resolve_artifact(request_path.parent, entry["path"])
        payload, sources, expected = prepare(
            normalized, request_path.parent, attempt_root, 64 * 1024**2
        )
        # prepare already hashes each source and enforces media wiring/role order.
        return payload, sources, expected, normalized

    def recover_usage(self, root):
        return CallJournal(root / "usage", threading.Event(), 0).snapshot()

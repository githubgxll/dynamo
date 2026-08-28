# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application service joining HTTP input, persistent state and dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dingo.video_gateway.adapters.base import UploadedArtifact, VideoBackendAdapter
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.config import GatewayConfig, PoolConfig
from dingo.video_gateway.dispatcher import VideoDispatcher
from dingo.video_gateway.errors import GatewayError, StoreConflict
from dingo.video_gateway.ids import new_video_id
from dingo.video_gateway.models import StoredTask, TaskStatus, VideoTask, now_ms
from dingo.video_gateway.task_store import TaskStore


@dataclass(frozen=True, slots=True)
class Submission:
    stored: StoredTask
    created: bool


class VideoGatewayService:
    def __init__(
        self,
        config: GatewayConfig,
        store: TaskStore,
        artifacts: FileArtifactStore,
        dispatcher: VideoDispatcher,
        adapters: Mapping[str, VideoBackendAdapter],
    ) -> None:
        self.config = config
        self.store = store
        self.artifacts = artifacts
        self.dispatcher = dispatcher
        self.adapters = adapters
        self.upstream_models: list[dict] = []

    def resolve_pool(self, model: str) -> PoolConfig:
        pool = self.config.pool_for_model(model)
        if pool is None:
            raise GatewayError(
                404,
                "model_not_found",
                f"video model {model!r} is not configured",
                "model",
            )
        return pool

    async def ensure_submission_capacity(self, required_bytes: int) -> None:
        watermarks = self.config.artifact_store
        capacity = await self.artifacts.capacity()
        remaining = capacity.free_bytes - required_bytes
        if (
            remaining < watermarks.hard_min_free_bytes
            or (
                watermarks.soft_min_free_bytes > 0
                and remaining < watermarks.soft_min_free_bytes
            )
        ):
            await self.dispatcher.sweep_now()
            capacity = await self.artifacts.capacity()
            remaining = capacity.free_bytes - required_bytes
        if remaining < watermarks.hard_min_free_bytes:
            raise GatewayError(
                507,
                "insufficient_artifact_storage",
                "video artifact storage has insufficient free capacity",
                error_type="server_error",
            )

    def resolve_request_model(self, fields: Mapping[str, list[str]]) -> str:
        values = fields.get("model", [])
        if len(values) > 1:
            raise GatewayError(
                400, "duplicate_field", "model may appear only once", "model"
            )
        if values and values[0].strip():
            return values[0].strip()
        if self.config.http.default_model is not None:
            return self.config.http.default_model
        if len(self.config.pools_by_model) == 1:
            return next(iter(self.config.pools_by_model))
        raise GatewayError(
            400,
            "missing_required_field",
            "model is required when multiple video models are configured",
            "model",
        )

    async def submit(
        self,
        *,
        fields: Mapping[str, list[str]],
        uploads: Sequence[UploadedArtifact],
        upload_root: Path,
        delivery_mode: str,
        idempotency_key: str | None,
        principal: str = "validation",
    ) -> Submission:
        model = self.resolve_request_model(fields)
        pool = self.resolve_pool(model)
        if idempotency_key is not None:
            if not idempotency_key or len(idempotency_key.encode()) > 256:
                await self.artifacts.discard(upload_root)
                raise GatewayError(
                    400,
                    "invalid_idempotency_key",
                    "Idempotency-Key must be between 1 and 256 bytes",
                )
            idempotency_hash = hashlib.sha256(idempotency_key.encode()).hexdigest()
        else:
            idempotency_hash = None
        principal_hash = hashlib.sha256(principal.encode()).hexdigest()

        adapter = self.adapters[pool.pool_id]
        normalized_fields = {key: list(values) for key, values in fields.items()}
        normalized_fields["model"] = [model]
        try:
            normalized = await asyncio.to_thread(
                adapter.normalize_request, normalized_fields, uploads, model
            )
        except Exception:
            await self.artifacts.discard(upload_root)
            raise

        metadata = {
            int(item["ordinal"]): item
            for item in normalized.get("reference_metadata", [])
        }
        manifest = []
        for upload in uploads:
            entry = upload.manifest_entry(relative_to=upload_root)
            entry.update(metadata.get(upload.ordinal, {}))
            manifest.append(entry)
        encoded_reference_bytes = adapter.estimate_encoded_reference_bytes(manifest)
        encoded_reference_limit = min(
            self.config.media.max_encoded_reference_bytes,
            adapter.max_encoded_reference_bytes,
        )
        if encoded_reference_bytes > encoded_reference_limit:
            await self.artifacts.discard(upload_root)
            raise GatewayError(
                413,
                "encoded_references_too_large",
                "encoded Worker reference payload exceeds the configured limit",
                "input_references",
            )
        self.dispatcher.record_legacy_input(encoded_reference_bytes)
        # Reserve two encoded-input copies for mixed-reference JSON assembly.
        estimated_payload_bytes = (
            2 * encoded_reference_bytes
            + self.config.media.max_result_encoded_bytes
            + self.config.media.task_memory_overhead_bytes
        )
        digest_request = dict(normalized)
        if digest_request.pop("seed_generated", False):
            digest_request.pop("seed", None)
        digest_payload = {
            "model": model,
            "pool_id": pool.pool_id,
            "configuration_revision": pool.configuration_revision,
            "normalized_request": digest_request,
            "references": [
                {
                    "ordinal": entry["ordinal"],
                    "bytes": entry["bytes"],
                    "sha256": entry["sha256"],
                    "content_type": entry["content_type"],
                }
                for entry in manifest
            ],
            "compatibility_version": pool.adapter.compatibility_version,
            "gateway_protocol_version": 1,
        }
        request_digest = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    digest_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
        )

        try:
            if idempotency_hash is not None:
                existing = await self.store.get_idempotent(
                    principal_hash, idempotency_hash
                )
                if existing is not None:
                    if existing.task.request_digest != request_digest:
                        raise GatewayError(
                            409,
                            "idempotency_conflict",
                            "Idempotency-Key was already used with another request",
                        )
                    await self.artifacts.discard(upload_root)
                    return Submission(stored=existing, created=False)
            if (
                not pool.scheduling.accept_without_workers
                and not self.dispatcher.has_workers(pool.pool_id)
            ):
                raise GatewayError(
                    503,
                    "no_worker_available",
                    f"no Worker is registered for model {model!r}",
                    "model",
                    error_type="service_unavailable_error",
                )
            if (
                await self.store.queue_depth(pool.pool_id)
                >= pool.scheduling.queue_limit
            ):
                raise GatewayError(429, "queue_full", "video queue is full")
        except Exception:
            if upload_root.exists():
                await self.artifacts.discard(upload_root)
            raise

        task_root: Path | None = None
        try:
            task_id = new_video_id()
            created_at = now_ms()
            expires_at_ms = created_at + int(
                self.config.lifecycle.queue_ttl_s * 1000
            )
            task_root = await self.artifacts.commit_upload(
                upload_root,
                self.config.deployment_id,
                pool.pool_id,
                task_id,
                artifact_manifest={
                    "schema_version": 1,
                    "task_id": task_id,
                    "deployment_id": self.config.deployment_id,
                    "pool_id": pool.pool_id,
                    "created_at_ms": created_at,
                    "expires_at_ms": expires_at_ms,
                },
            )
            request_path = task_root / "request.json"
            manifest_path = task_root / "input-manifest.json"
            await self.artifacts.write_json(request_path, normalized)
            await self.artifacts.write_json(manifest_path, manifest)
            task = VideoTask(
                schema_version=1,
                id=task_id,
                deployment_id=self.config.deployment_id,
                pool_id=pool.pool_id,
                model=model,
                backend_model=pool.backend_model,
                backend_target=pool.backend_target,
                configuration_revision=pool.configuration_revision,
                delivery_mode=delivery_mode,
                status=TaskStatus.QUEUED,
                request_digest=request_digest,
                request_path=str(request_path),
                input_manifest_path=str(manifest_path),
                created_at_ms=created_at,
                queued_at_ms=created_at,
                expires_at_ms=expires_at_ms,
                principal_hash=principal_hash,
                idempotency_hash=idempotency_hash,
                estimated_payload_bytes=estimated_payload_bytes,
                normalized_request=normalized,
            )
            stored, created = await self.store.create_task(
                task,
                principal_hash=principal_hash,
                idempotency_hash=idempotency_hash,
                queue_limit=pool.scheduling.queue_limit,
            )
        except Exception:
            await self.artifacts.discard(task_root or upload_root)
            raise
        if not created and task_root is not None:
            await self.artifacts.discard(task_root)
        else:
            self.dispatcher.notify(pool.pool_id)
        return Submission(stored=stored, created=created)

    async def expire(self, stored: StoredTask) -> StoredTask:
        for _ in range(8):
            if stored.task.status not in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.EXPIRED,
            }:
                return await self.dispatcher.cancel(stored.task.id)
            try:
                return await self.dispatcher.expire_terminal(stored)
            except StoreConflict:
                latest = await self.store.get_task(stored.task.id)
                if latest is None:
                    raise KeyError(stored.task.id) from None
                stored = latest
        raise StoreConflict("delete lost repeated task state races")

"""Ref-only, eight-rank identity and idle-cache RPCs; no monitoring imports."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import time

from . import legacy_idle_cache as legacy
from .state import identifier

IDENTITY_SCHEMA = "h3sr-worker-identities/1"
SCHEMA = "h3sr-idle-cache-release/2"
IDENTITY_METHOD = "h3sr_worker_identities"
METHOD = "h3sr_release_idle_cache_v2"
REASONS = {"after-sr-load-before-first-h3", "after-sr-before-next-h3", "before-consecutive-h3"}
SHA = re.compile(r"[0-9a-f]{64}\Z")
validate_legacy_report = legacy.validate_report


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def identity_fields(row, epoch, extension_sha256):
    value = legacy._identity_fields(row)
    if (row.get("world_size") != 8 or row.get("service_epoch") != epoch
            or row.get("extension_sha256") != extension_sha256):
        raise RuntimeError("SR epoch/world/extension source identity differs")
    return {**value, "world_size": 8, "service_epoch": epoch, "extension_sha256": extension_sha256}


def validate_identities(rows, epoch, extension_sha256):
    identifier(epoch)
    if not isinstance(extension_sha256, str) or not SHA.fullmatch(extension_sha256):
        raise ValueError("Require exact extension source SHA256")
    if not isinstance(rows, list) or len(rows) != 8:
        raise RuntimeError("Exactly eight SR worker identities required")
    normalized = [identity_fields(row, epoch, extension_sha256) for row in rows]
    legacy.validate_expected_identities(normalized, 8)
    return {row["rank"]: row for row in normalized}


def validate_identity_report(report, epoch, extension_sha256):
    if (not isinstance(report, dict) or report.get("schema") != IDENTITY_SCHEMA
            or report.get("service_epoch") != epoch or report.get("extension_sha256") != extension_sha256):
        raise RuntimeError("SR identity report differs from service epoch/source")
    rows = report.get("ranks")
    if not isinstance(rows, list) or any(not isinstance(r, dict) or r.get("ok") is not True for r in rows):
        raise RuntimeError("One or more SR worker identity RPCs failed")
    validate_identities(rows, epoch, extension_sha256)
    return report


def _local_identity(worker, torch, epoch, expected_sha):
    rank = int(torch.distributed.get_rank())
    row = {"rank": rank, "ok": False, "pid": os.getpid()}
    try:
        identifier(epoch)
        if expected_sha != source_sha256():
            raise RuntimeError("Worker extension source SHA differs from controller")
        if torch.distributed.get_world_size() != 8:
            raise RuntimeError("SR world size must be eight")
        if type(worker.model_runner.pipeline).__name__ != "SeedVR2Pipeline":
            raise RuntimeError("Ref cache control only accepts SeedVR2 workers")
        device = worker.device
        if getattr(device, "type", None) != "cuda":
            raise RuntimeError("CUDA worker required")
        with torch.cuda.device(device):
            if torch.cuda.current_device() != device.index:
                raise RuntimeError("Worker current CUDA device differs")
            row.update(local_cuda_index=device.index,
                gpu_uuid=legacy._normalize_uuid(torch.cuda.get_device_properties(device).uuid),
                **legacy._birth_identity(), world_size=8, service_epoch=epoch,
                extension_sha256=source_sha256())
        identity_fields(row, epoch, expected_sha)
        row["ok"] = True
    except BaseException as error:
        row.update(error_type=type(error).__name__, error=str(error))
    return row


def _gather(torch, local, group):
    rows = [None] * torch.distributed.get_world_size()
    # Local validation errors are data, not an early exit from this collective.
    torch.distributed.all_gather_object(rows, local, group=group)
    return rows


def validate_report(report, handoff_id, reason, expected_rank_identities, service_epoch,
                    extension_sha256, collect_memory=False, collect_timing=False):
    identifier(handoff_id)
    if reason not in REASONS or type(collect_memory) is not bool or type(collect_timing) is not bool:
        raise ValueError("Invalid handoff reason or optional metric controls")
    # Historical /1 receipts are intentionally handled only by the separate
    # validate_legacy_report reader, never by an online epoch-bound v2 call.
    expected = validate_identities(expected_rank_identities, service_epoch, extension_sha256)
    if (report.get("schema") != SCHEMA or report.get("handoff_id") != handoff_id
            or report.get("reason") != reason or report.get("service_epoch") != service_epoch
            or report.get("extension_sha256") != extension_sha256
            or report.get("metrics") != {"memory": collect_memory, "timing": collect_timing}):
        raise RuntimeError("Cache release receipt identity differs")
    rows = report.get("ranks")
    if not isinstance(rows, list) or len(rows) != 8:
        raise RuntimeError("All eight rank acknowledgements required")
    actual = validate_identities(rows, service_epoch, extension_sha256)
    if actual != expected:
        raise RuntimeError("Live worker identity changed since Ready")
    for row in rows:
        if row.get("ok") is not True or row.get("handoff_id") != handoff_id or row.get("reason") != reason:
            raise RuntimeError("One or more rank cache releases failed")
        if collect_memory:
            for name in ("before", "after"):
                stat = row.get(name)
                if (not isinstance(stat, dict) or any(type(stat.get(k)) is not int or stat[k] < 0 for k in
                    ("allocated_bytes", "reserved_bytes", "device_free_bytes", "device_total_bytes"))
                    or not 0 <= stat["allocated_bytes"] <= stat["reserved_bytes"] <= stat["device_total_bytes"]
                    or not 0 <= stat["device_free_bytes"] <= stat["device_total_bytes"] or stat["device_total_bytes"] <= 0):
                    raise RuntimeError("Invalid optional allocator receipt")
        elif any(name in row for name in ("before", "after", "reserved_released_bytes", "live_allocated_delta_bytes")):
            raise RuntimeError("Memory telemetry was returned while disabled")
        if collect_timing:
            value = row.get("empty_cache_wall_seconds")
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise RuntimeError("Invalid optional timing receipt")
        elif "empty_cache_wall_seconds" in row:
            raise RuntimeError("Timing was returned while disabled")
    return report


class IdleCacheWorkerExtension:
    def h3sr_worker_identities(self, *, service_epoch, extension_sha256):
        import torch
        from vllm_omni.diffusion.distributed.parallel_state import get_world_group
        if not torch.distributed.is_initialized():
            raise RuntimeError("SR world is not initialized")
        rows = _gather(torch, _local_identity(self, torch, service_epoch, extension_sha256), get_world_group().cpu_group)
        report = {"schema": IDENTITY_SCHEMA, "service_epoch": service_epoch,
                  "extension_sha256": extension_sha256, "ranks": rows}
        return validate_identity_report(report, service_epoch, extension_sha256)

    def h3sr_release_idle_cache_v2(self, *, handoff_id, reason, service_epoch, extension_sha256,
                                 expected_rank_identities, collect_memory=False, collect_timing=False):
        import torch
        from vllm_omni.diffusion.distributed.parallel_state import get_world_group
        if not torch.distributed.is_initialized():
            raise RuntimeError("SR world is not initialized")
        row = _local_identity(self, torch, service_epoch, extension_sha256)
        row.update(handoff_id=handoff_id, reason=reason)
        try:
            identifier(handoff_id)
            if reason not in REASONS or type(collect_memory) is not bool or type(collect_timing) is not bool:
                raise ValueError("Invalid cache release controls")
            expected = validate_identities(expected_rank_identities, service_epoch, extension_sha256)
            if row.get("ok") is not True or identity_fields(row, service_epoch, extension_sha256) != expected[row["rank"]]:
                raise RuntimeError("Live worker identity differs from Ready identity")
            row["ok"] = False
            with torch.cuda.device(self.device):
                if collect_memory:
                    row["before"] = legacy._cuda_snapshot(torch, self.device)
                begin = time.perf_counter() if collect_timing else None
                torch.cuda.empty_cache()
                if collect_timing:
                    row["empty_cache_wall_seconds"] = time.perf_counter() - begin
                if collect_memory:
                    row["after"] = legacy._cuda_snapshot(torch, self.device)
            row["ok"] = True
        except BaseException as error:
            row.update(ok=False, error_type=type(error).__name__, error=str(error))
        rows = _gather(torch, row, get_world_group().cpu_group)
        report = {"schema": SCHEMA, "handoff_id": handoff_id, "reason": reason, "service_epoch": service_epoch,
                  "extension_sha256": extension_sha256, "ranks": rows,
                  "metrics": {"memory": collect_memory, "timing": collect_timing}}
        return validate_report(report, handoff_id, reason, expected_rank_identities, service_epoch,
                               extension_sha256, collect_memory, collect_timing)

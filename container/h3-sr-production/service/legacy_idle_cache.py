"""Opt-in H100 / Omni 0.32 SR cache handoff prototype; no import side effects.

Mount this module read-only and select IdleCacheWorkerExtension in the SR
OmniDiffusionConfig. The external controller must serialize BOTH services.
No source patching, sleep, weight migration, retry or inference happens here.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import inspect
import os
from pathlib import Path
import re
import time

SCHEMA = "h3sr-idle-cache-release/1"
METHOD = "h3sr_release_idle_cache"
_HANDOFF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_REASONS = {"after-sr-load-before-first-h3", "after-sr-before-next-h3"}
_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_PIDNS = re.compile(r"pid:\[\d+\]\Z")


def _validate_contract(handoff_id, reason, expected_world_size):
    if not isinstance(handoff_id, str) or not _HANDOFF.fullmatch(handoff_id):
        raise ValueError("Invalid handoff_id")
    if reason not in _REASONS:
        raise ValueError("Unsupported handoff reason")
    if type(expected_world_size) is not int or expected_world_size != 8:
        raise ValueError("This prototype is frozen to SR world_size=8")


def _cuda_snapshot(torch, device):
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "device_free_bytes": int(free),
        "device_total_bytes": int(total),
    }


def _normalize_uuid(value):
    value = str(value).lower()
    if value.startswith("gpu-"):
        value = value[4:]
    value = "GPU-" + value
    if not _UUID.fullmatch(value):
        raise RuntimeError("CUDA device UUID missing or malformed")
    return value


def _birth_identity():
    fields = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    return {"process_start_ticks": int(fields[19]),
            "pid_namespace": os.readlink("/proc/self/ns/pid")}


def _identity_fields(row):
    if (not isinstance(row, dict) or type(row.get("rank")) is not int
            or type(row.get("local_cuda_index")) is not int
            or not 0 <= row["local_cuda_index"] < 8
            or type(row.get("pid")) is not int or row["pid"] <= 0
            or type(row.get("process_start_ticks")) is not int or row["process_start_ticks"] <= 0
            or not isinstance(row.get("pid_namespace"), str)
            or not _PIDNS.fullmatch(row["pid_namespace"])):
        raise RuntimeError("Incomplete or invalid SR rank/process identity")
    gpu_uuid = row.get("gpu_uuid")
    if not isinstance(gpu_uuid, str):
        raise RuntimeError("Expected CUDA GPU UUID string")
    result = {k: row[k] for k in ("rank", "gpu_uuid", "local_cuda_index", "pid",
                                 "process_start_ticks", "pid_namespace")}
    result["gpu_uuid"] = _normalize_uuid(gpu_uuid)
    return result


def validate_expected_identities(expected_rank_identities, expected_world_size=8):
    if not isinstance(expected_rank_identities, list) or len(expected_rank_identities) != expected_world_size:
        raise RuntimeError("Require frozen monitoring registration for every SR rank")
    rows = [_identity_fields(row) for row in expected_rank_identities]
    if sorted(r["rank"] for r in rows) != list(range(expected_world_size)):
        raise RuntimeError("Expected SR registration ranks must be distinct integers 0..7")
    if len({r["gpu_uuid"] for r in rows}) != expected_world_size:
        raise RuntimeError("Expected SR registration must identify eight distinct GPUs")
    if len({r["local_cuda_index"] for r in rows}) != expected_world_size:
        raise RuntimeError("Expected SR registration must identify eight distinct local CUDA devices")
    if len({(r["pid_namespace"], r["pid"], r["process_start_ticks"]) for r in rows}) != expected_world_size:
        raise RuntimeError("Expected SR registration must identify eight distinct worker births")
    return {r["rank"]: r for r in rows}


def _run_local_release(worker, torch, handoff_id, reason, expected_world_size, expected_rank_identities):
    """Return a status even on local error so healthy ranks can reach gather."""
    rank = int(torch.distributed.get_rank())
    status = {"rank": rank, "pid": os.getpid(), "ok": False,
              "handoff_id": handoff_id, "reason": reason}
    try:
        _validate_contract(handoff_id, reason, expected_world_size)
        expected = validate_expected_identities(expected_rank_identities, expected_world_size)
        if torch.distributed.get_world_size() != expected_world_size:
            raise RuntimeError("Unexpected world size")
        pipeline = worker.model_runner.pipeline
        if type(pipeline).__name__ != "SeedVR2Pipeline":
            raise RuntimeError("This operation is restricted to SeedVR2Pipeline")
        device = worker.device
        if getattr(device, "type", None) != "cuda":
            raise RuntimeError("Only the frozen CUDA H100 path is supported")
        # Bind to the worker's ALREADY selected device; never use CUDA 0 implicitly.
        with torch.cuda.device(device):
            if torch.cuda.current_device() != device.index:
                raise RuntimeError("Current worker CUDA device mismatch")
            status.update(local_cuda_index=device.index,
                          gpu_uuid=_normalize_uuid(torch.cuda.get_device_properties(device).uuid),
                          **_birth_identity())
            if _identity_fields(status) != expected[rank]:
                raise RuntimeError("Worker identity differs from frozen monitoring registration")
            before = _cuda_snapshot(torch, device)
            begin = time.perf_counter()
            torch.cuda.empty_cache()
            elapsed = time.perf_counter() - begin
            after = _cuda_snapshot(torch, device)
        status.update(ok=True, before=before, after=after,
                      empty_cache_wall_seconds=elapsed,
                      reserved_released_bytes=max(0, before["reserved_bytes"] - after["reserved_bytes"]),
                      live_allocated_delta_bytes=after["allocated_bytes"] - before["allocated_bytes"])
    except BaseException as exc:
        # Catch local cancellation/failure until gather so other healthy ranks
        # do not wait forever simply because this rank skipped the collective.
        status.update(error_type=type(exc).__name__, error=str(exc))
    return status


class IdleCacheWorkerExtension:
    """SR-only worker mixin selected by config; no override of model methods."""

    def h3sr_release_idle_cache(self, *, handoff_id, reason, expected_rank_identities, expected_world_size=8):
        import torch
        from vllm_omni.diffusion.distributed.parallel_state import get_world_group

        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed SR worker group is not initialized")
        local = _run_local_release(self, torch, handoff_id, reason, expected_world_size, expected_rank_identities)
        rows = [None] * torch.distributed.get_world_size()
        # Detailed results must be gathered explicitly: Omni's outer RPC status
        # envelope is validated then stripped and retains only rank0's result.
        torch.distributed.all_gather_object(rows, local, group=get_world_group().cpu_group)
        report = {"schema": SCHEMA, "handoff_id": handoff_id, "reason": reason,
                  "expected_world_size": expected_world_size, "ranks": rows}
        validate_report(report, handoff_id, reason, expected_rank_identities, expected_world_size)
        return report


def _one_result(value):
    # AsyncOmni -> stage -> executor each may add a singleton list.
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        raise RuntimeError("Expected exactly one SR replica result")
    return value


def validate_report(report, handoff_id, reason, expected_rank_identities, expected_world_size=8):
    _validate_contract(handoff_id, reason, expected_world_size)
    expected = validate_expected_identities(expected_rank_identities, expected_world_size)
    if (report.get("schema") != SCHEMA or report.get("handoff_id") != handoff_id
            or report.get("reason") != reason
            or report.get("expected_world_size") != expected_world_size):
        raise RuntimeError("SR release identity does not match handoff")
    rows = report.get("ranks")
    if not isinstance(rows, list) or len(rows) != expected_world_size:
        raise RuntimeError("Missing SR rank acknowledgements")
    if any(not isinstance(r, dict) for r in rows):
        raise RuntimeError("Malformed SR rank status")
    if any(type(r.get("rank")) is not int for r in rows):
        raise RuntimeError("SR rank must be an integer, not bool")
    if sorted(r["rank"] for r in rows) != list(range(expected_world_size)):
        raise RuntimeError("Duplicate or missing SR rank")
    seen_uuids = set()
    for row in rows:
        if (row.get("ok") is not True or row.get("handoff_id") != handoff_id
                or row.get("reason") != reason):
            raise RuntimeError("SR rank release failed: " + str(row))
        identity = _identity_fields(row)
        if identity != expected[row["rank"]]:
            raise RuntimeError("SR worker identity does not match frozen registration")
        seen_uuids.add(identity["gpu_uuid"])
        for key in ("before", "after"):
            stat = row.get(key)
            if (not isinstance(stat, dict)
                    or any(type(stat.get(k)) is not int or stat[k] < 0 for k in
                           ("allocated_bytes", "reserved_bytes", "device_free_bytes", "device_total_bytes"))):
                raise RuntimeError("Missing SR rank memory statistics")
            if (stat["device_total_bytes"] <= 0
                    or stat["device_free_bytes"] > stat["device_total_bytes"]
                    or not stat["allocated_bytes"] <= stat["reserved_bytes"] <= stat["device_total_bytes"]):
                raise RuntimeError("Inconsistent SR rank memory statistics")
    if len(seen_uuids) != expected_world_size:
        raise RuntimeError("SR acknowledgements do not cover eight distinct GPUs")
    return report


class FailureLatch:
    """One shared irreversible gate; every future ingress MUST assert_open().

    persist_blocked must be synchronous and atomically write the shared blocked
    state. Memory is latched BEFORE persistence; a disk error never reopens it.
    """

    def __init__(self, persist_blocked):
        if not callable(persist_blocked) or inspect.iscoroutinefunction(persist_blocked):
            raise TypeError("persist_blocked must be a synchronous callable")
        self._persist = persist_blocked
        self.blocked = None

    def assert_open(self):
        if self.blocked is not None:
            raise RuntimeError("Service handoff permanently blocked: " + str(self.blocked))

    def trip(self, record):
        if self.blocked is not None:
            return
        self.blocked = dict(record)  # shared in-process state first, under lock
        result = self._persist(dict(self.blocked))
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError("persist_blocked returned an awaitable")


async def _assert_idle(check):
    result = check()
    if inspect.isawaitable(result):
        result = await result
    if result is not True:
        raise RuntimeError("Service idle state not proven; H3 handoff blocked")


@asynccontextmanager
async def sr_idle_cache_handoff(engine_client, *, handoff_lock, assert_idle, failure_latch,
                                handoff_id, reason, stage_ids, expected_rank_identities, timeout=60):
    """Keep BOTH services locked through release AND the caller's H3 request.

    `handoff_lock` must be the same cross-service lock used by every inference
    ingress. assert_idle must cover SR active_task/window work and H3 work.
    Every BaseException (including timeout/cancellation) trips the shared latch
    and synchronously persists it INSIDE the lock before releasing the lock.
    """
    _validate_contract(handoff_id, reason, 8)
    if not isinstance(stage_ids, list) or len(stage_ids) != 1 or type(stage_ids[0]) is not int:
        raise ValueError("Select exactly one SR stage")
    validate_expected_identities(expected_rank_identities, 8)
    if not isinstance(failure_latch, FailureLatch):
        raise TypeError("Require the shared FailureLatch used by all service ingress")
    async with handoff_lock:
        try:
            failure_latch.assert_open()
            await _assert_idle(assert_idle)
            # Unlike a new extension method, this standard RPC also drains queued
            # async D2H/SHM outputs in the multiprocess worker before device sync.
            synced = await engine_client.collective_rpc(
                "synchronize_device", timeout=timeout, kwargs={"timeout": timeout}, stage_ids=stage_ids)
            value = synced
            while isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            if value is not None:
                raise RuntimeError("SR synchronize_device did not return a successful void result")
            await _assert_idle(assert_idle)
            raw = await engine_client.collective_rpc(
                METHOD, timeout=timeout, kwargs={"handoff_id": handoff_id, "reason": reason,
                    "expected_world_size": 8, "expected_rank_identities": expected_rank_identities}, stage_ids=stage_ids)
            report = validate_report(_one_result(raw), handoff_id, reason, expected_rank_identities, 8)
            yield report
        except BaseException as exc:
            try:
                failure_latch.trip({"handoff_id": handoff_id, "reason": reason,
                                    "error_type": type(exc).__name__, "generation_blocked": True})
            except BaseException as persist_error:
                raise RuntimeError("Handoff blocked in memory; blocked-state persistence failed") from persist_error
            raise

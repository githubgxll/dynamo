# SPDX-License-Identifier: Apache-2.0
"""Opt-in SeedVR2 observations; never mutate tensors, RNG or model settings.

CUDA times cover the current stream only.  Basic mode waits at the completed
window boundary, never at individual stages.  NVML is deliberately external.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time

SCHEMA = "sr-timing-memory/2"
KEY = "seedvr2_measurement"
LOG = logging.getLogger(__name__)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
IDENTITIES = ("run_id", "batch_id", "case_id", "action_id", "phase", "image_digest", "source_hashes", "model_manifest_hash")
WINDOW_FIELDS = ("window_index", "window_start", "input_frame_count", "padded_frame_count", "new_frame_count", "overlap_frames", "configured_window", "selected_window", "model_width", "model_height", "delivery_width", "delivery_height")
_TRANSFER_SCOPE = ContextVar("seedvr2_transfer_scope", default=None)
_ACTIVE_TRANSFER = ContextVar("seedvr2_active_transfer", default=None)


def validate_context(value):
    if not isinstance(value, dict) or value.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("Invalid SeedVR2 measurement schema")
    if value.get("measurement_mode") not in ("off", "basic", "diagnostic"):
        raise ValueError("Invalid SeedVR2 measurement mode")
    for field in ("sr_request_id", "model_request_id"):
        if not isinstance(value.get(field), str) or not _ID.fullmatch(value[field]):
            raise ValueError("Unsafe SeedVR2 measurement " + field)
    for field in WINDOW_FIELDS:
        item = value.get(field)
        if type(item) is not int or item < (0 if field in ("window_index", "window_start", "overlap_frames") else 1):
            raise ValueError("Invalid SeedVR2 measurement " + field)
    if value["input_frame_count"] > value["padded_frame_count"] or value["padded_frame_count"] % 4 != 1:
        raise ValueError("Invalid SeedVR2 measurement frame padding")
    if value["new_frame_count"] > value["input_frame_count"]:
        raise ValueError("Invalid SeedVR2 measurement new frame count")
    # Copy only explicit contract fields; never propagate filesystem paths.
    result = {key: value[key] for key in ("sr_request_id", "model_request_id", "measurement_mode", *WINDOW_FIELDS)}
    for field in IDENTITIES:
        if field in value:
            result[field] = value[field]
    if len(json.dumps(result, allow_nan=False)) > 32768:
        raise ValueError("SeedVR2 measurement context too large")
    return {"schema": SCHEMA, **result}


def safe_path(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("SeedVR2 metrics root must be absolute")
    for item in (path, *path.parents):
        if item.is_symlink() or (item.exists() and getattr(item.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("SeedVR2 metrics symlink/reparse path refused")
    return path


def output_directory(context):
    root = os.environ.get("SEEDVR2_METRICS_DIR")
    if not root:
        raise ValueError("SEEDVR2_METRICS_DIR is not configured")
    root = safe_path(root)
    target = safe_path(root / context["sr_request_id"] / "windows" / str(context["window_index"]))
    target.mkdir(parents=True, exist_ok=True)
    safe_path(target)
    return target


def atomic_json(path, value):
    path = safe_path(path)
    safe_path(path.parent)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(data)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def unique_json(path, value):
    """Retain a duplicate as failed evidence instead of replacing a producer."""
    path = safe_path(path)
    safe_path(path.parent)
    lock_path = path.with_name("." + path.name + ".publish-lock")
    descriptor = None
    try:
        try:
            # A per-receipt exclusive reservation makes existence+atomic
            # publication non-replacing across PIDs. It requires neither
            # symlink/hardlink privileges nor a shared process lock.
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            descriptor = None
        if descriptor is None or path.exists():
            duplicate = {**value, "status": "METRICS_INCOMPLETE", "duplicate_of": path.name,
                "errors": [*value.get("errors", []), {"stage": "receipt_publish", "error_type": "DuplicateProducer", "error": "Repeated receipt producer for the same window"}]}
            unique_json(path.with_name(path.stem + "-duplicate-" + str(time.perf_counter_ns()) + ".json"), duplicate)
        else:
            atomic_json(path, value)
    finally:
        if descriptor is not None:
            os.close(descriptor)
            lock_path.unlink()


def _proc_identity():
    result = {"pid": os.getpid(), "process_start_time": None, "nspid": None}
    try:
        # stat comm can contain spaces and parentheses; fields after final ')'.
        values = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        result["process_start_time"] = {"clock_domain": "linux_boot_ticks", "value": int(values[19])}
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("NSpid:"):
                result["nspid"] = [int(item) for item in line.split()[1:]]
    except (OSError, ValueError, IndexError):
        result["process_identity_reason"] = "procfs_unavailable"
    return result


def _source_hashes():
    here = Path(__file__)
    result = {}
    for path in (here, here.with_name("pipeline_seedvr2.py"), here.parents[2] / "ipc.py"):
        try:
            result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            result[path.name] = None
    return result


class Disabled:
    enabled = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def stage(self, _name):
        return nullcontext()

    def metadata(self):
        return None


class Window:
    enabled = True

    def __init__(self, context, device, torch_module=None, *, execution_scope="sr_rank", filename=None, reset_allocator_peak=True):
        self.context = validate_context(context)
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.device = device
        self.cuda = self.torch.cuda
        self.execution_scope = execution_scope
        self.filename = filename
        self.reset_allocator_peak = reset_allocator_peak
        self.stages = []
        self.errors = []
        self.explicit_stage_waits = 0
        self.boundary_waits = 0
        self.started = None
        self.parent = None
        self.before = None
        self.rank = None
        self.world_size = None
        self.gpu_uuid = None
        self.active = True

    def _error(self, stage, error):
        self.errors.append({"stage": stage, "error_type": type(error).__name__, "error": str(error)[:1000]})

    def allocator(self):
        return {"allocated_bytes": int(self.cuda.memory_allocated(self.device)), "reserved_bytes": int(self.cuda.memory_reserved(self.device))}

    def event(self):
        event = self.cuda.Event(enable_timing=True)
        event.record(self.cuda.current_stream(self.device))
        return event

    def __enter__(self):
        self.started = time.perf_counter_ns()
        try:
            output_directory(self.context)
            distributed = self.torch.distributed
            if distributed.is_available() and distributed.is_initialized():
                self.rank, self.world_size = distributed.get_rank(), distributed.get_world_size()
            else:
                self.rank, self.world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
            props = self.cuda.get_device_properties(self.device)
            self.gpu_uuid = str(props.uuid) if getattr(props, "uuid", None) is not None else None
            self.before = self.allocator()
            if self.reset_allocator_peak:
                self.cuda.reset_peak_memory_stats(self.device)
            self.parent = self.event()
        except Exception as error:
            self._error("window_start", error)
            self.active = False
        return self

    @contextmanager
    def stage(self, name):
        if not self.active:
            yield
            return
        start_cpu = time.perf_counter_ns()
        start_event = None
        before = None
        try:
            before = self.allocator()
            start_event = self.event()
        except Exception as error:
            self._error(name + ":start", error)
        failure = None
        try:
            yield
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            end_event = None
            after = None
            try:
                end_event = self.event()
                if self.context["measurement_mode"] == "diagnostic" and failure is None:
                    end_event.synchronize()
                    self.explicit_stage_waits += 1
                after = self.allocator()
            except Exception as error:
                self._error(name + ":end", error)
            self.stages.append({"phase": name, "cpu_start_ns": start_cpu, "cpu_end_ns": time.perf_counter_ns(),
                "allocator_before": before, "allocator_after": after, "failure": failure,
                "_events": (start_event, end_event)})

    def metadata(self):
        return {**self.context, "actual_model_request_id": getattr(self, "actual_model_request_id", None)}

    def __exit__(self, exc_type, exc_value, _traceback):
        submitted = time.perf_counter_ns()
        after = peak = parent_elapsed = None
        wait_start = wait_end = None
        try:
            end_event = self.event() if self.active else None
            if end_event is not None and exc_type is None:
                # One current-stream window boundary wait, no global barrier.
                wait_start = time.perf_counter_ns()
                end_event.synchronize()
                wait_end = time.perf_counter_ns()
                self.boundary_waits += 1
                parent_elapsed = float(self.parent.elapsed_time(end_event))
            if self.active:
                after = self.allocator()
                peak = {"allocated_bytes": int(self.cuda.max_memory_allocated(self.device)), "reserved_bytes": int(self.cuda.max_memory_reserved(self.device))}
        except Exception as error:
            self._error("window_end", error)
        ended = time.perf_counter_ns()
        records = []
        for stage in self.stages:
            start, end = stage.pop("_events")
            elapsed = None
            if exc_type is None and start is not None and end is not None and self.boundary_waits:
                try:
                    elapsed = float(start.elapsed_time(end))
                except Exception as error:
                    self._error(stage["phase"] + ":resolve", error)
            records.append({**stage, "cpu_wall_ms": (stage["cpu_end_ns"] - stage["cpu_start_ns"]) / 1e6,
                "cuda_stream_elapsed_ms": elapsed, "event_scope": "current_cuda_stream",
                "includes_collectives": "only_when_ordered_on_recorded_stream",
                "execution_scope": self.execution_scope,
                "includes_posterior_sampling": stage["phase"] == "vae_encode"})
        if exc_type is not None:
            self._error("model", exc_value)
        try:
            identity = _proc_identity()
            record = {**self.context, **identity, "rank": self.rank, "world_size": self.world_size,
                "gpu_uuid": self.gpu_uuid, "gpu_uuid_reason": None if self.gpu_uuid else "not_reported_by_torch",
                "execution_scope": self.execution_scope, "actual_model_request_id": getattr(self, "actual_model_request_id", None),
                "video_payload_path": "payload.video",
                "clock_domain": "process_perf_counter_ns", "started_utc_ns": getattr(self, "utc_ns", None),
                "cpu_start_ns": self.started, "cpu_end_ns": ended, "pipeline_gpu_window_wall_ms": (ended - self.started) / 1e6,
                "cpu_submission_end_ns": submitted, "window_boundary_wait_ms": None if wait_end is None else (wait_end - wait_start) / 1e6,
                "pipeline_gpu_window_stream_elapsed_ms": parent_elapsed, "stages": records,
                "measurement_source": "torch_allocator", "allocator_before": self.before,
                "allocator_after": after, "allocator_window_peak": peak if self.reset_allocator_peak else None,
                "allocator_process_peak_observed": None if self.reset_allocator_peak else peak,
                "allocator_peak_scope": "window_reset" if self.reset_allocator_peak else "process_since_last_external_reset_not_d2h_peak",
                "allocator_phase_values": "boundary_snapshots_not_phase_peaks", "event_scope": "current_cuda_stream",
                "explicit_stage_waits": self.explicit_stage_waits, "explicit_window_boundary_waits": self.boundary_waits,
                "status": ("CANCELLED" if exc_type and exc_type.__name__ in ("CancelledError", "JobCancelledError", "DiffusionRequestAbortedError")
                    else "FAIL" if exc_type else "METRICS_INCOMPLETE" if self.errors else "PASS"), "errors": self.errors,
                "actual_source_hashes": _source_hashes(),
                "device_to_host": "worker_transport_on_sender_rank_only; other_ranks_not_executed_by_design" if self.execution_scope == "sr_rank" else "executed_here"}
            filename = self.filename or f"rank-{self.rank}.json"
            atomic_json(output_directory(self.context) / filename, record)
        except Exception as error:
            LOG.warning("SeedVR2 metrics incomplete request=%s window=%s error=%s", self.context["sr_request_id"], self.context["window_index"], type(error).__name__)
        return False


def window(request, device):
    extra = getattr(request.sampling_params, "extra_args", None) or {}
    value = extra.get(KEY)
    if not value or (isinstance(value, dict) and value.get("measurement_mode") == "off"):
        return Disabled()
    try:
        item = Window(value, device)
        item.actual_model_request_id = request.request_id
        item.utc_ns = time.time_ns()
        return item
    except Exception as error:
        LOG.warning("SeedVR2 metrics context invalid: %s", type(error).__name__)
        return Disabled()


def attach_metadata(metadata, observation):
    context = observation.metadata()
    if context is not None:
        metadata[KEY] = context


def post_process(output, video):
    metadata = output.get("metadata", {})
    context = metadata.get(KEY) if isinstance(metadata, dict) else None
    if not context or context.get("measurement_mode") == "off":
        return Disabled()
    try:
        return CpuPostprocess(context, video)
    except Exception as error:
        LOG.warning("SeedVR2 postprocess metrics context invalid: %s", type(error).__name__)
        return Disabled()


class CpuPostprocess:
    """Observe the API's already-CPU tensor conversion without touching CUDA."""
    enabled = True

    def __init__(self, context, video):
        self.context = validate_context(context)
        self.actual_model_request_id = context.get("actual_model_request_id")
        self.source_device_type = getattr(video.device, "type", None)
        self.source_device = str(video.device)
        self.stages = []
        self.errors = []
        if self.source_device_type != "cpu":
            self.errors.append({"stage": "postprocess_source", "error_type": "UnexpectedDevice",
                "error": "API CPU postprocess received a non-CPU tensor; D2H coverage is unknown"})

    def __enter__(self):
        self.started = time.perf_counter_ns()
        self.utc_ns = time.time_ns()
        return self

    @contextmanager
    def stage(self, _original_name):
        # The existing pipeline wrapper is retained byte-for-byte.  Its old
        # `device_to_host` label is corrected here to the actual CPU operation.
        started = time.perf_counter_ns()
        failure = None
        try:
            yield
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            ended = time.perf_counter_ns()
            self.stages.append({"phase": "cpu_video_numpy", "cpu_start_ns": started, "cpu_end_ns": ended,
                "cpu_wall_ms": (ended - started) / 1e6, "cuda_stream_elapsed_ms": None,
                "event_scope": "not_applicable_cpu", "cuda_status": "not_applicable",
                "allocator_status": "not_applicable", "allocator_before": None, "allocator_after": None,
                "execution_scope": "api_cpu_postprocess", "failure": failure})

    def __exit__(self, exc_type, exc_value, _traceback):
        ended = time.perf_counter_ns()
        if exc_type is not None:
            self.errors.append({"stage": "cpu_postprocess", "error_type": exc_type.__name__, "error": str(exc_value)[:1000]})
        value = {**self.context, **_proc_identity(), "actual_model_request_id": self.actual_model_request_id,
            "execution_scope": "api_cpu_postprocess", "source_device_type": self.source_device_type,
            "video_payload_path": "payload.video", "local_d2h": "not_executed_here",
            "upstream_transfer_receipt": "transfer-rank-0.json",
            "source_device": self.source_device, "rank": None, "world_size": None, "gpu_uuid": None,
            "gpu_uuid_reason": "not_applicable_cpu_postprocess", "cuda_status": "not_applicable",
            "allocator_status": "not_applicable", "allocator_before": None, "allocator_after": None,
            "allocator_window_peak": None, "allocator_process_peak_observed": None,
            "cuda_stream_elapsed_ms": None, "event_scope": "not_applicable_cpu",
            "clock_domain": "process_perf_counter_ns", "started_utc_ns": self.utc_ns,
            "cpu_start_ns": self.started, "cpu_end_ns": ended,
            "cpu_postprocess_wall_ms": (ended - self.started) / 1e6,
            "stages": self.stages, "actual_source_hashes": _source_hashes(),
            "status": "FAIL" if exc_type else "METRICS_INCOMPLETE" if self.errors else "PASS", "errors": self.errors,
            "device_to_host": "not_executed_here; see_worker_transport_receipt"}
        try:
            unique_json(output_directory(self.context) / f"postprocess-{os.getpid()}.json", value)
        except Exception as error:
            LOG.warning("SeedVR2 CPU postprocess metrics incomplete request=%s window=%s error=%s",
                self.context["sr_request_id"], self.context["window_index"], type(error).__name__)
        return False


@contextmanager
def transfer_scope(output):
    """Bind one SeedVR2 legacy output only while its original IPC pack runs."""
    context = None
    video = None
    try:
        if isinstance(output, dict) and isinstance(output.get("metadata"), dict) and isinstance(output.get("payload"), dict):
            candidate = output["metadata"].get(KEY)
            if candidate and candidate.get("measurement_mode") != "off":
                context = validate_context(candidate)
                context["actual_model_request_id"] = candidate.get("actual_model_request_id")
                video = output["payload"].get("video")
    except Exception as error:
        LOG.warning("SeedVR2 transfer metadata invalid: %s", type(error).__name__)
    holder = {"context": context, "video": video, "started": time.perf_counter_ns(), "observations": []} if context is not None else None
    token = _TRANSFER_SCOPE.set(holder)
    failure = None
    try:
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        _TRANSFER_SCOPE.reset(token)
        if holder is not None:
            ended = time.perf_counter_ns()
            for observation in holder["observations"]:
                value = {**observation, "framework_output_pack_wall_ms": (ended - holder["started"]) / 1e6,
                    "framework_output_pack": {"phase": "framework_output_pack", "cpu_start_ns": holder["started"],
                        "cpu_end_ns": ended, "cpu_wall_ms": (ended - holder["started"]) / 1e6,
                        "execution_scope": "worker_transport", "event_scope": "cpu_wall_envelope",
                        "scope": "original_legacy_output_dictionary_pack_including_pinned_allocation_D2H_wait_numpy_and_SHM; excludes_IPC_queue_and_receiver"}}
                if failure is not None:
                    value["status"] = "FAIL"
                    value["errors"] = [*value["errors"], {"stage": "framework_output_pack", "error_type": type(failure).__name__, "error": str(failure)[:1000]}]
                try:
                    unique_json(output_directory(context) / f"transfer-rank-{value['rank']}.json", value)
                except Exception as error:
                    LOG.warning("SeedVR2 framework output pack metrics incomplete: %s", type(error).__name__)


class Transport(Window):
    """GPU copy-call events plus a separately named CPU D2H/SHM envelope.

    Only the existing IPC synchronization resolves events.  This observer adds
    no CUDA wait, allocation or peak reset.  The interval is a copy API stream
    span, not a promise of isolated hardware DMA timing.
    """
    def __init__(self, context, tensor, stream, holder=None):
        super().__init__(context, tensor.device, execution_scope="worker_transport", reset_allocator_peak=False)
        self.actual_model_request_id = context.get("actual_model_request_id")
        self.tensor = tensor
        self.stream = stream
        self.source_device_type = getattr(tensor.device, "type", None)
        self.transport_mode = "async_side_stream" if stream is not None else "sync_current_stream"
        self.copy_kind = None
        self.holder = holder
        self.tensor_identity = {"python_id": id(tensor), "source_device": str(tensor.device)}
        for key, getter in (("shape", lambda: list(tensor.shape)), ("stride", lambda: list(tensor.stride())),
                ("dtype", lambda: str(tensor.dtype)), ("data_ptr", lambda: int(tensor.data_ptr())),
                ("storage_offset", lambda: int(tensor.storage_offset())), ("numel", lambda: int(tensor.numel())),
                ("element_size", lambda: int(tensor.element_size())),
                ("storage_data_ptr", lambda: int(tensor.untyped_storage().data_ptr()))):
            try:
                self.tensor_identity[key] = getter()
            except Exception:
                self.tensor_identity[key] = None

    def event(self):
        event = self.cuda.Event(enable_timing=True)
        event.record(self.stream if self.stream is not None else self.cuda.current_stream(self.device))
        return event

    def __enter__(self):
        self.started = time.perf_counter_ns()
        self.utc_ns = time.time_ns()
        try:
            output_directory(self.context)
            if self.source_device_type != "cuda":
                raise ValueError("Observed worker tensor is not CUDA; actual D2H was not observed")
            distributed = self.torch.distributed
            if distributed.is_available() and distributed.is_initialized():
                self.rank, self.world_size = distributed.get_rank(), distributed.get_world_size()
            else:
                self.rank, self.world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
            props = self.cuda.get_device_properties(self.device)
            self.gpu_uuid = str(props.uuid) if getattr(props, "uuid", None) is not None else None
            self.before = self.allocator()
        except Exception as error:
            self._error("transport_start", error)
            self.active = False
        self.active_token = _ACTIVE_TRANSFER.set(self)
        return self

    @contextmanager
    def copy_stage(self, kind):
        self.copy_kind = kind
        if not self.active:
            yield
            return
        started = time.perf_counter_ns()
        before = start = None
        try:
            before = self.allocator()
            start = self.event()
        except Exception as error:
            self._error("copy_start", error)
        failure = None
        try:
            yield
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            end = after = None
            try:
                end = self.event()
                after = self.allocator()
            except Exception as error:
                self._error("copy_end", error)
            self.stages.append({"phase": "device_to_host_transport", "cpu_start_ns": started,
                "cpu_end_ns": time.perf_counter_ns(), "allocator_before": before, "allocator_after": after,
                "failure": failure, "_events": (start, end)})

    def __exit__(self, exc_type, exc_value, _traceback):
        _ACTIVE_TRANSFER.reset(self.active_token)
        ended = time.perf_counter_ns()
        after = peak = None
        records = []
        for stage in self.stages:
            start, end = stage.pop("_events")
            elapsed = None
            if exc_type is None and start is not None and end is not None:
                try:
                    if not end.query():
                        raise RuntimeError("Copy event is not complete at the original IPC boundary; no extra wait was inserted")
                    elapsed = float(start.elapsed_time(end))
                except Exception as error:
                    self._error("copy_resolve", error)
            records.append({**stage, "cpu_wall_ms": (stage["cpu_end_ns"] - stage["cpu_start_ns"]) / 1e6,
                "cuda_stream_elapsed_ms": elapsed, "event_scope": "passed_d2h_stream" if self.stream is not None else "current_cuda_stream",
                "execution_scope": "worker_transport", "cuda_status": "applicable",
                "includes_collectives": False, "scope": "copy_API_call_stream_span; excludes subsequent host_numpy_and_SHM_copy"})
        if self.active:
            try:
                after = self.allocator()
                peak = {"allocated_bytes": int(self.cuda.max_memory_allocated(self.device)), "reserved_bytes": int(self.cuda.max_memory_reserved(self.device))}
            except Exception as error:
                self._error("transport_end", error)
        if exc_type is not None:
            self._error("transport", exc_value)
        if len(records) != 1:
            self._error("copy_count", ValueError("Expected exactly one observed video copy API call"))
        value = {**self.context, **_proc_identity(), "actual_model_request_id": self.actual_model_request_id,
            "rank": self.rank, "world_size": self.world_size, "gpu_uuid": self.gpu_uuid,
            "gpu_uuid_reason": None if self.gpu_uuid else "not_reported_by_torch",
            "execution_scope": "worker_transport", "source_device": str(self.device), "source_device_type": self.source_device_type,
            "transport_mode": self.transport_mode, "copy_kind": self.copy_kind,
            "video_payload_path": "payload.video",
            "observed_copy_count": len(records), "matched_payload_video": True,
            "tensor_identity": self.tensor_identity,
            "transport_stream_identity": str(self.stream) if self.stream is not None else "current_stream_for_source_device",
            "clock_domain": "process_perf_counter_ns", "started_utc_ns": self.utc_ns,
            "cpu_start_ns": self.started, "cpu_end_ns": ended, "video_tensor_pack_wall_ms": (ended - self.started) / 1e6,
            "video_tensor_pack_scope": "matched_video_tensor_D2H_and_host_numpy_and_SHM; contained_in_framework_output_pack",
            "cuda_status": "applicable" if self.source_device_type == "cuda" else "not_observed",
            "allocator_status": "applicable" if self.source_device_type == "cuda" else "not_applicable",
            "allocator_before": self.before, "allocator_after": after, "allocator_window_peak": None,
            "allocator_process_peak_observed": peak, "allocator_peak_scope": "process_since_last_external_reset_not_D2H_peak",
            "stages": records, "explicit_stage_waits": 0, "explicit_window_boundary_waits": 0,
            "actual_source_hashes": _source_hashes(), "status": "FAIL" if exc_type else "METRICS_INCOMPLETE" if self.errors else "PASS", "errors": self.errors}
        if self.holder is not None:
            self.holder["observations"].append(value)
        else:
            # Direct diagnostic callers retain explicit narrower scope. The
            # receiver will require the enclosing framework_output_pack too.
            try:
                unique_json(output_directory(self.context) / f"transfer-rank-{self.rank}.json", value)
            except Exception as error:
                LOG.warning("SeedVR2 transport metrics incomplete request=%s window=%s error=%s",
                    self.context["sr_request_id"], self.context["window_index"], type(error).__name__)
        return False


def tensor_transfer(tensor, d2h_stream):
    scope = _TRANSFER_SCOPE.get()
    if scope is None or tensor is not scope["video"]:
        return Disabled()
    try:
        return Transport(scope["context"], tensor, d2h_stream, holder=scope)
    except Exception as error:
        LOG.warning("SeedVR2 transfer observation unavailable: %s", type(error).__name__)
        return Disabled()


def copy_stage(kind):
    observation = _ACTIVE_TRANSFER.get()
    return nullcontext() if observation is None else observation.copy_stage(kind)

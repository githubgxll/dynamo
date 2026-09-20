# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiohttp routes for asynchronous, synchronous and downloadable videos."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import resource
import stat
import time

import aiohttp
from aiohttp import web

from dingo.video_gateway.artifact_store import UPLOAD_HEARTBEAT_NAME
from dingo.video_gateway.errors import GatewayError, StoreUnavailable
from dingo.video_gateway.file_io import _drain_on_cancel, opened_file, run_file_io
from dingo.video_gateway.form_parser import parse_multipart
from dingo.video_gateway.models import TERMINAL_STATUSES, TaskStatus
from dingo.video_gateway.service import VideoGatewayService

logger = logging.getLogger(__name__)

_SERVICE_KEY = web.AppKey("video_service", VideoGatewayService)
_UPSTREAM_KEY = web.AppKey("video_upstream", aiohttp.ClientSession)
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
_UPLOAD_HEARTBEAT_MAX_INTERVAL_S = 30.0


async def _keep_upload_active(marker_fd: int, grace_s: float) -> None:
    def refresh() -> None:
        # A delayed request must not revive an already expired upload while
        # another Gateway is deleting it. The fd survives the commit rename.
        if time.time() - os.fstat(marker_fd).st_mtime >= grace_s:
            raise TimeoutError("upload heartbeat expired")
        os.utime(marker_fd, None)

    while True:
        await asyncio.sleep(min(_UPLOAD_HEARTBEAT_MAX_INTERVAL_S, grace_s / 3))
        await run_file_io(refresh)


def _service(request: web.Request) -> VideoGatewayService:
    return request.app[_SERVICE_KEY]


def _gateway_draining_error() -> GatewayError:
    return GatewayError(
        503,
        "gateway_draining",
        "video Gateway is draining and is not accepting new submissions",
        error_type="service_unavailable_error",
        headers={"Retry-After": "1"},
    )


def _is_loopback(remote: str | None) -> bool:
    if remote is None:
        return False
    try:
        return ipaddress.ip_address(remote).is_loopback
    except ValueError:
        return False


@web.middleware
async def error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except GatewayError as exc:
        return web.json_response(
            exc.as_response(), status=exc.status, headers=exc.headers
        )
    except StoreUnavailable as exc:
        error = GatewayError(
            503,
            "service_unavailable",
            str(exc),
            error_type="service_unavailable_error",
        )
        return web.json_response(error.as_response(), status=503)
    except web.HTTPRequestEntityTooLarge:
        error = GatewayError(413, "payload_too_large", "request body is too large")
        return web.json_response(error.as_response(), status=413)
    except web.HTTPException as exc:
        error = GatewayError(
            exc.status,
            "http_request_error",
            exc.reason or "HTTP request failed",
        )
        headers = {"Allow": exc.headers["Allow"]} if "Allow" in exc.headers else None
        return web.json_response(
            error.as_response(), status=exc.status, headers=headers
        )
    except Exception:
        logger.exception("unhandled video Gateway request error")
        error = GatewayError(
            500,
            "internal_error",
            "internal video Gateway error",
            error_type="server_error",
        )
        return web.json_response(error.as_response(), status=500)


async def live(request: web.Request) -> web.Response:
    if not _service(request).dispatcher.live:
        raise GatewayError(
            503,
            "not_live",
            "video Gateway lost its owner lease and must restart",
            error_type="service_unavailable_error",
        )
    return web.json_response({"status": "live"})


async def ready(request: web.Request) -> web.Response:
    service = _service(request)
    if service.dispatcher.draining:
        raise _gateway_draining_error()
    if not service.dispatcher.ready:
        raise GatewayError(
            503,
            "not_ready",
            "video Gateway recovery or dispatcher startup is incomplete",
            error_type="service_unavailable_error",
        )
    await service.store.health()
    await service.artifacts.health()
    return web.json_response({"status": "ready"})


async def drain(request: web.Request) -> web.Response:
    if not _is_loopback(request.remote):
        raise GatewayError(
            403,
            "drain_forbidden",
            "Gateway drain may only be requested through the local loopback interface",
        )
    dispatcher = _service(request).dispatcher
    if dispatcher.begin_drain():
        logger.info("video Gateway entered draining state")
    return web.json_response(
        {"status": "draining"},
        headers={"Connection": "close"},
    )


async def models(request: web.Request) -> web.Response:
    service = _service(request)
    data = list(service.upstream_models)
    for model, pool in sorted(service.config.pools_by_model.items()):
        adapter = service.adapters[pool.pool_id]
        data.append(
            {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "dingo-video-gateway",
                "available": service.dispatcher.has_workers(pool.pool_id),
                "video_capabilities": adapter.capabilities(
                    max_result_bytes=service.config.media.max_result_bytes
                ),
            }
        )
    return web.json_response({"object": "list", "data": data})


async def _submit(request: web.Request, *, delivery_mode: str):
    service = _service(request)
    if service.dispatcher.draining:
        raise _gateway_draining_error()
    anticipated_input = (
        request.content_length or service.config.media.max_total_file_bytes
    )
    await service.ensure_submission_capacity(
        anticipated_input + service.config.media.max_result_bytes
    )
    upload_root = await service.artifacts.create_upload()

    async def submit():
        parsed = await parse_multipart(
            request,
            service.artifacts,
            service.config.media,
            upload_root=upload_root,
        )
        return await service.submit(
            fields=parsed.fields,
            uploads=parsed.uploads,
            upload_root=parsed.upload_root,
            delivery_mode=delivery_mode,
            idempotency_key=request.headers.get("Idempotency-Key"),
        )

    try:
        async with opened_file(
            (upload_root / UPLOAD_HEARTBEAT_NAME).open, "rb"
        ) as marker:
            operation = asyncio.create_task(submit(), name="video-upload-submit")
            heartbeat = asyncio.create_task(
                _keep_upload_active(
                    marker.fileno(), service.config.lifecycle.upload_grace_s
                ),
                name="video-upload-heartbeat",
            )
            try:
                done, _ = await asyncio.wait(
                    {operation, heartbeat}, return_when=asyncio.FIRST_COMPLETED
                )
                if operation in done:
                    return await operation
                # Fail closed if heartbeat I/O fails; do not let an unprotected
                # upload continue. Drain its file work before closing/unlinking.
                try:
                    await heartbeat
                except Exception as exc:
                    raise GatewayError(
                        503,
                        "upload_heartbeat_failed",
                        "upload storage heartbeat could not be maintained",
                        error_type="service_unavailable_error",
                    ) from exc
                raise RuntimeError("upload heartbeat stopped unexpectedly")
            finally:
                operation.cancel()
                heartbeat.cancel()
                await _drain_on_cancel(
                    asyncio.gather(operation, heartbeat, return_exceptions=True)
                )
    finally:
        # commit_upload moved the directory on success. Cleanup also runs on
        # disconnect, malformed multipart, and cancellation during validation.
        await service.artifacts.discard(upload_root)


async def create_video(request: web.Request) -> web.Response:
    submission = await _submit(request, delivery_mode="async")
    status = (
        _service(request).config.http.async_submit_status_code
        if submission.created
        else 200
    )
    task = submission.stored.task
    return web.json_response(
        task.public_dict(),
        status=status,
        headers={"Location": f"/v1/videos/{task.id}"},
    )


async def create_video_sync(request: web.Request) -> web.StreamResponse:
    service = _service(request)
    submission = await _submit(request, delivery_mode="sync")
    task_id = submission.stored.task.id
    try:
        stored = await service.dispatcher.wait_terminal(
            task_id, service.config.http.sync_timeout_s
        )
    except TimeoutError as exc:
        raise GatewayError(
            504,
            "gateway_timeout",
            f"synchronous wait timed out; continue polling task {task_id}",
            error_type="server_error",
            headers={"X-Video-Id": task_id},
        ) from exc
    if stored.task.status != TaskStatus.COMPLETED:
        message = (
            stored.task.error.message if stored.task.error else stored.task.status.value
        )
        raise GatewayError(
            422,
            "video_generation_failed",
            message,
            headers={"X-Video-Id": task_id},
        )
    return await _content_response(request, service, stored.task)


async def get_video(request: web.Request) -> web.Response:
    stored = await _service(request).store.get_task(request.match_info["task_id"])
    if stored is None:
        raise GatewayError(404, "video_not_found", "video task was not found")
    return web.json_response(stored.task.public_dict())


async def list_videos(request: web.Request) -> web.Response:
    service = _service(request)
    try:
        limit = int(request.query.get("limit", "20"))
    except ValueError as exc:
        raise GatewayError(
            400, "invalid_limit", "limit must be an integer", "limit"
        ) from exc
    if not 1 <= limit <= 100:
        raise GatewayError(
            400, "invalid_limit", "limit must be between 1 and 100", "limit"
        )
    order = request.query.get("order", "desc")
    if order not in {"asc", "desc"}:
        raise GatewayError(400, "invalid_order", "order must be asc or desc", "order")
    status_raw = request.query.get("status")
    try:
        status = TaskStatus(status_raw) if status_raw is not None else None
    except ValueError as exc:
        raise GatewayError(
            400, "invalid_status", "unknown video status", "status"
        ) from exc
    model = request.query.get("model")
    pool_id = service.resolve_pool(model).pool_id if model is not None else None
    tasks = await service.store.list_tasks(
        pool_id=pool_id,
        status=status,
        after=request.query.get("after"),
        limit=limit + 1,
        descending=order == "desc",
    )
    has_more = len(tasks) > limit
    tasks = tasks[:limit]
    data = [stored.task.public_dict() for stored in tasks]
    return web.json_response(
        {
            "object": "list",
            "data": data,
            "has_more": has_more,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
        }
    )


def _etag_matches(header: str | None, etag: str) -> bool:
    if header is None:
        return False
    opaque = etag.removeprefix("W/")
    return any(
        candidate == "*" or candidate.removeprefix("W/") == opaque
        for candidate in (part.strip() for part in header.split(","))
    )


def _range_bounds(value: str | None, size: int) -> tuple[int, int, bool]:
    """Return the inclusive byte bounds and whether a Range was requested."""

    if value is None:
        return 0, max(size - 1, -1), False
    if not value.startswith("bytes=") or "," in value or size <= 0:
        raise ValueError("unsupported byte range")
    spec = value[6:].strip()
    if spec.count("-") != 1:
        raise ValueError("invalid byte range")
    first, last = (part.strip() for part in spec.split("-", 1))
    if first:
        if not first.isdecimal() or (last and not last.isdecimal()):
            raise ValueError("invalid byte range")
        start = int(first)
        if start >= size:
            raise ValueError("byte range starts beyond the result")
        end = min(int(last), size - 1) if last else size - 1
        if end < start:
            raise ValueError("byte range end precedes its start")
        return start, end, True
    if not last or not last.isdecimal():
        raise ValueError("invalid suffix byte range")
    suffix = int(last)
    if suffix <= 0:
        raise ValueError("invalid suffix byte range")
    return max(0, size - suffix), size - 1, True


async def _content_response(
    request: web.Request, service: VideoGatewayService, task
) -> web.StreamResponse:
    if task.status == TaskStatus.EXPIRED:
        raise GatewayError(410, "video_expired", "video result has expired")
    if task.status != TaskStatus.COMPLETED or task.result_path is None:
        if task.status in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
            raise GatewayError(422, "video_generation_failed", task.status.value)
        raise GatewayError(409, "video_not_ready", "video result is not ready")
    if not task.result_sha256:
        raise GatewayError(410, "video_expired", "video result checksum is unavailable")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)

    def _open():
        try:
            path = service.artifacts.result_path(task.result_path)
            descriptor = os.open(path, flags)
        except (OSError, RuntimeError) as exc:
            raise GatewayError(
                410, "video_expired", "video result artifact is unavailable"
            ) from exc
        try:
            return os.fdopen(descriptor, "rb", closefd=True)
        except BaseException:
            os.close(descriptor)
            raise

    async with opened_file(_open) as stream:
        metadata = await run_file_io(os.fstat, stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise GatewayError(
                410, "video_expired", "video result artifact is unavailable"
            )
        size = metadata.st_size
        if task.result_bytes is not None and size != task.result_bytes:
            raise GatewayError(
                410, "video_expired", "video result artifact size has changed"
            )

        etag = f'"sha256-{task.result_sha256}"'
        common_headers = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'attachment; filename="{task.id}.mp4"',
            "Content-Type": "video/mp4",
            "ETag": etag,
            "X-Video-Id": task.id,
        }
        seed = task.normalized_request.get("seed")
        if seed is not None:
            common_headers["X-Video-Seed"] = str(seed)
        media = task.normalized_request.get("_result_media") or {}
        if media.get("frames") is not None:
            common_headers["X-Video-Frames"] = str(media["frames"])
        if media.get("fps") is not None:
            common_headers["X-Video-FPS"] = f"{float(media['fps']):.9g}"
        duration = media.get("duration_s", media.get("video_duration_s"))
        if duration is not None:
            common_headers["X-Video-Duration-Seconds"] = f"{float(duration):.9g}"
        if _etag_matches(request.headers.get("If-None-Match"), etag):
            return web.Response(status=304, headers=common_headers)

        try:
            start, end, partial = _range_bounds(request.headers.get("Range"), size)
        except ValueError as exc:
            raise GatewayError(
                416,
                "range_not_satisfiable",
                "only one satisfiable byte range is supported",
                headers={"Content-Range": f"bytes */{size}"},
            ) from exc

        length = max(0, end - start + 1)
        headers = {**common_headers, "Content-Length": str(length)}
        if partial:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        response = web.StreamResponse(status=206 if partial else 200, headers=headers)
        await response.prepare(request)
        if request.method != "HEAD" and length:
            await run_file_io(stream.seek, start)
            remaining = length
            while remaining:
                chunk = await run_file_io(stream.read, min(1024 * 1024, remaining))
                if not chunk:
                    raise ConnectionError("video artifact ended during download")
                await response.write(chunk)
                remaining -= len(chunk)
        await response.write_eof()
        return response


async def get_video_content(request: web.Request) -> web.StreamResponse:
    service = _service(request)
    stored = await service.store.get_task(request.match_info["task_id"])
    if stored is None:
        raise GatewayError(404, "video_not_found", "video task was not found")
    return await _content_response(request, service, stored.task)


async def delete_video(request: web.Request) -> web.Response:
    service = _service(request)
    task_id = request.match_info["task_id"]
    stored = await service.store.get_task(task_id)
    if stored is None:
        raise GatewayError(404, "video_not_found", "video task was not found")
    if stored.task.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.EXPIRED,
    }:
        try:
            stored = await service.expire(stored)
        except KeyError as exc:
            raise GatewayError(
                404, "video_not_found", "video task was not found"
            ) from exc
        return web.json_response(
            {"id": task_id, "object": "video.deleted", "deleted": True}
        )
    try:
        stored = await service.dispatcher.cancel(task_id)
    except KeyError as exc:
        raise GatewayError(404, "video_not_found", "video task was not found") from exc
    # A queued task is cancelled atomically and remains queryable as cancelled
    # after its first DELETE.  This is also the native vLLM-Omni response shape.
    if stored.task.status == TaskStatus.CANCELLED:
        return web.json_response(
            {"id": task_id, "object": "video.deleted", "deleted": True}
        )
    # Close the race between the initial read and request_cancel(): generation
    # may have completed or failed in that interval. Preserve the existing
    # result-expiry semantics; only an active cancellation request returns 202.
    if stored.task.status in TERMINAL_STATUSES:
        try:
            await service.expire(stored)
        except KeyError as exc:
            raise GatewayError(
                404, "video_not_found", "video task was not found"
            ) from exc
        return web.json_response(
            {"id": task_id, "object": "video.deleted", "deleted": True}
        )
    return web.json_response(
        {
            "id": task_id,
            "object": "video.cancel",
            "accepted": True,
            "deleted": False,
            "status": stored.task.public_dict()["status"],
            "cancel_requested": stored.task.cancel_requested_at_ms is not None,
        },
        status=202,
        headers={"Location": f"/v1/videos/{task_id}"},
    )


async def metrics(request: web.Request) -> web.Response:
    service = _service(request)
    budget = await service.dispatcher.memory_budget_snapshot()
    media = service.dispatcher.media_runtime_snapshot()
    artifact = service.dispatcher.artifact_runtime_snapshot()
    capacity = await service.artifacts.capacity()
    rss_bytes = _current_rss_bytes()
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    lines = [
        "# TYPE dingo_video_workers gauge",
        "# TYPE dingo_video_worker_busy gauge",
        "# TYPE dingo_video_queue_depth gauge",
        "# TYPE dingo_video_tasks gauge",
        "# TYPE dingo_video_finalization_pending_local gauge",
        "# TYPE dingo_video_media_memory_budget_bytes gauge",
        "# TYPE dingo_video_media_memory_used_bytes gauge",
        "# TYPE dingo_video_media_memory_peak_bytes gauge",
        "# TYPE dingo_video_media_memory_waiting_tasks gauge",
        "# TYPE dingo_video_media_memory_active_tasks gauge",
        "# TYPE dingo_video_media_legacy_input_encoded_bytes_total counter",
        "# TYPE dingo_video_media_legacy_output_encoded_bytes_total counter",
        "# TYPE dingo_video_media_payload_build_seconds_total counter",
        "# TYPE dingo_video_media_payload_build_total counter",
        "# TYPE dingo_video_media_finalize_seconds_total counter",
        "# TYPE dingo_video_media_finalize_total counter",
        "# TYPE dingo_video_media_result_oversize_total counter",
        "# TYPE dingo_video_process_rss_bytes gauge",
        "# TYPE dingo_video_process_peak_rss_bytes gauge",
        "# TYPE dingo_video_artifact_total_bytes gauge",
        "# TYPE dingo_video_artifact_free_bytes gauge",
        "# TYPE dingo_video_artifact_sweep_due_tasks gauge",
        "# TYPE dingo_video_artifact_expired_tasks_total counter",
        "# TYPE dingo_video_artifact_orphan_candidates_total counter",
        "# TYPE dingo_video_artifact_orphan_trashed_total counter",
        "# TYPE dingo_video_artifact_cleanup_failures_total counter",
        "# TYPE dingo_video_artifact_released_bytes_total counter",
        f"dingo_video_media_memory_budget_bytes {budget.capacity_bytes}",
        f"dingo_video_media_memory_used_bytes {budget.used_bytes}",
        f"dingo_video_media_memory_peak_bytes {budget.peak_bytes}",
        f"dingo_video_media_memory_waiting_tasks {budget.waiting_tasks}",
        f"dingo_video_media_memory_active_tasks {budget.active_tasks}",
        "dingo_video_media_legacy_input_encoded_bytes_total "
        f"{media.legacy_input_encoded_bytes}",
        "dingo_video_media_legacy_output_encoded_bytes_total "
        f"{media.legacy_output_encoded_bytes}",
        f"dingo_video_media_payload_build_seconds_total {media.payload_build_seconds}",
        f"dingo_video_media_payload_build_total {media.payload_build_count}",
        f"dingo_video_media_finalize_seconds_total {media.finalize_seconds}",
        f"dingo_video_media_finalize_total {media.finalize_count}",
        f"dingo_video_media_result_oversize_total {media.result_oversize_count}",
        f"dingo_video_process_rss_bytes {rss_bytes}",
        f"dingo_video_process_peak_rss_bytes {peak_rss_bytes}",
        f"dingo_video_artifact_total_bytes {capacity.total_bytes}",
        f"dingo_video_artifact_free_bytes {capacity.free_bytes}",
        f"dingo_video_artifact_sweep_due_tasks {artifact.sweep_due_tasks}",
        f"dingo_video_artifact_expired_tasks_total {artifact.expired_tasks_total}",
        "dingo_video_artifact_orphan_candidates_total "
        f"{artifact.orphan_candidates_total}",
        f"dingo_video_artifact_orphan_trashed_total {artifact.orphan_trashed_total}",
        "dingo_video_artifact_cleanup_failures_total "
        f"{artifact.cleanup_failures_total}",
        f"dingo_video_artifact_released_bytes_total {artifact.released_bytes_total}",
    ]
    for pool in service.config.pools:
        queue = await service.store.queue_depth(pool.pool_id)
        # Dispatch may skip all queue/counter reads when no Worker is free.
        # Refresh these gauges at scrape time instead of exposing stale values
        # merely because the pool is saturated or has no registered Workers.
        if hasattr(service.store, "retry_budget_used"):
            waiting = await service.store.retry_queue_depth(pool.pool_id)
            credits = await service.store.retry_budget_used(pool.pool_id)
            service.telemetry.set_gauge(
                "dingo_video_retry_waiting_tasks",
                waiting,
                labels={"pool": pool.pool_id},
            )
            service.telemetry.set_gauge(
                "dingo_video_retry_credits_used",
                credits,
                labels={"pool": pool.pool_id},
            )
            service.telemetry.set_gauge(
                "dingo_video_normal_queue_depth",
                max(0, queue - waiting),
                labels={"pool": pool.pool_id},
            )
        leases = await service.dispatcher.pool_leases(pool.pool_id)
        worker_capacity = service.dispatcher.pool_capacity_snapshot(
            pool.pool_id, leases
        )
        scheduling = service.dispatcher.pools[pool.pool_id].config.scheduling
        worker_capacity.update(
            {
                "worker_execution_capacity_configured": scheduling.worker_capacity,
                "worker_prefetch_capacity_configured": scheduling.worker_prefetch_capacity,
                "early_release_slot_enabled": int(scheduling.early_release_slot),
                "finalization_concurrency_configured": scheduling.finalization_concurrency,
                "finalization_pending_limit_configured": scheduling.finalization_pending_limit,
            }
        )
        counts = await service.store.task_counts(pool.pool_id)
        for name, value in worker_capacity.items():
            if pool is service.config.pools[0] and name not in {
                "workers",
                "worker_busy",
            }:
                lines.append(f"# TYPE dingo_video_{name} gauge")
            lines.append(f'dingo_video_{name}{{pool="{pool.pool_id}"}} {value}')
        lines.append(
            f'dingo_video_finalization_pending_local{{pool="{pool.pool_id}"}} '
            f"{service.dispatcher.pool_finalization_pending(pool.pool_id)}"
        )
        lines.append(f'dingo_video_queue_depth{{pool="{pool.pool_id}"}} {queue}')
        lines.extend(
            f'dingo_video_tasks{{pool="{pool.pool_id}",status="{status.value}"}} '
            f"{counts[status]}"
            for status in TaskStatus
        )
    lines.extend(service.telemetry.render_prometheus())
    return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


def _current_rss_bytes() -> int:
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


async def unsupported_stream(_request: web.Request) -> web.Response:
    raise GatewayError(
        404, "unsupported_endpoint", "/v1/videos/stream is not implemented"
    )


async def proxy(request: web.Request) -> web.StreamResponse:
    service = _service(request)
    upstream = service.config.http.upstream_url
    if upstream is None:
        raise GatewayError(404, "not_found", "route was not found")
    session = request.app[_UPSTREAM_KEY]
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in _HOP_BY_HOP
        and name.lower() != "host"
        and name.lower() != "content-length"
    }
    target = upstream + request.rel_url.path_qs
    body = request.content.iter_chunked(64 * 1024) if request.can_read_body else None
    try:
        async with session.request(
            request.method,
            target,
            headers=headers,
            data=body,
            allow_redirects=False,
        ) as upstream_response:
            response = web.StreamResponse(status=upstream_response.status)
            for name, value in upstream_response.headers.items():
                if name.lower() not in _HOP_BY_HOP and name.lower() != "content-length":
                    response.headers[name] = value
            await response.prepare(request)
            async for chunk in upstream_response.content.iter_chunked(64 * 1024):
                await response.write(chunk)
            await response.write_eof()
            return response
    except aiohttp.ClientError as exc:
        raise GatewayError(
            502,
            "upstream_unavailable",
            "configured upstream request failed",
            error_type="server_error",
        ) from exc


def register_routes(app: web.Application) -> None:
    app.router.add_get("/live", live)
    app.router.add_get("/ready", ready)
    app.router.add_post("/internal/drain", drain)
    # vLLM-Omni exposes /health; use readiness semantics so clients are not
    # sent to a Gateway before its stores and discovery loops are usable.
    app.router.add_get("/health", ready)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/videos", create_video)
    app.router.add_post("/v1/videos/sync", create_video_sync)
    app.router.add_post("/v1/videos/stream", unsupported_stream)
    app.router.add_get("/v1/videos", list_videos)
    app.router.add_get(
        "/v1/videos/{task_id}/content", get_video_content, allow_head=True
    )
    app.router.add_get("/v1/videos/{task_id}", get_video)
    app.router.add_delete("/v1/videos/{task_id}", delete_video)
    app.router.add_route("*", "/{tail:.*}", proxy)


def install_state(app: web.Application, service: VideoGatewayService) -> None:
    app[_SERVICE_KEY] = service


async def create_upstream_session(app: web.Application):
    app[_UPSTREAM_KEY] = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=10.0),
        auto_decompress=False,
    )
    session = app[_UPSTREAM_KEY]
    service = app[_SERVICE_KEY]
    try:
        assert service.config.http.upstream_url is not None
        async with session.get(
            service.config.http.upstream_url + "/v1/models"
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"configured upstream /v1/models returned HTTP {response.status}"
                )
            payload = await response.json()
        upstream_models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(upstream_models, list) or not all(
            isinstance(item, dict) and isinstance(item.get("id"), str)
            for item in upstream_models
        ):
            raise RuntimeError("configured upstream /v1/models has an invalid shape")
        upstream_ids = {item["id"] for item in upstream_models}
        conflicts = upstream_ids & set(service.config.pools_by_model)
        if conflicts:
            raise RuntimeError(
                "video/upstream model ID conflict: " + ", ".join(sorted(conflicts))
            )
        service.upstream_models = list(upstream_models)
        yield
    finally:
        await session.close()

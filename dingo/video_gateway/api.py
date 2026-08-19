# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiohttp routes for asynchronous, synchronous and downloadable videos."""

from __future__ import annotations

import asyncio
import logging
import os
import stat

import aiohttp
from aiohttp import web

from dingo.video_gateway.errors import GatewayError, StoreUnavailable
from dingo.video_gateway.form_parser import parse_multipart
from dingo.video_gateway.models import TaskStatus
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


def _service(request: web.Request) -> VideoGatewayService:
    return request.app[_SERVICE_KEY]


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


async def live(_request: web.Request) -> web.Response:
    return web.json_response({"status": "live"})


async def ready(request: web.Request) -> web.Response:
    service = _service(request)
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


async def models(request: web.Request) -> web.Response:
    service = _service(request)
    data = list(service.upstream_models)
    for model, pool in sorted(service.config.pools_by_model.items()):
        data.append(
            {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "dingo-video-gateway",
                "available": service.dispatcher.has_workers(pool.pool_id),
            }
        )
    return web.json_response({"object": "list", "data": data})


async def _submit(request: web.Request, *, delivery_mode: str):
    service = _service(request)
    parsed = await parse_multipart(request, service.artifacts)
    try:
        return await service.submit(
            fields=parsed.fields,
            uploads=parsed.uploads,
            upload_root=parsed.upload_root,
            delivery_mode=delivery_mode,
            idempotency_key=request.headers.get("Idempotency-Key"),
        )
    except Exception:
        await service.artifacts.discard(parsed.upload_root)
        raise


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
    try:
        path = service.artifacts.result_path(task.result_path)
    except (FileNotFoundError, RuntimeError) as exc:
        raise GatewayError(
            410, "video_expired", "video result artifact is unavailable"
        ) from exc
    if not task.result_sha256:
        raise GatewayError(410, "video_expired", "video result checksum is unavailable")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GatewayError(
            410, "video_expired", "video result artifact is unavailable"
        ) from exc

    stream = os.fdopen(descriptor, "rb", closefd=True)
    try:
        metadata = os.fstat(descriptor)
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
            await asyncio.to_thread(stream.seek, start)
            remaining = length
            while remaining:
                chunk = await asyncio.to_thread(
                    stream.read, min(1024 * 1024, remaining)
                )
                if not chunk:
                    raise ConnectionError("video artifact ended during download")
                await response.write(chunk)
                remaining -= len(chunk)
        await response.write_eof()
        return response
    finally:
        stream.close()


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
    pool = service.config.pools_by_id[stored.task.pool_id]
    try:
        stored = await service.dispatcher.wait_terminal(
            task_id, pool.scheduling.abort_grace_s
        )
    except TimeoutError:
        stored = await service.dispatcher.force_cancel(task_id)
    return web.json_response(
        {
            "id": task_id,
            "object": "video.deleted",
            "deleted": stored.task.status == TaskStatus.CANCELLED,
        }
    )


async def metrics(request: web.Request) -> web.Response:
    service = _service(request)
    lines = [
        "# TYPE dingo_video_workers gauge",
        "# TYPE dingo_video_worker_busy gauge",
        "# TYPE dingo_video_queue_depth gauge",
        "# TYPE dingo_video_tasks gauge",
    ]
    for pool in service.config.pools:
        workers = len(service.dispatcher.pool_instances(pool.pool_id))
        queue = await service.store.queue_depth(pool.pool_id)
        leases = await service.store.list_leases(pool.pool_id)
        busy = sum(lease.state != "quarantined" for lease in leases)
        tasks = await service.store.list_tasks(pool_id=pool.pool_id, limit=10_000)
        counts = {
            status: sum(task.task.status == status for task in tasks)
            for status in TaskStatus
        }
        lines.append(f'dingo_video_workers{{pool="{pool.pool_id}"}} {workers}')
        lines.append(f'dingo_video_worker_busy{{pool="{pool.pool_id}"}} {busy}')
        lines.append(f'dingo_video_queue_depth{{pool="{pool.pool_id}"}} {queue}')
        lines.extend(
            f'dingo_video_tasks{{pool="{pool.pool_id}",status="{status.value}"}} '
            f"{counts[status]}"
            for status in TaskStatus
        )
    return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


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

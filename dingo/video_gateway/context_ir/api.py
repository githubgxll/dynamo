"""Public compiler routes, registered before any catch-all upstream proxy."""
import asyncio
from aiohttp import web
from dingo.video_gateway.errors import GatewayError, StoreUnavailable, StoreConflict
from dingo.video_gateway.form_parser import parse_multipart

KEY = web.AppKey("context_ir_service", object)


def service(request):
    value = request.app.get(KEY)
    if value is None:
        raise GatewayError(503, "context_ir_disabled", "ContextIR is not enabled")
    return value


async def create(request):
    s = service(request)
    root, task_id, reservation = await s.begin_upload()
    ambiguous = False
    try:
        async with asyncio.timeout(s.config.upload_timeout_s):
            parsed = await parse_multipart(request, s.artifacts, s.video.config.media, upload_root=root)
            stored, created = await s.submit(parsed, task_id, reservation, request.headers.get("Idempotency-Key"))
        return web.json_response(await s.public(stored), status=202 if created else 200,
                                 headers={"Location": f"/v1/context-ir/{stored.task.id}"})
    except StoreUnavailable:
        ambiguous = True
        raise
    except TimeoutError as exc:
        raise GatewayError(408, "context_ir_upload_timeout", "compiler upload deadline exceeded") from exc
    except StoreConflict as exc:
        raise GatewayError(503, "context_ir_store_busy", "compiler admission is busy; retry with the same Idempotency-Key") from exc
    finally:
        s.uploading.discard(root.name)
        if not ambiguous:
            # Cleanup rechecks the persisted reservation/task, including on
            # cancellation after a successful admission but before response.
            await asyncio.shield(s.abandon_upload(root))


async def get(request):
    s = service(request)
    return web.json_response(await s.public(await s.get(request.match_info["task_id"])))


async def delete(request):
    s = service(request)
    stored = await s.delete(request.match_info["task_id"])
    return web.Response(status=204) if stored is None else web.json_response(await s.public(stored))


async def diagnostics(request):
    s = service(request)
    stored = await s.get(request.match_info["task_id"])
    t = stored.task
    value = {"id": t.id, "status": t.status, "phase": t.phase, "attempt": t.attempt,
                              "adapter": t.model_adapter_id, "adapter_version": t.model_adapter_version,
                              "metrics": t.metrics, "error": t.error, "usage": t.usage}
    try:
        normalized = await s.artifacts.read_bounded(t.request_path, s.config.result_max_bytes)
    except FileNotFoundError:
        latest = await s.store.get_task(t.id)
        if latest is None or latest.task.status == "expired":
            raise GatewayError(410, "context_ir_expired", "compiler task has expired")
        raise
    value["references"] = [{k: entry.get(k) for k in ("ordinal", "kind", "role", "soundtrack_policy")}
                           for entry in normalized["references"]]
    if t.attempt:
        try:
            upstream = await s.artifacts.read_bounded(
                f"tasks/{t.id}/attempts/{t.attempt:04d}/upstream-response.json", s.config.response_max_bytes)
        except FileNotFoundError:
            upstream = None
        if isinstance(upstream, dict):
            import re
            rules = [entry.get("rule") for entry in upstream.get("ir", {}).get("diagnostics", [])
                     if isinstance(entry, dict) and isinstance(entry.get("rule"), str)
                     and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", entry["rule"])]
            value["upstream"] = {"status": upstream.get("status") if upstream.get("status") in {"ready","degraded","invalid","needs_input"} else "error",
                                 "source": upstream.get("source") if upstream.get("source") in {"written","draft"} else None,
                                 "rule_codes": rules}
    return web.json_response(value)


def register(app):
    app.router.add_post("/v1/context-ir", create)
    app.router.add_get("/v1/context-ir/{task_id}", get)
    app.router.add_delete("/v1/context-ir/{task_id}", delete)
    app.router.add_get("/v1/context-ir/{task_id}/diagnostics", diagnostics)

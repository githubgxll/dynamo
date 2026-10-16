# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiohttp application factory with explicit Gateway lifecycle ownership."""

from __future__ import annotations

import asyncio
import os

from aiohttp import web

from dingo.video_gateway.api import (
    create_upstream_session,
    error_middleware,
    install_state,
    register_routes,
)
from dingo.video_gateway.service import VideoGatewayService


def create_app(service: VideoGatewayService) -> web.Application:
    if service.performance_cache is not None:
        token = os.environ.get(service.config.performance_cache.token_env, "")
        if token and (
            len(token) < 16 or not token.isascii() or any(c.isspace() for c in token)
        ):
            raise ValueError(
                "configured performance token must contain at least 16 ASCII characters without whitespace"
            )
    app = web.Application(
        client_max_size=service.config.http.max_body_bytes,
        middlewares=[error_middleware],
    )
    install_state(app, service)
    register_routes(app)

    async def dispatcher_context(_app: web.Application):
        await service.dispatcher.start()
        yield
        await service.dispatcher.stop()

    app.cleanup_ctx.append(dispatcher_context)
    if service.performance_cache is not None:

        async def expire_context(_app: web.Application):
            async def expire():
                while True:
                    service.performance_cache.stats()
                    await asyncio.sleep(
                        min(5.0, service.config.performance_cache.retention_s)
                    )

            task = asyncio.create_task(expire())
            try:
                yield
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        app.cleanup_ctx.append(expire_context)
    if service.config.http.upstream_url is not None:
        app.cleanup_ctx.append(create_upstream_session)
    return app

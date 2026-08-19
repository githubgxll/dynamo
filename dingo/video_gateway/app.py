# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiohttp application factory with explicit Gateway lifecycle ownership."""

from __future__ import annotations

from aiohttp import web

from dingo.video_gateway.api import (
    create_upstream_session,
    error_middleware,
    install_state,
    register_routes,
)
from dingo.video_gateway.service import VideoGatewayService


def create_app(service: VideoGatewayService) -> web.Application:
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
    if service.config.http.upstream_url is not None:
        app.cleanup_ctx.append(create_upstream_session)
    return app

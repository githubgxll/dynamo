# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the opt-in asynchronous video Gateway."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

from aiohttp import web

from dingo.video_gateway.adapters import create_adapter
from dingo.video_gateway.app import create_app
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.config import load_config
from dingo.video_gateway.dingo_adapter import create_pool_clients
from dingo.video_gateway.dispatcher import VideoDispatcher
from dingo.video_gateway.etcd_http import EtcdHttpClient
from dingo.video_gateway.service import VideoGatewayService
from dingo.video_gateway.task_store import EtcdTaskStore, MemoryTaskStore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dingo asynchronous video Gateway")
    parser.add_argument(
        "--config", required=True, help="versioned YAML/JSON config path"
    )
    parser.add_argument(
        "--allow-memory-store",
        action="store_true",
        help="allow the non-persistent memory Task Store for local tests only",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    from dynamo.runtime import DistributedRuntime
    from dynamo.runtime.logging import configure_dynamo_logging

    configure_dynamo_logging()
    config = load_config(args.config)
    if config.task_store.kind == "memory" and not args.allow_memory_store:
        raise RuntimeError("memory Task Store requires --allow-memory-store")
    adapters = {pool.pool_id: create_adapter(pool) for pool in config.pools}
    os.environ.pop("DYN_SYSTEM_PORT", None)
    runtime = DistributedRuntime(
        asyncio.get_running_loop(),
        config.runtime.discovery_backend,
        config.runtime.request_plane,
        event_plane=config.runtime.event_plane,
    )
    runner: web.AppRunner | None = None
    try:
        clients = await create_pool_clients(runtime, config)
        artifacts = FileArtifactStore(config.artifact_store.root)
        if config.task_store.kind == "memory":
            store = MemoryTaskStore()
        else:
            assert config.task_store.url is not None
            store = EtcdTaskStore(
                EtcdHttpClient(
                    config.task_store.url,
                    timeout_s=config.task_store.request_timeout_s,
                ),
                prefix=config.task_store.prefix,
                deployment_id=config.deployment_id,
            )
        dispatcher = VideoDispatcher(config, store, artifacts, clients, adapters)
        service = VideoGatewayService(config, store, artifacts, dispatcher, adapters)
        app = create_app(service)
        runner = web.AppRunner(app, handle_signals=False)
        await runner.setup()
        site = web.TCPSite(runner, config.http.host, config.http.port)
        await site.start()
        logger.info(
            "video Gateway listening on %s:%d with pools=%s",
            config.http.host,
            config.http.port,
            [pool.pool_id for pool in config.pools],
        )
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stopped.set)
        await stopped.wait()
    finally:
        try:
            if runner is not None:
                await runner.cleanup()
        finally:
            runtime.shutdown()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

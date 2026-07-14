# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import sys

import uvloop

from dingo.common.config_dump import dump_config
from dingo.common.constants import DisaggregationMode
from dingo.common.snapshot.restore_context import (
    parse_snapshot_restore_runtime_config,
    refresh_snapshot_restore_config,
)
from dingo.common.utils.runtime import create_runtime
from dynamo.runtime.logging import configure_dynamo_logging
from dingo.sglang.args import parse_args
from dingo.sglang.init_diffusion import (
    init_image_diffusion,
    init_llm_diffusion,
    init_video_diffusion,
)
from dingo.sglang.init_embedding import init_embedding
from dingo.sglang.init_llm import init_decode, init_prefill
from dingo.sglang.init_multimodal import (
    init_multimodal_encode_worker,
    init_multimodal_prefill_worker,
    init_multimodal_worker,
)
from dingo.sglang.shutdown import install_graceful_shutdown
from dingo.sglang.snapshot import prepare_snapshot_engine

configure_dynamo_logging()
logger = logging.getLogger(__name__)


async def worker(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]
    config = await parse_args(argv)
    dump_config(config.dynamo_args.dump_config_to, config)

    if config.server_args.load_format == "gms":
        from gpu_memory_service.integrations.sglang import setup_gms

        config.server_args.load_format = setup_gms(config.server_args)

    # Snapshot mode: engine must be created before runtime so CRIU captures no
    # NATS/etcd connections.
    snapshot_controller = await prepare_snapshot_engine(config.server_args)

    dynamo_args = config.dynamo_args
    snapshot_engine = None
    if snapshot_controller is not None:
        snapshot_engine = snapshot_controller.engine
        dynamo_args = await refresh_snapshot_restore_config(
            dynamo_args,
            lambda: parse_snapshot_restore_runtime_config(argv),
        )

    shutdown_event = asyncio.Event()
    shutdown_endpoints: list = []
    runtime, loop = create_runtime(
        discovery_backend=dynamo_args.discovery_backend,
        request_plane=dynamo_args.request_plane,
        event_plane=dynamo_args.event_plane,
    )

    run_deferred_handlers = install_graceful_shutdown(
        loop, runtime, shutdown_endpoints, shutdown_event
    )
    logger.info(
        "Signal handlers set up for graceful shutdown "
        "(discovery unregister + grace period, with chaining)"
    )

    if config.dynamo_args.image_diffusion_worker:
        await init_image_diffusion(
            runtime, config, shutdown_endpoints, run_deferred_handlers
        )
    elif config.dynamo_args.video_generation_worker:
        await init_video_diffusion(
            runtime, config, shutdown_endpoints, run_deferred_handlers
        )
    elif config.dynamo_args.embedding_worker:
        await init_embedding(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            run_deferred_handlers,
        )
    elif config.dynamo_args.multimodal_encode_worker:
        await init_multimodal_encode_worker(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            run_deferred_handlers,
        )
    elif config.dynamo_args.multimodal_worker:
        if config.serving_mode != DisaggregationMode.PREFILL:
            await init_multimodal_worker(
                runtime,
                config,
                shutdown_event,
                shutdown_endpoints,
                run_deferred_handlers,
            )
        else:
            await init_multimodal_prefill_worker(
                runtime,
                config,
                shutdown_event,
                shutdown_endpoints,
                run_deferred_handlers,
            )
    elif config.dynamo_args.diffusion_worker:
        await init_llm_diffusion(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            run_deferred_handlers,
        )
    elif config.serving_mode != DisaggregationMode.PREFILL:
        await init_decode(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            run_deferred_handlers,
            snapshot_engine=snapshot_engine,
        )
    else:
        await init_prefill(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            run_deferred_handlers,
            snapshot_engine=snapshot_engine,
        )


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()

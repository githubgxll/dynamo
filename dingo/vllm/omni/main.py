# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omni worker entrypoint for python -m dingo.vllm.omni."""

import asyncio
import json
import logging
import os

import uvloop
from dynamo import prometheus_names
from dynamo.llm import ModelInput, ModelType, WorkerType, fetch_model, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

from dingo.common.config_dump import dump_config
from dingo.common.storage import get_fs
from dingo.common.utils.graceful_shutdown import install_signal_handlers
from dingo.common.utils.output_modalities import get_output_modalities
from dingo.common.utils.runtime import create_runtime
from dingo.common.utils.runtime_termination import (
    run_with_runtime_termination_guard,
)
from dingo.vllm.health_check import VllmOmniHealthCheckPayload
from dingo.vllm.main import setup_metrics_collection
from dingo.vllm.omni.engine_monitor import OmniEngineMonitor

from .args import OmniConfig, parse_omni_args

configure_dynamo_logging()
logger = logging.getLogger(__name__)
shutdown_endpoints: list = []


async def init_omni(
    runtime: DistributedRuntime, config: OmniConfig, shutdown_event: asyncio.Event
):
    """Initialize Omni worker for multi-stage pipeline generation."""
    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    media_fs = (
        get_fs(config.media_output_fs_url) if config.media_output_fs_url else None
    )

    detached_manager = None
    sr_process = None
    video_readiness = None
    video_recovery = None
    production = config.h3_sr_profile is not None
    if production:
        from .production_handler import ProductionOmniHandler
        handler_class = ProductionOmniHandler
    else:
        from dingo.vllm.omni import OmniHandler
        handler_class = OmniHandler
    handler = handler_class(
        runtime=runtime,
        config=config,
        default_sampling_params={},
        # Detached tasks outlive the submitting endpoint request and must drain
        # before engine cleanup. The manager owns their cancellation deadline.
        shutdown_event=(
            None if config.detached_video_task_root is not None else shutdown_event
        ),
        media_output_fs=media_fs,
        media_output_http_url=config.media_output_http_url,
    )
    # Monitor AsyncOmni independently from the standard AsyncLLM worker path so
    # a dead stage/rank cannot leave a registered endpoint with no live engine.
    if production:
        await handler.check_ready()
        _engine_monitor = handler.start_monitor(runtime, shutdown_event)
    else:
        _engine_monitor = OmniEngineMonitor(runtime, handler.engine_client, shutdown_event)
    serve_handler = handler.generate
    if config.detached_video_task_root is not None:
        from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager

        task_handler = handler
        sr_config = os.getenv("DINGO_VIDEO_SEEDVR2_CONFIG")
        if sr_config and not production:
            from dingo.vllm.omni.seedvr2_process import SeedVR2Process

            if (config.diffusion.max_num_seqs or 1) != 1:
                raise ValueError("composite SR Worker requires execution capacity one")
            if os.getenv("DINGO_VIDEO_BINARY_RESULTS") != "1":
                raise ValueError("composite SR Worker requires binary results")
            sr_process = SeedVR2Process(json.loads(sr_config))
            try:
                task_handler = await sr_process.start(handler)
            except BaseException:
                handler.cleanup()
                raise
        base_recovery = not production and os.getenv("DINGO_VIDEO_H3_RECOVERY") == "1"
        if base_recovery and sr_process is None:
            if config.request_adapter != "minimax_h3" or config.request_adapter_workflow not in {"fl2va", "ref2va"}:
                raise ValueError("base recovery requires the H3 video adapter")
            if (config.diffusion.max_num_seqs or 1) != 1:
                raise ValueError("H3 video recovery requires execution capacity one")
            from dingo.vllm.omni.h3_video_handler import H3VideoHandler
            task_handler = H3VideoHandler(handler)
        detached_manager = DetachedOmniTaskManager(
            task_handler,
            config.detached_video_task_root,
            drain_timeout_s=config.detached_video_drain_timeout,
            execution_capacity=config.diffusion.max_num_seqs or 1,
            prefetch_capacity=config.detached_video_prefetch_capacity,
        )
        if (sr_process is not None or base_recovery) and os.getenv("DINGO_VIDEO_READINESS_PORT"):
            from dingo.vllm.omni.video_readiness import VideoReadinessServer
            video_readiness = VideoReadinessServer(
                detached_manager, handler.engine_client,
                port=int(os.environ["DINGO_VIDEO_READINESS_PORT"]),
                sr_port=8124 if sr_process is not None else None,
            )
            await video_readiness.start()
        if sr_process is not None or base_recovery:
            from dingo.vllm.omni.video_recovery import CompositeWorkerRecovery
            video_recovery = CompositeWorkerRecovery(
                detached_manager, generate_endpoint, handler, sr_process,
                shutdown_event=shutdown_event,
            )
            video_recovery.start()
        serve_handler = detached_manager.generate
        logger.info(
            "Enabled detached Omni video tasks below %s",
            config.detached_video_task_root,
        )

    logger.info("Omni worker initialized for model: %s", config.model)

    setup_metrics_collection(config, generate_endpoint, logger)

    if config.engine_args.data_parallel_rank:
        logger.info(
            "Non-leader DP rank %d; skipping endpoint registration",
            config.engine_args.data_parallel_rank,
        )
        await shutdown_event.wait()
        return

    model_type = get_output_modalities(config.output_modalities, config.model)
    if model_type is None:
        model_type = ModelType.Images

    try:
        await register_model(
            ModelInput.Text,
            model_type,
            generate_endpoint,
            config.model,
            config.served_model_name,
            kv_cache_block_size=config.engine_args.block_size,
            # Omni workers serve the full multi-stage pipeline behind one
            # endpoint; there is no prefill/decode split visible to the
            # frontend, so they register as Aggregated.
            worker_type=WorkerType.Aggregated,
            needs=[],
        )

        if production:
            await handler.publish_registration()

        logger.info("Starting to serve Omni worker endpoint...")

        health_check_payload = handler.health_check_payload() if production else (
            await VllmOmniHealthCheckPayload.create(handler.engine_client)
        ).to_dict()

        await generate_endpoint.serve_endpoint(
            serve_handler,
            graceful_shutdown=True,
            metrics_labels=[
                (
                    prometheus_names.labels.MODEL,
                    config.served_model_name or config.model,
                ),
                (
                    prometheus_names.labels.MODEL_NAME,
                    config.served_model_name or config.model,
                ),
            ],
            health_check_payload=health_check_payload,
        )
    except Exception as e:
        logger.error("Omni worker failed: %s", e)
        raise
    finally:
        logger.debug("Cleaning up Omni worker")
        try:
            if video_recovery is not None:
                await video_recovery.close()
            if video_readiness is not None:
                await video_readiness.close()
            if detached_manager is not None:
                await detached_manager.shutdown()
        finally:
            try:
                await asyncio.to_thread(handler.output_formatter.close)
            finally:
                try:
                    if production:
                        await handler.aclose()
                    if sr_process is not None:
                        await sr_process.close()
                finally:
                    handler.cleanup()


async def worker():
    config = parse_omni_args()

    dump_config(config.dump_config_to, config)

    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model

    if config.h3_sr_profile is None and not os.path.exists(config.model):
        await fetch_model(config.model)

    shutdown_event = asyncio.Event()
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
    )

    install_signal_handlers(loop, runtime, shutdown_endpoints, shutdown_event)

    if config.stage_id is not None:
        # Optional execution modes have their own upstream dependencies. Do not
        # import them when serving a full diffusion pipeline (or parsing --help).
        from dingo.vllm.omni.stage_worker import init_omni_stage

        await init_omni_stage(runtime, config, shutdown_endpoints, shutdown_event)
        logger.debug("init_omni_stage completed (stage %d)", config.stage_id)
    elif config.omni_router:
        from dingo.vllm.omni.stage_router import init_omni_stage_router

        await init_omni_stage_router(runtime, config, shutdown_endpoints)
        logger.debug("init_omni_stage_router completed")
    elif config.realtime:
        from dingo.vllm.omni.realtime_utils import init_omni_realtime

        await init_omni_realtime(runtime, config, shutdown_endpoints, shutdown_event)
        logger.debug("init_omni_realtime completed, exiting...")
    else:
        omni = init_omni(runtime, config, shutdown_event)
        if config.detached_video_task_root is not None:
            await run_with_runtime_termination_guard(
                omni,
                runtime,
                shutdown_event,
                component="detached vLLM-Omni Video Worker",
            )
        else:
            await omni
        logger.debug("Omni worker completed, exiting...")


def main():
    uvloop.run(worker())

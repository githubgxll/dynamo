# SPDX-License-Identifier: Apache-2.0
"""Dedicated SeedVR2 video Worker. Never constructs or loads an H3 engine."""
import argparse
import asyncio
import contextlib
import logging
import os


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--served-model-name', default='seedvr2-3b-sr')
    p.add_argument('--namespace', required=True)
    p.add_argument('--artifact-root', required=True)
    p.add_argument('--work-root', required=True)
    p.add_argument('--storage-id', required=True)
    p.add_argument('--image-digest', required=True)
    p.add_argument('--gpus', type=int, choices=(4,8), default=4)
    p.add_argument('--readiness-port', type=int, default=8001)
    return p


async def worker(args):
    from aiohttp import web
    from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
    from dingo.common.utils.runtime import create_runtime
    from dingo.common.utils.graceful_shutdown import install_signal_handlers
    from dingo.common.utils.runtime_termination import run_with_runtime_termination_guard
    from dingo.common.video_super_resolution import H3_SR_PROFILE_V3
    from dingo.common.video_task_protocol import ENVELOPE_KEY, SCHEMA_VERSION
    from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
    from dingo.vllm.omni.seedvr2_process import SeedVR2Process
    from dingo.vllm.omni.sr_only_handler import SROnlyHandler
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    runtime, loop = create_runtime(discovery_backend='etcd', request_plane='tcp', event_plane='zmq')
    endpoint = runtime.endpoint(args.namespace+'.backend.generate')
    stop = asyncio.Event()
    install_signal_handlers(loop, runtime, [endpoint], stop)
    process = SeedVR2Process(dict(model=args.model, work_root=args.work_root, gpus=args.gpus,
                                 engine_identity='standalone-sr-v1-'+args.image_digest,
                                 profile=H3_SR_PROFILE_V3, image_digest=args.image_digest))
    manager = None
    registered = False
    async def health(request):
        healthy = process.process is None or process.process.returncode is None
        return web.json_response({'healthy':healthy},status=200 if healthy else 503)
    async def ready(request):
        ok = registered and manager is not None and manager.accepting
        return web.json_response({'ready':ok},status=200 if ok else 503)
    async def metrics(request):
        return web.Response(body=generate_latest(),headers={'Content-Type':CONTENT_TYPE_LATEST})
    app = web.Application();app.router.add_get('/health',health);app.router.add_get('/ready',ready);app.router.add_get('/metrics',metrics)
    server = web.AppRunner(app);await server.setup();await web.TCPSite(server,'0.0.0.0',args.readiness_port).start()
    async def serve():
        nonlocal manager, registered
        base_free_wrapper = await process.start(None)
        handler = SROnlyHandler(base_free_wrapper, storage_id=args.storage_id)
        manager = DetachedOmniTaskManager(handler,args.artifact_root,execution_capacity=1,prefetch_capacity=0,
                                         binary_results=True,inline_results=True,drain_timeout_s=120)
        await register_model(ModelInput.Text,ModelType.Videos,endpoint,args.model,args.served_model_name,
                             kv_cache_block_size=16,worker_type=WorkerType.Aggregated,needs=[])
        registered = True
        async def supervise():
            while not stop.is_set():
                if not process.healthy() or not handler.video_ready:
                    # Allow the detached task to persist its original failure.
                    await asyncio.sleep(5)
                    await endpoint.unregister_endpoint_instance()
                    raise RuntimeError('SR process exited or was quarantined; restarting Worker')
                await asyncio.sleep(1)
        serving = asyncio.create_task(endpoint.serve_endpoint(manager.generate,graceful_shutdown=True,
            health_check_payload={ENVELOPE_KEY:{'schema_version':SCHEMA_VERSION,'op':'capabilities'}}))
        watching = asyncio.create_task(supervise())
        try:
            done,_=await asyncio.wait([serving,watching],return_when=asyncio.FIRST_COMPLETED)
            for task in done:await task
        finally:
            registered=False
            for task in (serving,watching):task.cancel()
            await asyncio.gather(serving,watching,return_exceptions=True)
    try:
        await run_with_runtime_termination_guard(serve(),runtime,stop,component='standalone SR Worker')
    finally:
        if manager is not None:
            with contextlib.suppress(Exception):await manager.shutdown()
        await process.close();await server.cleanup()


def main():
    logging.basicConfig(level=logging.INFO)
    args=parser().parse_args()
    import uvloop
    uvloop.run(worker(args))


if __name__=='__main__':
    main()

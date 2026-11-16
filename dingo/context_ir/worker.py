# SPDX-License-Identifier: Apache-2.0
"""ContextIR runtime Worker, with a dedicated Qwen and shared artifacts."""
import asyncio
import logging
import os
import re
import signal
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

from dingo.context_ir.handler import ContextIRHandler
from dingo.context_ir.runtime import create_task_manager
from dingo.worker.config import load_worker_settings


async def run():
    from dynamo.runtime import DistributedRuntime

    capacity = int(os.getenv("CONTEXT_IR_CAPACITY", "16"))
    settings = load_worker_settings(
        os.getenv("DINGO_WORKER_CONFIG"),
        adapter="context_ir_compile",
        execution_capacity=capacity,
        max_execution_capacity=capacity,
    )
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "etcd", "tcp", event_plane="zmq"
    )
    endpoint = runtime.endpoint(os.environ["CONTEXT_IR_ENDPOINT"])
    root = Path(os.environ["CONTEXT_IR_ARTIFACT_ROOT"])
    session = ClientSession(timeout=ClientTimeout(total=5))
    llm = os.environ["H3IR_LLM_URL"].removesuffix("/v1").rstrip("/")
    idle_streak = 0

    async def engine_idle():
        nonlocal idle_streak
        try:
            async with session.get(llm + "/metrics") as reply:
                if reply.status != 200:
                    return False
                text = await reply.text()
            counts = []
            for name in ("num_requests_running", "num_requests_waiting"):
                matches = re.findall(
                    r"^vllm:" + name + r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",
                    text,
                    re.MULTILINE,
                )
                if not matches:
                    return False
                counts.extend(float(v) for v in matches)
            idle_streak = idle_streak + 1 if all(v == 0 for v in counts) else 0
            return idle_streak >= 2
        except Exception:
            logging.getLogger(__name__).debug(
                "dedicated Qwen idle proof unavailable", exc_info=True
            )
            idle_streak = 0
            return False

    manager = create_task_manager(
        ContextIRHandler(root, storage_id=os.environ["CONTEXT_IR_STORAGE_ID"]),
        root,
        deployment_id=os.environ["CONTEXT_IR_DEPLOYMENT_ID"],
        pool_id=os.environ["CONTEXT_IR_POOL"],
        instance_id=str(endpoint.connection_id()),
        capacity_domain=os.environ["CONTEXT_IR_CAPACITY_DOMAIN"],
        storage_id=os.environ["CONTEXT_IR_STORAGE_ID"],
        capacity=settings.execution_capacity,
        engine_idle=engine_idle,
    )
    manager.components_ready = False
    await manager.start()
    app = web.Application()

    async def health(request):
        async with session.get(llm + "/health") as response:
            status = 200 if response.status == 200 and manager.accepting else 503
        return web.json_response(
            {
                "ready": status == 200,
                "active": len(manager.jobs),
                "recovering": manager.recovering,
                "draining": manager.draining,
            },
            status=status,
        )

    app.router.add_get("/ready", health)
    app.router.add_get("/health", lambda request: web.json_response({"alive": True}))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(
        runner, "0.0.0.0", int(os.getenv("CONTEXT_IR_HEALTH_PORT", "8126"))
    ).start()
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopped.set)

    async def serve_when_model_ready():
        while not stopped.is_set():
            try:
                async with session.get(llm + "/health") as response:
                    if response.status == 200:
                        manager.components_ready = True
                        return await endpoint.serve_endpoint(
                            manager.generate, graceful_shutdown=True
                        )
            except Exception:
                logging.getLogger(__name__).debug(
                    "waiting for dedicated Qwen", exc_info=True
                )
            await asyncio.sleep(2)

    async def watch_model():
        while not stopped.is_set():
            try:
                async with session.get(llm + "/health") as response:
                    manager.components_ready = response.status == 200
            except Exception:
                manager.components_ready = False
                logging.getLogger(__name__).debug(
                    "model health unavailable", exc_info=True
                )
            await asyncio.sleep(2)

    health_watch = asyncio.create_task(watch_model())
    serving = asyncio.create_task(serve_when_model_ready())
    stopping = asyncio.create_task(stopped.wait())
    try:
        done, _ = await asyncio.wait(
            [serving, stopping], return_when=asyncio.FIRST_COMPLETED
        )
        if serving in done:
            await serving
        else:
            await manager.shutdown()
    finally:
        manager.draining = True
        health_watch.cancel()
        stopping.cancel()
        serving.cancel()
        await asyncio.gather(stopping, serving, health_watch, return_exceptions=True)
        if manager.recovery_task:
            manager.recovery_task.cancel()
        await session.close()
        await runner.cleanup()
        runtime.shutdown()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()

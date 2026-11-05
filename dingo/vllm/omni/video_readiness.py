# SPDX-License-Identifier: Apache-2.0
"""Composite readiness is distinct from Runtime process liveness.

A paused process stays alive for status/result/drain RPCs. Confirmed request
settlement can restore admission; engine death or unresolved execution exits
the video Worker. Readiness never submits inference or retries work.
"""
import asyncio
from aiohttp import ClientSession, ClientTimeout, web

class VideoReadinessServer:
    def __init__(self, manager, engine, *, port, runtime_port=8081, sr_port=8124):
        if port in (runtime_port, sr_port) or not 1024 <= port <= 65535:
            raise ValueError('readiness needs a separate unprivileged port')
        self.manager, self.engine = manager, engine
        self.port, self.runtime_port, self.sr_port = port, runtime_port, sr_port
        self.runner = None
        self.monitor = None

    async def check_components(self):
        if not self.manager._accepting or not bool(getattr(self.manager.handler, "video_ready", getattr(self.manager.handler, "sr_ready", True))):
            return False
        try:
            await asyncio.wait_for(self.engine.check_health(), timeout=1.0)
            async with ClientSession(trust_env=False, timeout=ClientTimeout(total=1.0)) as session:
                for port, route in ((self.runtime_port, '/health'), (self.sr_port, '/health')):
                    if port is None:
                        continue
                    async with session.get(f'http://127.0.0.1:{port}{route}', allow_redirects=False) as response:
                        if response.status != 200:
                            return False
        except Exception:
            return False
        return True

    async def watch(self):
        while True:
            self.manager.components_ready = await self.check_components()
            await asyncio.sleep(2)

    async def ready(self, request):
        ready = self.manager.accepting
        return web.json_response({'status': 'ready' if ready else 'not_ready'}, status=200 if ready else 503)

    async def start(self):
        self.manager.components_ready = False
        app = web.Application()
        app.router.add_get('/ready', self.ready)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        await web.TCPSite(self.runner, '0.0.0.0', self.port).start()
        self.monitor = asyncio.create_task(self.watch())

    async def close(self):
        self.manager.components_ready = False
        if self.monitor is not None:
            self.monitor.cancel()
            await asyncio.gather(self.monitor, return_exceptions=True)
        if self.runner is not None:
            await self.runner.cleanup()

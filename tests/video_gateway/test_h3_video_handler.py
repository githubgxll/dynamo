import asyncio
from types import SimpleNamespace as N
from unittest.mock import AsyncMock
import pytest
from dingo.common.video_worker_error import VideoWorkerFailure
from dingo.common.video_task_protocol import ENVELOPE_KEY
from dingo.vllm.omni.h3_video_handler import H3VideoHandler
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from dingo.vllm.omni.video_recovery import CompositeWorkerRecovery
from dingo.vllm.omni.video_readiness import VideoReadinessServer

class Engine:
    def __init__(self):
        self.client=N(collective_rpc_async=AsyncMock())
        self.engine_client=N(check_health=AsyncMock(),engine=N(stage_pools=[N(clients=[self.client])]))
        self.fail=True
    async def generate(self,payload,context):
        if self.fail:
            yield {'status':'failed','error':'CUDA out of memory','_internal_error':{'schema_version':1,'reason':'resource_exhausted','stage':'base_generation'}}
        else:yield {'status':'completed','data':[]}

@pytest.mark.asyncio
@pytest.mark.parametrize('timeout',[False,True])
async def test_base_oom_settlement_and_capabilities(tmp_path,timeout):
    engine=Engine();handler=H3VideoHandler(engine)
    if timeout:engine.client.collective_rpc_async.side_effect=TimeoutError('not settled')
    with pytest.raises(VideoWorkerFailure) as error:
        _=[x async for x in handler.generate({},None)]
    assert error.value.internal_error['reason']=='resource_exhausted'
    assert handler.quarantined is timeout
    manager=DetachedOmniTaskManager(handler,tmp_path,execution_capacity=1,prefetch_capacity=1)
    value=[x async for x in manager.generate({ENVELOPE_KEY:{'schema_version':1,'op':'capabilities'}},N())][0]
    assert 'super_resolution' not in value
    assert 'video_super_resolution_v1' not in value['capabilities']
    assert value['accepting'] is (not timeout)
    assert value['admission_capacity']==(0 if timeout else 2)
    if not timeout:
        engine.fail=False
        assert [x async for x in handler.generate({},None)]==[{'status':'completed','data':[]}]

@pytest.mark.asyncio
async def test_base_recovery_without_sr(tmp_path):
    handler=H3VideoHandler(Engine());handler.quarantine('unsettled')
    calls=[]
    manager=N(handler=handler,_accepting=True,components_ready=True,shutdown=AsyncMock())
    endpoint=N(unregister_endpoint_instance=AsyncMock())
    raw=N(engine_client=handler.handler.engine_client,cleanup=lambda:calls.append('cleanup'))
    recovery=CompositeWorkerRecovery(manager,endpoint,raw,None,exit_fn=lambda code:calls.append(code),ranks_factory=lambda *a:pytest.fail('SR rank tracking in base mode'))
    await recovery.watch()
    assert calls==['cleanup',70]
    endpoint.unregister_endpoint_instance.assert_awaited_once()
    assert not manager._accepting

@pytest.mark.asyncio
async def test_base_readiness_only_checks_runtime(tmp_path):
    from aiohttp import web
    app=web.Application();app.router.add_get('/health',lambda request:web.Response())
    runner=web.AppRunner(app);await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    port=site._server.sockets[0].getsockname()[1]
    try:
        handler=H3VideoHandler(Engine());manager=DetachedOmniTaskManager(handler,tmp_path)
        server=VideoReadinessServer(manager,handler.handler.engine_client,port=18082,runtime_port=port,sr_port=None)
        assert await server.check_components()
        handler.pause();assert not await server.check_components()
    finally:await runner.cleanup()

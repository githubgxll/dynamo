import asyncio, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import AsyncMock
from dingo.common.video_worker_error import classify_worker_error, VideoWorkerFailure, normalize_internal_error
from dingo.vllm.omni.seedvr2_handler import SeedVR2Handler
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from dingo.vllm.omni.video_readiness import VideoReadinessServer
from dingo.video_gateway.dispatcher import VideoDispatcher
from dingo.video_gateway.errors import worker_execution_error, WorkerUnavailable
from dingo.video_gateway.models import VideoTask, TaskStatus, TaskError

class Contracts(unittest.IsolatedAsyncioTestCase):
    def test_classification_does_not_infer_generic_runtime_oom(self):
        self.assertEqual(classify_worker_error(RuntimeError('unexpected'))['reason'],'unknown')
        self.assertEqual(classify_worker_error(RuntimeError('CUDA out of memory. GPU 0 /secret'))['reason'],'resource_exhausted')
        self.assertEqual(classify_worker_error(ValueError('bad input'),stage='input_processing')['reason'],'input_processing_failed')
        self.assertEqual(classify_worker_error(RuntimeError('Executor shut down'))['reason'],'engine_unavailable')
        e=worker_execution_error({'code':'worker_failed','message':'GPU 0 /secret','internal_error':{'schema_version':1,'reason':'resource_exhausted','stage':'base_generation','stack':'secret'}})
        self.assertNotIsInstance(e,WorkerUnavailable)
        self.assertEqual(e.internal_error,{'schema_version':1,'reason':'resource_exhausted','stage':'base_generation'})
        self.assertFalse(hasattr(worker_execution_error({'code':'worker_failed','message':'unrecognized'}),'internal_error'))

    def test_malformed_internal_metadata_is_unknown(self):
        self.assertEqual(normalize_internal_error({"schema_version":1,"reason":[],"stage":{}}), {"schema_version":1,"reason":"unknown","stage":"unknown"})

    def test_public_serialization_excludes_internal_diagnostics(self):
        t=VideoTask(1,'test','dep','pool','model','backend','target','revision','detached',TaskStatus.FAILED,'digest','req','manifest',1,1,2)
        t.error=TaskError('worker_failed','video Worker execution or result validation failed')
        t.internal_error={'schema_version':1,'reason':'resource_exhausted','stage':'base_generation','secret':'GPU 7 /private/path'}
        self.assertIn('internal_error',t.to_dict())
        for value in [t.public_dict(), t.diagnostics_dict()]:
            text=json.dumps(value)
            for secret in ['internal_error','resource_exhausted','private/path','GPU 7']:self.assertNotIn(secret,text)
        self.assertEqual(VideoTask.from_dict(t.to_dict()).internal_error,t.internal_error)

    async def test_composite_preserves_failed_base_reason(self):
        for reason, message in [('resource_exhausted', 'CUDA out of memory. GPU 0 /private/path'), ('unknown', 'unrecognized engine failure')]:
            class H3:
                async def generate(self,payload,ctx):
                    yield {'status':'failed','error':message,'_internal_error':{'schema_version':1,'reason':reason,'stage':'base_generation'}}
            h3=H3()
            h3.engine_client=N(check_health=AsyncMock(),engine=N(stage_pools=[N(clients=[N(collective_rpc_async=AsyncMock())])]))
            h=SeedVR2Handler(h3,N(quarantine=lambda r:None),engine_identity='test')
            with tempfile.TemporaryDirectory() as td:
                root=Path(td)/'attempts'/'one';root.mkdir(parents=True)
                with self.assertRaises(VideoWorkerFailure) as c:await h._base({},root,'fingerprint',N(is_stopped=lambda:False))
                self.assertEqual(c.exception.internal_error['reason'],reason)
                e=worker_execution_error({'code':'worker_failed','message':str(c.exception),'internal_error':classify_worker_error(c.exception)})
                self.assertEqual(e.internal_error['reason'],reason)
            self.assertEqual(h.sr_ready, reason != 'resource_exhausted')

    async def test_known_input_failure_does_not_quarantine(self):
        class H3:
            async def generate(self,payload,ctx):
                yield {'status':'failed','error':'invalid input','_internal_error':{'schema_version':1,'reason':'input_processing_failed','stage':'input_processing'}}
        h=SeedVR2Handler(H3(),N(quarantine=lambda r:None),engine_identity='test')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'attempts'/'one';root.mkdir(parents=True)
            with self.assertRaises(VideoWorkerFailure):await h._base({},root,'fp',N(is_stopped=lambda:False))
        self.assertTrue(h.sr_ready)

    async def test_quarantine_capacity_and_readiness_agree(self):
        h=SeedVR2Handler(N(),N(quarantine=lambda r:None),engine_identity='test')
        with tempfile.TemporaryDirectory() as td:
            m=DetachedOmniTaskManager(h,td)
            health=VideoReadinessServer(m,N(),port=18082)
            self.assertEqual((await health.ready(None)).status,200)
            h.quarantine('injected failure')
            self.assertEqual((await health.ready(None)).status,503)
            from dingo.common.video_task_protocol import ENVELOPE_KEY
            c=[v async for v in m.generate({ENVELOPE_KEY:{'schema_version':1,'op':'capabilities'}},N())][0]
            self.assertFalse(c['accepting']);self.assertEqual(c['execution_capacity'],0)
            self.assertFalse(c['super_resolution']['ready'])

    async def test_gateway_zero_is_not_legacy_fallback(self):
        from dingo.common.video_task_protocol import EXECUTION_CAPACITY_CAPABILITY
        class Client:
            async def direct(self,*a):
                async def stream():yield {'schema_version':1,'capabilities':[EXECUTION_CAPACITY_CAPABILITY], 'execution_capacity':0,'accepting':False}
                return stream()
        d=VideoDispatcher.__new__(VideoDispatcher);d.context_factory=lambda *a:N(stop_generating=lambda:None)
        p=N(client=Client(),prefetch_instances=set(),sr_profiles={},config=N(pool_id='p',sr_profile='profile',scheduling=N(worker_capacity=1,worker_prefetch_capacity=0)),instance_ids=[7],capacity_cache={})
        capacity,until=await d._worker_capacity(p,7);self.assertEqual(capacity,0)
        p.capacity_cache[7]=(capacity,until)
        self.assertEqual(d._worker_slots(p),[])
        d.pools={'p':p};self.assertFalse(d.has_workers('p'))

if __name__=='__main__':unittest.main()

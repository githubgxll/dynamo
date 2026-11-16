# SPDX-License-Identifier: Apache-2.0
"""Real multipart Gateway -> runtime control -> durable Worker integration."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter
from dingo.video_gateway.api import error_middleware
from dingo.video_gateway.context_ir.api import KEY, register
from dingo.video_gateway.context_ir.config import BackendConfig, ContextIRConfig
from dingo.video_gateway.context_ir.service import ContextIRService
from dingo.context_ir.runtime import create_task_manager
from tests.video_gateway.test_context_ir_http import form
from tests.video_gateway.test_etcd_task_store import FakeEtcd


@pytest.fixture
async def worker_stack(make_gateway_config):
 cfg=make_gateway_config()
 backend=BackendConfig(transport='worker',worker_pool='compiler',backend_target='dyn://test.compiler.compile',storage_id='shared')
 cfg=replace(cfg,context_ir=ContextIRConfig(enabled=True,backends={'h3':backend},model_bindings={'public-fl':'h3'},sweeper_interval_s=.1))
 gate=asyncio.Event();gate.set()
 class Handler:
  async def execute(self,request,root,cancelled):
   await gate.wait()
   return {'result':{'content':{'prompt':'A bird flies.'},'warnings':[]},'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15,'complete':True}}
 manager=create_task_manager(Handler(),cfg.artifact_store.root/'_context_ir'/cfg.deployment_id,
      deployment_id=cfg.deployment_id,pool_id='compiler',instance_id='1',capacity_domain='qwen',storage_id='shared')
 await manager.start()
 class Client:
  def instance_ids(self):return [1]
  async def direct(self,payload,instance_id,context):return manager.generate(payload)
 async def capacity(required):pass
 video=SimpleNamespace(config=cfg,dispatcher=SimpleNamespace(draining=False,context_factory=lambda *a:SimpleNamespace(stop_generating=lambda:None)),
   resolve_pool=cfg.pool_for_model,adapters={cfg.pools[0].pool_id:MiniMaxH3VideoAdapter(cfg.pools[0])},
   ensure_submission_capacity=capacity,context_ir_clients={'h3':Client()})
 service=ContextIRService(video,client=FakeEtcd())
 app=web.Application(middlewares=[error_middleware]);app[KEY]=service;register(app)
 client=TestClient(TestServer(app));await client.start_server();await service.start()
 yield service,manager,client,gate
 gate.set();await manager.shutdown(2);await service.stop();await client.close()

async def wait_terminal(client,task_id):
 for _ in range(80):
  data=await (await client.get('/v1/context-ir/'+task_id)).json()
  if data['status'] in {'completed','failed','cancelled'}:return data
  await asyncio.sleep(.1)
 raise AssertionError(data)

async def test_worker_transport_public_result_and_usage(worker_stack):
 _service,manager,client,_gate=worker_stack
 response=await client.post('/v1/context-ir',data=form())
 assert response.status==202,await response.text()
 created=await response.json()
 result=await wait_terminal(client,created['id'])
 assert result['status']=='completed',result
 assert result['content']['prompt']=='A bird flies.'
 assert result['usage']['total_tokens']==15
 assert set(result)-set(created)=={'completed_at','expires_at','content','warnings'}-{'completed_at','expires_at'}
 assert not manager.jobs

async def test_worker_cancel_keeps_slot_until_physical_completion(worker_stack):
 service,manager,client,gate=worker_stack
 gate.clear()
 response=await client.post('/v1/context-ir',data=form())
 created=await response.json();task_id=created['id']
 for _ in range(40):
  if manager.jobs:break
  await asyncio.sleep(.1)
 assert manager.jobs
 await client.delete('/v1/context-ir/'+task_id)
 assert (await wait_terminal(client,task_id))['status']=='cancelled'
 assert manager.jobs
 gate.set()
 for _ in range(50):
  active=await service.store.client.range('/dingo/task-runtime/v2/'+service.video.config.deployment_id+'/active/',prefix=True)
  if not active:break
  await asyncio.sleep(.1)
 assert not active

async def test_terminal_worker_releases_unused_storage_reservation(worker_stack):
 service,manager,client,gate=worker_stack
 response=await client.post("/v1/context-ir",data=form())
 created=await response.json()
 result=await wait_terminal(client,created["id"])
 assert result["status"]=="completed"
 for _ in range(20):
  stored=await service.store.get_task(created["id"])
  if stored.task.reserved_bytes < 1024*1024:break
  await asyncio.sleep(.1)
 assert stored.task.reserved_bytes<1024*1024
 reserved=await service.store.client.get(service.store.key("counters/h3/reserved_bytes"))
 assert int(reserved.value)==stored.task.reserved_bytes

async def test_worker_prometheus_terminal_and_timings_are_recorded_once(worker_stack):
 service,manager,client,gate=worker_stack
 response=await client.post('/v1/context-ir',data=form())
 tid=(await response.json())['id']
 result=await wait_terminal(client,tid)
 assert result['status']=='completed'
 lines='\n'.join(await service.metrics())
 assert 'dingo_context_ir_terminal_total{backend="h3",model="public-fl",state="completed"} 1' in lines
 for stage in ['execution_time_s','queue_wait_s']:
  assert f'dingo_context_ir_stage_duration_seconds_count{{backend="h3",model="public-fl",stage="{stage}"}} 1' in lines
 for _ in range(3):
  await client.get('/v1/context-ir/'+tid)
 assert lines=='\n'.join(await service.metrics())

async def test_worker_cancel_is_counted_before_settlement_without_double_count(worker_stack):
 service,manager,client,gate=worker_stack
 gate.clear()
 response=await client.post('/v1/context-ir',data=form())
 tid=(await response.json())['id']
 for _ in range(40):
  if manager.jobs:break
  await asyncio.sleep(.01)
 await client.delete('/v1/context-ir/'+tid)
 assert (await wait_terminal(client,tid))['status']=='cancelled'
 gate.set()
 for _ in range(80):
  slots,_=await service.store.client.range_all('/dingo/task-runtime/v2/'+service.video.config.deployment_id+'/slots/',prefix=True)
  if not slots:break
  await asyncio.sleep(.01)
 lines='\n'.join(await service.metrics())
 assert 'dingo_context_ir_terminal_total{backend="h3",model="public-fl",state="cancelled"} 1' in lines
 assert 'state="completed"' not in lines

async def test_metrics_separate_queue_limit_occupied_and_free_slots(worker_stack):
 import time
 from prometheus_client.parser import text_string_to_metric_families
 service,manager,client,gate=worker_stack
 async def values():
  return {s.name:s.value for f in text_string_to_metric_families('\n'.join(await service.metrics())) for s in f.samples}
 await service.dispatcher.worker.refresh()
 idle=await values()
 assert idle['dingo_context_ir_queue_limit']==32
 assert idle['dingo_context_ir_queued']==0
 assert idle['dingo_context_ir_inflight']==0
 assert idle['dingo_context_ir_available_execution_slots']>0
 gate.clear()
 created=await (await client.post('/v1/context-ir',data=form())).json()
 for _ in range(50):
  active=await values()
  if active['dingo_context_ir_inflight']==1:break
  await asyncio.sleep(.05)
 assert active['dingo_context_ir_inflight']==1
 assert active['dingo_context_ir_available_execution_slots']==idle['dingo_context_ir_available_execution_slots']-1
 gate.set();await wait_terminal(client,created['id'])

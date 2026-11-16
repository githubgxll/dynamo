# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from dataclasses import asdict

import pytest

from dingo.common.task_protocol import ControlRequest, InputDescriptor, TaskIdentity
from dingo.context_ir.runtime import create_task_manager


class Handler:
    def __init__(self):
        self.calls=0; self.release=asyncio.Event()
    async def execute(self,request,root,cancelled):
        self.calls+=1
        await self.release.wait()
        return {'result':{'content':{'prompt':'compiled'}},'usage':{'complete':True}}

def manager(tmp_path, handler):
    return create_task_manager(handler,tmp_path,deployment_id='d',pool_id='p',instance_id='1',
        capacity_domain='qwen-one',storage_id='shared')

def message(m, op='submit', i=0):
    identity=TaskIdentity('d','context_ir','p',f'task-{i}',1,f'{i+1:032x}','1',m.generation)
    payload=InputDescriptor('shared','request.json','manifest.json','a'*64,'route.json','b'*64,'c'*64)
    return ControlRequest(op,'d','p','1',m.generation,
        identity if op not in {'capabilities','drain'} else None,
        payload if op=='submit' else None,
        int(time.time()*1000)+60000 if op=='submit' else None,
        100 if op=='wait' else None).to_dict()

async def call(m, message):
    responses=[item async for item in m.generate(message)]
    assert len(responses)==1
    return responses[0]

async def test_capacity_duplicate_and_drain(tmp_path):
    handler=Handler();m=manager(tmp_path,handler);await m.start()
    requests=[message(m,i=i) for i in range(17)]
    results=await asyncio.gather(*(call(m,r) for r in requests))
    assert sum(r['state']=='accepted' for r in results)==16
    assert results[-1]['state']=='not_accepted'
    assert (await call(m,requests[0]))['state'] in {'accepted','running'}
    await asyncio.sleep(0)
    assert handler.calls==16
    await call(m,message(m,'drain'))
    assert not m.accepting
    handler.release.set();assert await m.shutdown(2)
    assert (await call(m,message(m,'status')))['state']=='succeeded'

async def test_cancel_keeps_physical_permit_until_handler_finishes(tmp_path):
    handler=Handler();m=manager(tmp_path,handler);await m.start()
    await call(m,message(m));await asyncio.sleep(0)
    await call(m,message(m,'cancel'))
    assert len(m.jobs)==1
    handler.release.set();assert await m.shutdown(2)
    assert (await call(m,message(m,'status')))['state']=='cancelled'

async def test_disconnect_does_not_cancel_and_result_survives_restart(tmp_path):
    handler=Handler();handler.release.set();m=manager(tmp_path,handler);await m.start()
    request=message(m);old_status=message(m,'status')
    await call(m,request);await m.shutdown(2)
    second=manager(tmp_path,Handler());await second.start()
    assert second.accepting
    assert (await call(second,old_status))['state']=='succeeded'
    assert (await call(second,request))['state']=='succeeded'
    assert second.handler.calls==0

async def test_duplicate_changed_input_rejected(tmp_path):
    h=Handler();m=manager(tmp_path,h);await m.start();request=message(m)
    await call(m,request)
    request['_dingo_task']['payload']['request_digest']='d'*64
    with pytest.raises(ValueError):await call(m,request)
    h.release.set();await m.shutdown(2)

async def test_restart_nonterminal_is_not_free_capacity_without_idle_proof(tmp_path):
    from dingo.worker.runtime_io import atomic_json
    h=Handler();m=manager(tmp_path,h)
    identity=TaskIdentity('d','context_ir','p','task-0',1,'1'*32,'old','old-generation')
    root=m.attempt_root(identity)
    atomic_json(root/'accepted.json',{'identity':asdict(identity),'capacity_domain':'qwen-one'})
    atomic_json(root/'status.json',m.response(identity,'running'))
    idle=asyncio.Event()
    async def proof():return idle.is_set()
    m.engine_idle=proof
    await m.start();assert not m.accepting
    idle.set()
    await asyncio.wait_for(m.recovery_task,3)
    assert m.accepting
    assert root.joinpath('status.json').exists()

async def test_one_compiler_per_physical_qwen_domain(tmp_path):
    first=manager(tmp_path,Handler());second=manager(tmp_path,Handler())
    await first.start()
    with pytest.raises(RuntimeError,match="already has"):
        await second.start()
    await first.shutdown()
    await second.start()
    await second.shutdown()

# SPDX-License-Identifier: Apache-2.0
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from dingo.common.video_sr_only import SOURCE_KEY
from dingo.vllm.omni.sr_only_handler import validate_previous_attempt,SROnlyHandler
from dingo.vllm.omni.seedvr2_execution import UnsafeRecovery,write_journal,read_journal
from tests.video_gateway.test_seedvr2_adapter import source_fixture
OLD,NEW='a'*32,'b'*32
FIRST=Path('1-'+'c'*32);SECOND=Path('2-'+'d'*32)

@pytest.mark.parametrize('phase',['submit_intent','submitted'])
def test_new_attempt_new_service_does_not_rewrite_old_journal(phase):
    old={'owner':OLD,'phase':phase,'job_id':'old-job'};before=copy.deepcopy(old)
    validate_previous_attempt(old,FIRST,SECOND,NEW)
    assert old==before

@pytest.mark.parametrize('owner',[OLD,None,'','not-an-incarnation'])
def test_unsettled_same_or_unknown_native_service_is_rejected(owner):
    with pytest.raises(UnsafeRecovery):validate_previous_attempt({'owner':owner,'phase':'submitted'},FIRST,SECOND,OLD)

@pytest.mark.parametrize('previous',[SECOND,Path('3-'+'e'*32),Path('bad-name')])
def test_changed_service_alone_does_not_authorize_same_or_older_attempt(previous):
    with pytest.raises(UnsafeRecovery):validate_previous_attempt({'owner':OLD,'phase':'submitted'},previous,SECOND,NEW)

def test_unknown_state_fails_closed():
    with pytest.raises(UnsafeRecovery):validate_previous_attempt({'owner':OLD,'phase':'unknown'},FIRST,SECOND,NEW)

@pytest.mark.parametrize('phase',['completed','failed','cancelled'])
def test_settled_previous_attempt_does_not_block(phase):
    validate_previous_attempt({'owner':OLD,'phase':phase},FIRST,SECOND,OLD)

@pytest.mark.asyncio
async def test_retry_stages_original_input_without_changing_previous_attempt(tmp_path,monkeypatch):
    from dingo.vllm.omni import sr_only_handler as mod
    root,d,source=source_fixture(tmp_path);root.rename(root.parent/SECOND);root=root.parent/SECOND
    oldroot=root.parent/FIRST;oldroot.mkdir();journal=oldroot/'sr-journal.json'
    previous={'owner':OLD,'phase':'submitted','job_id':'old-job'};write_journal(journal,previous)
    handler=object.__new__(SROnlyHandler);handler.storage_id='test-store';handler.execution=SimpleNamespace(owner=NEW)
    d['media']={'width':1344,'height':768,'frames':107,'audio':False}
    async def probe(_):return {'copy_compatible':True,'width':1344,'height':768,'frames':107,'audio':None}
    monkeypatch.setattr(mod.media,'probe',probe)
    out,_,_=await handler._base({SOURCE_KEY:d},root,'fingerprint',SimpleNamespace(is_stopped=lambda:False))
    assert out.read_bytes()==source.read_bytes()
    assert read_journal(journal)==previous and not (root/'sr-journal.json').exists()

@pytest.mark.asyncio
async def test_late_failure_from_old_attempt_does_not_fail_current_attempt(make_gateway_config):
    from dataclasses import replace
    from dingo.video_gateway.models import TaskStatus
    from tests.video_gateway.test_task_store import _task
    from tests.video_gateway.test_dispatcher import FakeClient,_stack
    cfg=make_gateway_config()
    store,_,dispatcher,_=_stack(cfg,{'fl-pool':FakeClient()})
    current=_task('sr-retry',pool_id='fl-pool')
    current.status=TaskStatus.IN_PROGRESS;current.attempt=2
    current.owner_generation=dispatcher.generation;current.execution_token='b'*32
    current.worker_instance_id=9;current.worker_key='new-worker'
    old=replace(current,attempt=1,execution_token='a'*32,worker_instance_id=7,worker_key='old-worker')
    stored,_=await store.create_task(current,principal_hash='p',idempotency_hash='i',queue_limit=10)
    await dispatcher._finish_failed(dispatcher.pools['fl-pool'],current.id,'worker_failed','late old response',
                                    quarantine=True,retryable=True,expected_execution=old)
    after=await store.get_task(current.id)
    assert after.revision==stored.revision and after.task.status==TaskStatus.IN_PROGRESS
    assert after.task.attempt==2 and after.task.execution_token=='b'*32 and after.task.error is None

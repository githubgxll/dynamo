# SPDX-License-Identifier: Apache-2.0
"""Uploaded usage remains stable through task persistence and public HTTP routes."""
import asyncio
import copy
import pytest
from dingo.video_gateway.models import VideoTask, TaskStatus, input_usage
from tests.video_gateway.test_task_store import _task
from tests.video_gateway.test_api import _client, _form, _wait_completed


def test_uploaded_counts_survive_retries_terminal_states_and_storage():
    task = _task('video-usage')
    task.normalized_request = {'task':'ref2va','seconds':15,
        'reference_kinds':['image','image','video','video','audio'],
        'reference_metadata':[{'kind':'image'},{'kind':'image'},
          {'kind':'video','duration_s':4.25,'frame_count':102},
          {'kind':'video','duration_s':6.5,'frame_count':156},
          {'kind':'audio','duration_s':12}]}
    original=copy.deepcopy(task.normalized_request)
    for status in TaskStatus:
        task.status=status;task.attempt+=1
        task.normalized_request['_result_media']={'duration_s':30,'frames':720,'fps':24}
        restored=VideoTask.from_dict(task.to_dict()).public_dict()
        assert restored['task']=='ref2va'
        assert restored['input_usage']=={'image_count':2,'video_duration_seconds':10.75}
    assert {k:v for k,v in task.normalized_request.items() if k!='_result_media'}==original


@pytest.mark.parametrize('value',[None,-1,True,'3',float('nan'),float('inf'),10**1000])
def test_unknown_video_duration_is_not_zero_or_partial(value):
    request={'reference_kinds':['image','video','video'],
             'reference_metadata':[{'kind':'image'},{'kind':'video','duration_s':3},{'kind':'video','duration_s':value}]}
    assert input_usage(request)=={'image_count':1,'video_duration_seconds':None}


def test_absence_is_zero_but_missing_history_is_unknown():
    assert input_usage({})=={'image_count':None,'video_duration_seconds':None}
    assert input_usage({'reference_kinds':[]})=={'image_count':0,'video_duration_seconds':0}
    assert input_usage({'reference_metadata':[]})=={'image_count':0,'video_duration_seconds':0}
    assert input_usage({'reference_kinds':['video']})['video_duration_seconds'] is None
    assert input_usage({'reference_metadata':[{'kind':'image'}]})['image_count']==1
    assert _task('legacy').public_dict()['task'] is None


@pytest.mark.parametrize('task_type',['t2va','fl2va','ref2va'])
def test_type_uses_persisted_task_not_public_model_name(task_type):
    task=_task('video-type');task.model='arbitrary-public-name'
    task.normalized_request={'task':task_type,'workflow':'fl2va','reference_kinds':[]}
    assert task.public_dict()['task']==task_type


async def test_create_query_list_and_idempotent_replay_share_metadata(make_gateway_config):
    client=await _client(make_gateway_config)
    try:
        response=await client.post('/v1/videos',data=_form(),headers={'Idempotency-Key':'input-usage'})
        assert response.status==202
        admitted=await response.json();id=admitted['id']
        expected={'image_count':0,'video_duration_seconds':0}
        assert admitted['task']=='t2va' and admitted['input_usage']==expected
        _,done=await _wait_completed(client,id)
        assert done['task']=='t2va' and done['input_usage']==expected
        listing=await (await client.get('/v1/videos')).json()
        listed=next(t for t in listing['data'] if t['id']==id)
        assert listed['task']=='t2va' and listed['input_usage']==expected
        replay=await (await client.post('/v1/videos',data=_form(),headers={'Idempotency-Key':'input-usage'})).json()
        assert replay['id']==id and replay['task']=='t2va' and replay['input_usage']==expected
    finally:await client.close()

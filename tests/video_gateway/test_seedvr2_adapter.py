# SPDX-License-Identifier: Apache-2.0
"""SR-only upload contracts, actual media probing and safe shared-file staging."""
import dataclasses
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from dingo.common.video_sr_only import SOURCE_KEY, validate_sr_only
from dingo.common.video_super_resolution import SR_PAYLOAD_KEY
from dingo.video_gateway.adapters import create_adapter
from dingo.video_gateway.adapters.base import UploadedArtifact
from dingo.video_gateway.errors import GatewayError
from dingo.vllm.omni.sr_only_handler import stage_source


def adapter(make_gateway_config):
    p=make_gateway_config().pools[0]
    p=dataclasses.replace(p,execution_mode='detached',adapter=dataclasses.replace(
        p.adapter,name='seedvr2',workflow='sr_only',options={'storage_id':'test-store'}))
    return create_adapter(p)


def video(tmp_path, *, size=(1344,768), fps=24, frames=2):
    import av
    path=tmp_path/'source.mp4'
    with av.open(str(path),'w') as c:
        s=c.add_stream('libx264',rate=fps);s.width,s.height=size;s.pix_fmt='yuv420p'
        s.options={'preset':'ultrafast'}
        for _ in range(frames):
            for packet in s.encode(av.VideoFrame.from_image(Image.new('RGB',size,'blue'))):c.mux(packet)
        for packet in s.encode():c.mux(packet)
    data=path.read_bytes()
    return UploadedArtifact('input_references',0,path.name,'video/mp4',path,len(data),hashlib.sha256(data).hexdigest())


def fields(**extra):
    return {k:[v] for k,v in {'model':'public-sr','operation':'sr_only','resolution':'1080P',**extra}.items()}


@pytest.mark.parametrize('tier,size',[('1080P',(1920,1080)),('2K',(2560,1440))])
def test_sr_only_media_and_shared_payload(make_gateway_config,tmp_path,tier,size):
    a=adapter(make_gateway_config);u=video(tmp_path)
    n=a.normalize_request(fields(resolution=tier),[u],'public-sr')
    assert (n['width'],n['height'])==size
    assert n['num_frames']==2 and n['generate_sound'] is False
    assert 'seed' not in n and 'prompt' not in n
    manifest=[{'ordinal':0,'path':'inputs/source.mp4','content_type':'video/mp4','bytes':u.size,'sha256':u.sha256}]
    (tmp_path/'input-manifest.json').write_text(json.dumps(manifest))
    payload=a.build_worker_payload(n,manifest,tmp_path)
    assert SOURCE_KEY in payload and 'input_reference' not in payload
    assert len(json.dumps(payload))<4096
    assert validate_sr_only(payload)['output_width']==size[0]
    assert a.pool.sr_profile=='minimax_h3_seedvr2.v3'


@pytest.mark.parametrize('key,value',[('prompt','hello'),('seed','1'),('seconds','4'),('size','1920x1080'),('num_inference_steps','50')])
def test_sr_only_rejects_generation_parameters(make_gateway_config,tmp_path,key,value):
    with pytest.raises(GatewayError,match='generation parameters'):
        adapter(make_gateway_config).normalize_request(fields(**{key:value}),[video(tmp_path)],'public-sr')


@pytest.mark.parametrize('fps', [30, Fraction(24001, 1000)])
def test_sr_only_rejects_bad_fps(make_gateway_config,tmp_path,fps):
    with pytest.raises(GatewayError,match='24 FPS'):
        adapter(make_gateway_config).normalize_request(fields(),[video(tmp_path,fps=fps)],'public-sr')


def source_fixture(tmp_path):
    task=tmp_path/'task';(task/'inputs').mkdir(parents=True)
    src=task/'inputs/source.mp4';src.write_bytes(b'test source bytes')
    entry={'ordinal':0,'path':'inputs/source.mp4','content_type':'video/mp4','bytes':src.stat().st_size,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
    raw=json.dumps([entry]).encode();(task/'input-manifest.json').write_bytes(raw)
    root=task/'attempts/1-token';root.mkdir(parents=True)
    descriptor={'schema_version':1,'storage_id':'test-store','manifest_sha256':hashlib.sha256(raw).hexdigest(),'media':{},'entry':entry}
    return root,descriptor,src


def test_source_is_task_scoped_and_verified(tmp_path):
    root,d,src=source_fixture(tmp_path)
    target=stage_source(root,d,'test-store')
    assert target.read_bytes()==src.read_bytes()


def test_source_symlink_is_rejected(tmp_path):
    root,d,src=source_fixture(tmp_path)
    outside=tmp_path/'outside.mp4';outside.write_bytes(src.read_bytes());src.unlink();src.symlink_to(outside)
    with pytest.raises(OSError):stage_source(root,d,'test-store')


def test_source_changed_after_upload_is_rejected(tmp_path):
    root,d,src=source_fixture(tmp_path);src.write_bytes(b'wrong file bytes!')
    with pytest.raises(ValueError):stage_source(root,d,'test-store')


def test_source_cannot_cross_tasks(tmp_path):
    root,d,src=source_fixture(tmp_path);d['entry']['path']='../source.mp4'
    with pytest.raises(ValueError,match='task inputs'):stage_source(root,d,'test-store')


def test_sr_only_public_task_has_operation_without_generation_seed():
    from tests.video_gateway.test_task_store import _task
    task = _task('sr-task')
    task.normalized_request = {'operation': 'sr_only', 'width': 1920, 'height': 1080}
    public = task.public_dict()
    assert public['operation'] == 'sr_only'
    assert 'seed' not in public and 'seed_generated' not in public


async def test_sr_only_http_validation_and_no_worker(make_gateway_config, tmp_path):
    import aiohttp
    from aiohttp.test_utils import TestClient, TestServer
    from dingo.video_gateway.app import create_app
    from tests.video_gateway.test_dispatcher import FakeClient, _stack

    config = make_gateway_config()
    pool = adapter(make_gateway_config).pool
    pool = dataclasses.replace(pool, scheduling=dataclasses.replace(
        pool.scheduling, accept_without_workers=False))
    config = dataclasses.replace(config, pools=[pool])
    _, _, _, service = _stack(config, {pool.pool_id: FakeClient(available=False)})
    upload = video(tmp_path)
    async with TestClient(TestServer(create_app(service))) as client:
        for operation, expected in [('generate', 400), ('sr_only', 503)]:
            form = aiohttp.FormData()
            for name, value in {'model':pool.served_models[0], 'operation':operation,
                                'resolution':'1080P'}.items():
                form.add_field(name, value)
            form.add_field('input_references', upload.path.read_bytes(),
                           filename='source.mp4', content_type='video/mp4')
            response = await client.post('/v1/videos', data=form)
            body = await response.json()
            assert response.status == expected, body
        assert (await (await client.get('/v1/videos')).json())['data'] == []


@pytest.mark.parametrize('quarantined,alive', [(True, True), (False, False)])
async def test_sr_supervisor_exits_after_failure_grace(quarantined, alive):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from dingo.vllm.sr_worker import supervise_worker
    endpoint = SimpleNamespace(unregister_endpoint_instance=AsyncMock())
    with pytest.raises(RuntimeError, match='restarting Worker'):
        await supervise_worker(SimpleNamespace(healthy=lambda: alive),
            SimpleNamespace(quarantined=quarantined), endpoint, asyncio.Event(),
            grace_s=0, interval_s=.001)
    endpoint.unregister_endpoint_instance.assert_awaited_once()


async def test_sr_supervisor_does_not_restart_cancellation_pause():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from dingo.vllm.sr_worker import supervise_worker
    stop = asyncio.Event()
    endpoint = SimpleNamespace(unregister_endpoint_instance=AsyncMock())
    handler = SimpleNamespace(quarantined=False, video_ready=False)
    watcher = asyncio.create_task(supervise_worker(
        SimpleNamespace(healthy=lambda: True), handler, endpoint, stop,
        grace_s=0, interval_s=.001))
    try:
        await asyncio.sleep(.01)
        assert not watcher.done()
        handler.video_ready = True
        stop.set()
        await asyncio.wait_for(watcher, timeout=1)
        endpoint.unregister_endpoint_instance.assert_not_awaited()
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)

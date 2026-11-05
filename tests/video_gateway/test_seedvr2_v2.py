import ast,dataclasses,json
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import AsyncMock
import pytest
from aiohttp import web
from dingo.common.video_super_resolution import H3_SR_PROFILE, H3_SR_PROFILE_V2, configured_sr_profile, negotiated_sr_profiles
from dingo.video_gateway.adapters.h3_output_policy import plan_h3_output
from dingo.vllm.omni.seedvr2_handler import validate_sr_request
from dingo.vllm.omni import seedvr2_media
from tests.video_gateway.test_seedvr2_client import server,JOB

@pytest.mark.parametrize('output,internal,window',[(1800,(1840,1104),33),(2400,(2400,1440),65)])
def test_v2_geometry(output,internal,window):
    height=1080 if output==1800 else 1440
    p=plan_h3_output(output,height,sr_profile=H3_SR_PROFILE_V2)
    assert (p.base_width,p.base_height)==(1280,768)
    assert (p.sr_width,p.sr_height)==internal
    sr={'schema_version':1,'profile':p.sr_profile,'base_width':p.base_width,'base_height':p.base_height,'output_width':output,'output_height':height,'sr_width':p.sr_width,'sr_height':p.sr_height}
    payload={'dingo_super_resolution':sr,'size':'1280x768','nvext':{'fps':24}}
    assert validate_sr_request(payload,profile=H3_SR_PROFILE_V2)==sr
    with pytest.raises(ValueError):validate_sr_request(payload,profile=H3_SR_PROFILE)
    with pytest.raises(ValueError):plan_h3_output(output,height,sr_profile=H3_SR_PROFILE)
    assert configured_sr_profile('minimax_h3',{'super_resolution':{'enabled':True,'profile':H3_SR_PROFILE_V2}})==H3_SR_PROFILE_V2

@pytest.mark.asyncio
async def test_single_encode_multipart_and_metrics(tmp_path):
    seen=[];polls=0
    async def handle(request):
        nonlocal polls
        if request.method=='POST':
            form=await request.post();seen.append(dict(form))
            return web.json_response({'id':JOB,'status':'queued'},status=202)
        polls+=1
        if polls==1:return web.Response(status=404)
        return web.json_response({'metrics_status':'PASS_COMPLETE'})
    source=tmp_path/'base.mp4';source.write_bytes(b'base')
    options={'window_frames':33,'delivery_size':'1800x1080','encoding_mode':'single_encode','measurement_mode':'basic','measurement_context':{'run_id':'test-real-identity'}}
    async with server(handle) as c:
        await c.submit(source,width=1840,height=1104,frames=158,seed=7723,options=options)
        assert (await c.metrics(JOB))['metrics_status']=='PASS_COMPLETE'
    assert seen[0]['window_frames']=='33' and seen[0]['delivery_size']=='1800x1080'
    assert seen[0]['encoding_mode']=='single_encode' and seen[0]['seed']=='7723'
    assert json.loads(seen[0]['measurement_context'])==options['measurement_context']

@pytest.mark.asyncio
async def test_single_encode_never_falls_back_to_encoding(monkeypatch,tmp_path):
    base={'frames':158,'width':1280,'height':768,'audio':False}
    raw={'frames':158,'width':1840,'height':1104,'audio':False,'copy_compatible':True}
    monkeypatch.setattr(seedvr2_media,'probe',AsyncMock(side_effect=[base,raw]))
    command=AsyncMock();monkeypatch.setattr(seedvr2_media,'command',command)
    with pytest.raises(ValueError,match='stream copy'):
        await seedvr2_media.finalize(tmp_path/'raw',tmp_path/'base',tmp_path/'out',width=1800,height=1080,require_stream_copy=True)
    command.assert_not_awaited()

def test_offload_kwargs_are_forwarded_without_removed_keyword():
    root=Path(__file__).resolve().parents[2]
    source=ast.parse((root/'dingo/vllm/omni/base_handler.py').read_text())
    cls=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name=='BaseOmniHandler')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_build_omni_kwargs')
    env={'dataclasses':dataclasses,'Dict':dict,'Any':object,'DiffusionParallelConfig':None,'logger':N(warning=lambda *a:None)}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'actual_build_kwargs','exec'),env)
    C=dataclasses.make_dataclass('Options',[('diffusion_offload_config',dict),('enable_cpu_offload',bool),('enable_layerwise_offload',bool),('layerwise_num_gpu_layers',int)])
    value={'mode':'module','components':['dit','text_encoder'],'pin_memory':True}
    cfg=N(model='h3',engine_args=N(trust_remote_code=True,disable_log_stats=False),output_modalities=['video'],stage_configs_path=None,diffusion=C(value,False,False,1))
    result=env['_build_omni_kwargs'](None,cfg)
    assert result['diffusion_offload_config']==value and 'layerwise_num_gpu_layers' not in result
    cfg.diffusion.enable_cpu_offload=True
    with pytest.raises(ValueError,match='conflicts'):env['_build_omni_kwargs'](None,cfg)


def test_measurement_declares_all_native_rank_sources(tmp_path):
    import hashlib
    from dingo.vllm.omni.seedvr2_process import measurement_source_hashes
    sources = ["vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py",
               "vllm_omni/diffusion/models/seedvr2/sr_telemetry.py",
               "vllm_omni/diffusion/ipc.py",
               "vllm_omni/diffusion/models/seedvr2/long_video.py",
               "dingo/vllm/omni/seedvr2_handler.py"]
    for name in sources:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    declared=measurement_source_hashes(tmp_path)
    for basename in ["pipeline_seedvr2.py","sr_telemetry.py","ipc.py"]:
        matches=[(key,value) for key,value in declared.items() if key.endswith("/"+basename)]
        assert len(matches)==1
        key,value=matches[0];assert value==hashlib.sha256((tmp_path/key).read_bytes()).hexdigest()
    (tmp_path/"vllm_omni/diffusion/ipc.py").unlink()
    with pytest.raises(FileNotFoundError):measurement_source_hashes(tmp_path)

import importlib.util,json,sys
from contextlib import nullcontext
from pathlib import Path
import pytest
from dingo.common.video_super_resolution import H3_SR_PROFILE_V3,H3_SR_PROFILE_V2,negotiated_sr_profiles,SR_CAPABILITY
from dingo.video_gateway.adapters.h3_output_policy import plan_h3_output,H3_SR_V3_PRESETS
from dingo.video_gateway.adapters.h3_shape import resolve_output_shape
from dingo.vllm.omni.seedvr2_handler import validate_sr_request
from dingo.vllm.omni import seedvr2_media

@pytest.mark.parametrize('output',list(H3_SR_V3_PRESETS))
def test_v3_fixed_geometry_and_worker_payload(output):
 p=plan_h3_output(*output,sr_profile=H3_SR_PROFILE_V3)
 assert p.sr_profile==H3_SR_PROFILE_V3
 assert p.base_width*p.base_height<=1344*768
 if (p.base_width,p.base_height)!=(1280,768):resolve_output_shape(p.base_width,p.base_height,requested_aspect_ratio=p.aspect_ratio)
 assert p.sr_width%16==p.sr_height%16==0
 assert p.sr_width*p.base_height==p.sr_height*p.base_width
 assert p.sr_width>=output[0] and p.sr_height>=output[1]
 assert p.sr_width*p.sr_height<=3850240
 assert p.window_frames==H3_SR_V3_PRESETS[output][4]
 if p.window_frames==65:
  assert output[0]%32==output[1]%32==0
 sr={'schema_version':1,'profile':p.sr_profile,'base_width':p.base_width,'base_height':p.base_height,'output_width':p.output_width,'output_height':p.output_height,'sr_width':p.sr_width,'sr_height':p.sr_height}
 payload={'dingo_super_resolution':sr,'size':f'{p.base_width}x{p.base_height}','nvext':{'fps':24}}
 assert validate_sr_request(payload,profile=H3_SR_PROFILE_V3)==sr
 with pytest.raises(ValueError):validate_sr_request(payload,profile=H3_SR_PROFILE_V2)

@pytest.mark.parametrize('size',[(2048,1080),(2440,1440),(2560,1438)])
def test_unknown_outputs_rejected(size):
 with pytest.raises(ValueError):plan_h3_output(*size,sr_profile=H3_SR_PROFILE_V3)

def test_v3_requires_matching_worker_and_preserves_legacy():
 response={'schema_version':1,'accepting':True,'capabilities':[SR_CAPABILITY],'super_resolution':{'ready':True,'profiles':[H3_SR_PROFILE_V2]}}
 assert H3_SR_PROFILE_V3 not in negotiated_sr_profiles(response)
 for size in ((1800,1080),(2400,1440)):
  old=plan_h3_output(*size,sr_profile=H3_SR_PROFILE_V2);new=plan_h3_output(*size,sr_profile=H3_SR_PROFILE_V3)
  assert (old.base_width,old.base_height,old.sr_width,old.sr_height)==(new.base_width,new.base_height,new.sr_width,new.sr_height)

@pytest.mark.asyncio
async def test_finalizer_only_allows_explicit_base_geometry_and_stream_copy(monkeypatch,tmp_path):
 base,raw,out=(tmp_path/n for n in ('base.mp4','raw.mp4','out.mp4'))
 info={'frames':107,'audio':None,'start_time':'0','copy_compatible':True,'probe_timings':{}}
 original=dict(info,width=1344,height=768);final=dict(info,width=1920,height=1080)
 async def probe(path):return original if path==base else final
 calls=[]
 async def command(args):calls.append(args)
 monkeypatch.setattr(seedvr2_media,'probe',probe);monkeypatch.setattr(seedvr2_media,'command',command)
 with pytest.raises(ValueError):await seedvr2_media.finalize(raw,base,out,width=1920,height=1080,require_stream_copy=True)
 with pytest.raises(ValueError):await seedvr2_media.finalize(raw,base,out,width=1920,height=1080,require_stream_copy=True,expected_base_size=(1280,768))
 result=await seedvr2_media.finalize(raw,base,out,width=1920,height=1080,require_stream_copy=True,expected_base_size=(1344,768))
 assert result['postprocess_mode']=='stream_copy'
 assert calls[0][calls[0].index('-c:v')+1]=='copy' and '-vf' not in calls[0]

def test_v3_provenance_includes_crop_and_policy_without_changing_v2(tmp_path):
 from dingo.vllm.omni.seedvr2_process import measurement_source_hashes
 base=['vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py','vllm_omni/diffusion/models/seedvr2/sr_telemetry.py','vllm_omni/diffusion/ipc.py','vllm_omni/diffusion/models/seedvr2/long_video.py','dingo/vllm/omni/seedvr2_handler.py']
 extra=['vllm_omni/diffusion/models/seedvr2/long_video_support.py','dingo/video_gateway/adapters/h3_output_policy.py','dingo/common/video_super_resolution.py','dingo/vllm/omni/seedvr2_media.py']
 for name in base+extra:
  p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
 assert set(measurement_source_hashes(tmp_path))==set(base)
 before=measurement_source_hashes(tmp_path,include_geometry=True)
 assert set(before)==set(base+extra)
 (tmp_path/extra[0]).write_text('changed crop implementation')
 assert measurement_source_hashes(tmp_path,include_geometry=True)[extra[0]]!=before[extra[0]]

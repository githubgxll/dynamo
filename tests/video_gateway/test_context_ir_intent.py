# SPDX-License-Identifier: Apache-2.0
"""Compiler decisions must survive Gateway transport and native video handoff."""
import base64
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from dingo.video_gateway.adapters.base import UploadedArtifact
from dingo.video_gateway.context_ir.validation import normalize
from dingo.video_gateway.context_ir.protocol import prepare, validate_result, public_result
from dingo.video_gateway.context_ir.client import UpstreamRejected
from tests.video_gateway.test_context_ir_resolution import compiler_service, fields, image
pytest.importorskip('h3ir')
pytest.importorskip('fastapi')

from h3ir import engine
from h3ir.intent_resolution import resolve
from h3ir.models import Mode
from h3ir.grid import Target
from h3ir.plan import build_manifest
from h3ir.service import BriefIn, _to_brief


class Model:
    def __init__(self,rows):self.rows=rows;self.calls=0
    def chat(self,*args,**kwargs):
        self.calls+=1
        return SimpleNamespace(content=json.dumps({'decisions':self.rows}))


def decision(index,value,evidence='',field='role'):
    return {'index':index,'field':field,'value':value,'evidence':evidence}


def compile_contract(normalized,root,rows=()):
    payload,sources,expected=prepare(normalized,root,root/'attempt',64*1024**2)
    token=engine._sources.set(sources)
    try:brief=_to_brief(BriefIn.model_validate(payload))
    finally:engine._sources.reset(token)
    model=Model(rows);resolved,record=resolve(brief,model)
    target=Target.build(resolved.seconds,resolved.aspect,resolved.canvas,resolved.megapixels)
    manifest=[asdict(m) for m in build_manifest(resolved,target)]
    response={'status':'ready','source':'written','ir':{
        'prompt':'Valid compiled prompt.','mode':resolved.mode.value if resolved.mode else normalized['mode'],
        'target':{'frames':target.frames,'canvas':list(target.canvas)},'manifest':manifest,
        'provenance':{'intent_resolution':record}}}
    return response,expected,payload,model


def audio(root,ordinal=1):
    p=root/'inputs'/f'audio-{ordinal}.wav';p.parent.mkdir(exist_ok=True,parents=True)
    with wave.open(str(p),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(32000);stream.writeframes(b'\0\0'*32000*3)
    return UploadedArtifact('input_references',ordinal,p.name,'audio/wav',p,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest())


@pytest.mark.parametrize('tail',[False,True])
def test_prompt_single_anchor_round_trip(make_gateway_config,tmp_path,tail):
    service,adapter=compiler_service(make_gateway_config)
    text='以图1结束。' if tail else '从图1开始。'
    uploads=[image(tmp_path,(1344,768))]
    n=normalize(service,fields(prompt=text),uploads)
    r,expected,_,model=compile_contract(n,tmp_path,[decision(0,'frame_anchor_last' if tail else 'frame_anchor_first',text[:-1])])
    result=public_result(validate_result(n,expected,200,r))
    assert result['generation_parameters']['frame_indices']==([-1] if tail else [0])
    assert result['references'][0]['role']==('frame_anchor_last' if tail else 'frame_anchor_first')
    assert result['generation_parameters']['ratio']=='adaptive' and model.calls==1


def test_two_reversed_images_select_first_shape_and_native_order(make_gateway_config,tmp_path):
    service,adapter=compiler_service(make_gateway_config)
    uploads=[image(tmp_path,(1344,768)),image(tmp_path,(768,1344),1)]
    n=normalize(service,fields(prompt='从图2开始，以图1结束。'),uploads)
    rows=[decision(0,'frame_anchor_last','以图1结束'),decision(1,'frame_anchor_first','从图2开始')]
    r,expected,_,_=compile_contract(n,tmp_path,rows)
    assert r['ir']['target']['canvas']==[768,1344]
    result=public_result(validate_result(n,expected,200,r))
    assert result['generation_parameters']['frame_indices']==[-1,0]
    assert result['references'][0]['labels']==['<Picture 2>']
    assert result['references'][1]['labels']==['<Picture 1>']
    params={'model':'public-fl','prompt':result['content']['prompt'],'resolution':'768P',**result['generation_parameters']}
    vf={k:[json.dumps(v) if isinstance(v,list) else str(v)] for k,v in params.items()}
    v=adapter.normalize_request(vf,uploads,'public-fl')
    assert [v['width'],v['height']]==[768,1344]
    manifest=[u.manifest_entry(relative_to=tmp_path) for u in uploads]
    payload=adapter.build_worker_payload(v,manifest,tmp_path)
    values=json.loads(payload['input_reference'])
    assert base64.b64decode(values[0].split(',',1)[1])==uploads[1].path.read_bytes()
    assert base64.b64decode(values[1].split(',',1)[1])==uploads[0].path.read_bytes()


def test_audio_missing_role_is_accepted_and_defaulted(make_gateway_config,tmp_path):
    service,_=compiler_service(make_gateway_config,workflow='ref2va')
    uploads=[image(tmp_path,(768,768)),audio(tmp_path)]
    n=normalize(service,fields(ratio='16:9'),uploads)
    r,e,_,m=compile_contract(n,tmp_path)
    result=validate_result(n,e,200,r)
    assert result['references'][1]['role']=='voice_timbre'
    assert any(w['code']=='intent_resolution_defaulted' for w in result['warnings'])


def test_audio_prompt_bgm_and_explicit_voice_override(make_gateway_config,tmp_path):
    service,_=compiler_service(make_gateway_config,workflow='ref2va')
    uploads=[image(tmp_path,(768,768)),audio(tmp_path)]
    n=normalize(service,fields(ratio='16:9',prompt='音频1作为背景音乐。'),uploads)
    r,e,_,_=compile_contract(n,tmp_path,[decision(1,'bgm','音频1作为背景音乐')])
    assert validate_result(n,e,200,r)['references'][1]['role']=='bgm'
    explicit=normalize(service,fields(ratio='16:9',prompt='音频1作为背景音乐。',reference_options='[{"index":1,"role":"voice_timbre"}]'),uploads)
    r,e,_,m=compile_contract(explicit,tmp_path)
    assert m.calls==0 and validate_result(explicit,e,200,r)['references'][1]['role']=='voice_timbre'


@pytest.mark.parametrize('change',['missing_record','wrong_hash','changed_explicit','invalid_mode','wrong_canvas','wrong_label'])
def test_intent_contract_rejects_mismatches(make_gateway_config,tmp_path,change):
    service,_=compiler_service(make_gateway_config)
    n=normalize(service,fields(reference_options='[{"index":0,"role":"frame_anchor_last"}]'),[image(tmp_path,(1344,768))])
    r,e,_,_=compile_contract(n,tmp_path)
    ir=r['ir']
    if change=='missing_record':ir['provenance']={}
    elif change=='wrong_hash':ir['manifest'][0]['sha256']='0'*64
    elif change=='changed_explicit':ir['provenance']['intent_resolution']['decisions'][0]['value']='frame_anchor_first'
    elif change=='invalid_mode':ir['mode']='ref2va'
    elif change=='wrong_canvas':ir['target']['canvas']=[768,768]
    elif change=='wrong_label':ir['manifest'][0]['label']='<Picture 2>'
    with pytest.raises(UpstreamRejected):validate_result(n,e,200,r)


@pytest.mark.parametrize('policy,expected_policy',[(None,'preserve'),('ignore','ignore'),('preserve','preserve')])
def test_video_soundtrack_default_and_explicit_contract(make_gateway_config,tmp_path,policy,expected_policy):
    import subprocess
    service,_=compiler_service(make_gateway_config,workflow='ref2va')
    p=tmp_path/'inputs/video.mp4';p.parent.mkdir(exist_ok=True)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','color=c=blue:s=512x288:r=24',
                    '-f','lavfi','-i','sine=frequency=440:sample_rate=32000','-t','3','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(p)],check=True)
    u=UploadedArtifact('input_references',0,p.name,'video/mp4',p,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest())
    extra={'ratio':'16:9'}
    if policy is not None:extra['reference_options']=json.dumps([{'index':0,'soundtrack_policy':policy}])
    n=normalize(service,fields(**extra),[u])
    r,e,_,m=compile_contract(n,tmp_path)
    assert r['ir']['provenance']['intent_resolution']['decisions'][0]['value']==expected_policy
    marker='weak_reference' if expected_policy=='ignore' else 'partially_copy'
    r['ir']['prompt']=f'<Audio 1>: {marker} - source soundtrack policy.'
    assert validate_result(n,e,200,r)['references'][0]['kind']=='video'
    if policy is None:assert m.calls==1
    else:assert m.calls==0
    r['ir']['prompt']='<Audio 1>: fully_copy - source soundtrack.' if expected_policy=='ignore' else '<Audio 1>: weak_reference - source soundtrack.'
    with pytest.raises(UpstreamRejected):validate_result(n,e,200,r)

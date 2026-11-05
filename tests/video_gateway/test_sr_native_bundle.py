# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source lock integrity and independent RGB crop checks for the shipped bundle."""
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'container/deps/vllm/patches/h3_seedvr2_v3'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_all_reviewed_source_bytes_and_idempotent_installer(tmp_path):
    lock=json.loads((BUNDLE/'source-lock.json').read_text())
    assert len(lock['files'])==13
    for name in ['vllm','vllm-omni']:
        meta=tmp_path/(name.replace('-','_')+'-0.32.0.dist-info');meta.mkdir()
        (meta/'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: 0.32.0\n')
    for e in lock['files']:
        data=(BUNDLE/'sources'/e['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==e['after_sha256']
        compile(data,e['path'],'exec')
        target=tmp_path/e['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    installer=load('_bundle_install',BUNDLE/'apply.py')
    assert installer.apply(tmp_path,check=True)==installer.apply(tmp_path)==0
    target=tmp_path/lock['files'][0]['path'];target.write_text('unknown source')
    with pytest.raises(ValueError,match='base checksum mismatch'):installer.apply(tmp_path)


def presets():
    tree=ast.parse((ROOT/'dingo/video_gateway/adapters/h3_output_policy.py').read_text())
    return next(ast.literal_eval(n.value.args[0]) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='H3_SR_V3_PRESETS' for t in n.targets)).items()


@pytest.mark.parametrize('output,dimensions',list(presets()))
def test_native_delivery_matches_gateway_plan_and_center_pixels(output,dimensions):
    av=pytest.importorskip('av');np=pytest.importorskip('numpy')
    module=load('_native_delivery',BUNDLE/'sources/vllm_omni/diffusion/models/seedvr2/long_video_support.py')
    dw,dh=output;bw,bh,sw,sh,window,ratio=dimensions
    config=module.options('single_encode',f'{dw}x{dh}',window)
    module.validate_source(config,bw,bh,sw,sh)
    image=np.random.default_rng(42).integers(0,256,(sh,sw,3),dtype=np.uint8)
    metrics=type('Metrics',(),{'stage':lambda *a,**kw:nullcontext()})()
    actual=module.delivery_array(image,dw,dh,metrics).to_ndarray(format='rgb24')
    writer=module.writer_spec(config,sw,sh)
    assert actual.shape==(dh,dw,3)
    if output not in [(1920,1080),(1080,1920)]:
        x,y=(sw-dw)//2,(sh-dh)//2
        assert np.array_equal(actual,image[y:y+dh,x:x+dw])
        assert not writer['requires_resize']
    else:
        previous=module.delivery_frame(av.VideoFrame.from_ndarray(image,format='rgb24'),dw,dh,metrics).to_ndarray(format='rgb24')
        assert np.array_equal(actual,previous) and writer['requires_resize']


def test_documented_v3_canvases_and_delivery_rules_match_native_writer():
    """Keep the public matrix aligned with both planning and actual delivery."""
    doc=(ROOT/'docs/video-gateway/super-resolution.md').read_text()
    section=doc.split('## 1080P 白名单',1)[1].split('## 裁剪和缩放规则',1)[0]
    actual={}
    tier_window=33
    for line in section.splitlines():
        if line.startswith('## 2K 白名单'):
            tier_window=65
        if not line.startswith('| '):
            continue
        columns=[c.strip() for c in line.strip('|').split('|')]
        if len(columns)!=6 or not columns[1][0].isdigit():
            continue
        output=tuple(map(int,columns[1].split('x')))
        assert output not in actual
        actual[output]=(tuple(map(int,columns[2].split('x'))),tuple(map(int,columns[3].split('x'))),tuple(map(int,columns[4].split(','))),columns[5],tier_window)
    expected=dict(presets())
    assert set(actual)==set(expected)
    native=load('_document_native_writer',BUNDLE/'sources/vllm_omni/diffusion/models/seedvr2/long_video_support.py')
    for output,row in expected.items():
        base,internal,crop,resize,window=actual[output]
        assert base==tuple(row[:2]) and internal==tuple(row[2:4]) and window==row[4]
        writer=native.writer_spec({'delivery_size':output,'encoding_mode':'single_encode'},*internal)
        assert crop==tuple(writer['crop_box'])
        assert resize==('Lanczos 至 '+'x'.join(map(str,output)) if writer['requires_resize'] else '不缩放')
    api=(ROOT/'docs/video-gateway/video-gateway-api.md').read_text()
    assert '(super-resolution.md)' in api
    assert 'path: video-gateway/super-resolution.md' in (ROOT/'docs/index.yml').read_text()

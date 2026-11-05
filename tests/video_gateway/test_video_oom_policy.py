# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SR OOM status survives the loopback boundary without device details."""
import ast
import json
import os
from pathlib import Path

import pytest

SOURCE=Path(__file__).resolve().parents[2]/'container/deps/vllm/patches/h3_seedvr2_v3/sources/vllm_omni/diffusion/models/seedvr2/long_video.py'

def native_status_functions():
    tree=ast.parse(SOURCE.read_text())
    names={'_is_oom','JobOOMError','_status'}
    module=ast.Module(body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names],type_ignores=[])
    namespace={'JobError':RuntimeError,'Path':Path,'json':json,'os':os}
    exec(compile(module,str(SOURCE),'exec'),namespace)
    return namespace

@pytest.mark.parametrize('cause',[MemoryError(),RuntimeError('CUDA out of memory. GPU private details')])
def test_native_oom_cause_chain_becomes_bounded_metadata(tmp_path,cause):
    native=native_status_functions()
    wrapped=RuntimeError('native restoration failed');wrapped.__cause__=cause
    assert native['_is_oom'](wrapped)
    native['_status'](tmp_path,'failed',0,'native restoration failed',oom=native['_is_oom'](wrapped))
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['internal_error']=={'schema_version':1,'reason':'resource_exhausted','stage':'super_resolution'}
    assert 'private details' not in json.dumps(status)

def test_generic_failure_is_not_misclassified_as_oom(tmp_path):
    native=native_status_functions()
    assert not native['_is_oom'](ValueError('invalid video'))
    native['_status'](tmp_path,'failed',0,'invalid video')
    assert 'internal_error' not in json.loads((tmp_path/'status.json').read_text())

@pytest.mark.parametrize('before,after,is_oom',[(0,1,True),(7,8,True),(7,7,False)])
def test_h3_stable_abi_error_uses_request_oom_delta(before,after,is_oom):
    from types import SimpleNamespace as N
    from unittest.mock import Mock
    source=SOURCE.parents[1]/'minimax_h3/pipeline_minimax_h3.py'
    tree=ast.parse(source.read_text())
    node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_h3_forward_with_oom_cause')
    class OutOfMemoryError(RuntimeError):pass
    torch=N(cuda=N(memory_stats=Mock(side_effect=[{'num_ooms':before},{'num_ooms':after}])),OutOfMemoryError=OutOfMemoryError)
    namespace={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    original=RuntimeError('aten::new_empty API call failed')
    def fail(request):raise original
    with pytest.raises(OutOfMemoryError if is_oom else RuntimeError) as caught:
        namespace['_h3_forward_with_oom_cause'](fail,None)
    if is_oom:assert caught.value.__cause__ is original
    else:assert caught.value is original

def test_h3_success_does_not_turn_allocator_retries_into_oom():
    from types import SimpleNamespace as N
    from unittest.mock import Mock
    source=SOURCE.parents[1]/'minimax_h3/pipeline_minimax_h3.py'
    tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_h3_forward_with_oom_cause')
    stats=Mock(return_value={'num_ooms':7,'num_alloc_retries':100})
    namespace={'torch':N(cuda=N(memory_stats=stats),OutOfMemoryError=MemoryError)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    assert namespace['_h3_forward_with_oom_cause'](lambda request:'ok',None)=='ok'
    stats.assert_called_once()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy and current engine adapters without importing a GPU runtime."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_compat(monkeypatch, *, legacy=False):
    entry = ModuleType('vllm_omni.entrypoints.utils')
    expected = ('legacy.yaml', ['old-stage'])
    if legacy:
        entry.load_and_resolve_stage_configs = lambda *a, **kw: expected
    else:
        entry.prepare_stage_config_inputs = lambda model, kwargs, **kw: SimpleNamespace(
            model=model, kwargs=kwargs, trust_remote_code=False,
            deploy_config_path=kwargs.get('deploy_config'), stage_overrides=None,
            strategy_config_path=None,
        )
    engine = ModuleType('vllm_omni.engine.omni_engine_base')
    calls = []
    def current_loader(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(config_path='current.yaml', stage_configs=['new-stage'])
    engine.load_and_resolve_stage_configs = current_loader
    for name in ['vllm_omni', 'vllm_omni.entrypoints', 'vllm_omni.engine']:
        package=ModuleType(name);package.__path__=[];monkeypatch.setitem(sys.modules,name,package)
    monkeypatch.setitem(sys.modules,entry.__name__,entry)
    monkeypatch.setitem(sys.modules,engine.__name__,engine)
    spec=importlib.util.spec_from_file_location('_test_stage_compat',ROOT/'dingo/vllm/omni/stage_config_compat.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,calls


def test_legacy_config_keeps_legacy_keyword(monkeypatch):
    module,calls=load_compat(monkeypatch,legacy=True)
    assert module.stage_config_engine_kwargs(None)=={}
    assert module.stage_config_engine_kwargs('old.yaml')=={'stage_configs_path':'old.yaml'}
    assert module.load_and_resolve_stage_configs('model','old.yaml',kwargs={})==('legacy.yaml',['old-stage'])
    assert not calls


def test_current_config_uses_deploy_schema_and_adapts_result(monkeypatch,tmp_path):
    module,calls=load_compat(monkeypatch)
    config=tmp_path/'deploy.yaml';config.write_text('stages: []\n')
    assert module.stage_config_engine_kwargs(str(config))=={'deploy_config':str(config)}
    assert module.load_and_resolve_stage_configs('model',str(config),kwargs={'trust_remote_code':False})==('current.yaml',['new-stage'])
    assert calls[0][1]['deploy_config_path']==str(config)
    config.write_text('stage_args: []\n')
    with pytest.raises(ValueError,match='removed the stage_args'):
        module.stage_config_engine_kwargs(str(config))


def test_text_output_accepts_flattened_and_nested_shapes():
    source=ROOT/'dingo/vllm/omni/output_formatter.py';tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_text_request_output')
    namespace={'Any':Any};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    get=namespace['get_text_request_output'];flat=SimpleNamespace(outputs=[])
    assert get(flat) is flat
    assert get(SimpleNamespace(request_output=flat)) is flat
    assert get(SimpleNamespace()) is None

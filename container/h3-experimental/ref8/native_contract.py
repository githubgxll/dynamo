"""Execute hash-pinned native parser/validator AST without importing CUDA libs.

Only the simple filename regex uses stdlib re. Original function bodies and
task guards are compiled verbatim. This is not full pipeline/GPU execution.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import re
from types import SimpleNamespace
from typing import Any

from apply_patch import ARTIFACT


class NativeClientError(Exception):
    pass


def native_functions(lora_bytes, pipeline_bytes):
    nodes = []
    for node in ast.parse(lora_bytes).body:
        if (isinstance(node, ast.Assign) and node.lineno < 100 and
                any(isinstance(t, ast.Name) and t.id.startswith('_TURBO_') for t in node.targets)):
            nodes.append(node)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ('TurboSpec', 'parse_turbo_filename'):
            nodes.append(node)
    ns = dict(__name__=__name__, re=re, math=math, dataclass=dataclass, Any=Any, OmniClientError=NativeClientError)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'native-lora.py', 'exec'), ns)
    pipeline = ast.parse(pipeline_bytes)
    for name in ('_validate_turbo_sampling', '_resolve_task', '_active_turbo_spec'):
        selected = [n for n in ast.walk(pipeline) if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(selected) != 1:
            raise ValueError('Native function absent/ambiguous: ' + name)
        exec(compile(ast.Module(body=selected, type_ignores=[]), 'native-pipeline.py', 'exec'), ns)
    return ns


def check(lora_bytes, pipeline_bytes, *, expected_ref_shift):
    ns = native_functions(lora_bytes, pipeline_bytes)
    worker = SimpleNamespace(default_video_shift=12, default_audio_shift=3,
                             partition='combined', supported_tasks={'t2va', 'fl2va', 'ref2va'}, _fasth3=None)
    rows = []

    def expect(label, fn, accepted):
        try:
            fn()
            actual, reason = True, None
        except NativeClientError as exc:
            actual, reason = False, str(exc)
        if actual != accepted:
            raise AssertionError(f'{label}: accepted={actual}; expected={accepted}; {reason}')
        rows.append({'check': label, 'passed': True, 'accepted': actual, 'error': reason})

    cases = [(ARTIFACT, 8, expected_ref_shift, 'ref2va'),
             ('minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors', 8, 6, 't2va'),
             ('minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors', 4, 6, 'fl2va'),
             ('minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors', 8, 12, 't2va')]
    for filename, steps, shift, task in cases:
        spec = ns['parse_turbo_filename'](filename)
        if spec.video_shift != shift or spec.audio_shift != 3 or spec.denoise_steps != steps or spec.sigma_points != steps + 1:
            raise AssertionError('Wrong artifact sampler specification: ' + filename)
        for request_shift in (6, 12):
            sample = SimpleNamespace(num_inference_steps=steps,
                                     extra_args={'flow_shift': request_shift, 'audio_flow_shift': 3})
            expect(filename + f':shift{request_shift}',
                   lambda: ns['_validate_turbo_sampling'](worker, sample, spec), request_shift == shift)
        for bad_steps, bad_audio in ((steps + 1, 3), (50, 3), (steps, 6)):
            sample = SimpleNamespace(num_inference_steps=bad_steps,
                                     extra_args={'flow_shift': shift, 'audio_flow_shift': bad_audio})
            expect(filename + f':reject-steps{bad_steps}-audio{bad_audio}',
                   lambda: ns['_validate_turbo_sampling'](worker, sample, spec), False)
        expect(filename + ':correct-task', lambda: ns['_resolve_task'](worker, task, turbo_spec=spec), True)
        wrong_task = 't2va' if task == 'ref2va' else 'ref2va'
        expect(filename + ':wrong-task', lambda: ns['_resolve_task'](worker, wrong_task, turbo_spec=spec), False)
    ref = ns['parse_turbo_filename'](ARTIFACT)
    worker._turbo_lora_specs = {7: ref}
    active = SimpleNamespace(lora_request=SimpleNamespace(lora_int_id=7), lora_scale=1)
    if ns['_active_turbo_spec'](worker, active) != ref:
        raise AssertionError('Active LoRA contract is not resolved')
    active.lora_scale = 0
    if ns['_active_turbo_spec'](worker, active) is not None:
        raise AssertionError('Native zero-scale semantics unexpectedly changed')
    # Base has no active adapter, so this patch must not apply a Turbo contract.
    active.lora_request = None
    if ns['_active_turbo_spec'](worker, active) is not None:
        raise AssertionError('Base was misidentified as Turbo')
    return {'status': 'PASS_NATIVE_CONTRACT_CPU_ONLY', 'checks': rows,
            'expected_ref8_video_shift': expected_ref_shift,
            'limits': ['Native function AST execution only; no HTTP/model/GPU execution',
                       'Full quality profile separately requires LoRA scale=1, name and path']}

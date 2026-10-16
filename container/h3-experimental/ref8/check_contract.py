"""Read-only 0.31 Ref8 contract gate. Local AST checks are separate from CPU Torch."""
from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import native_contract
import schedule_gate
from apply_patch import (HERE, TARGET, BEFORE, AFTER, SOURCE_MAP, emit, identity,
                         installed_sources, load_lock, read_json, require, transform)


def check_schedule_selection(pipeline):
    names = ('_resolve_sigma_positions', '_sigma_schedule_for_request', '_base_schedule_for_task')
    nodes = [schedule_gate.function(ast.parse(pipeline), name) for name in names]
    ns = {'OmniClientError': native_contract.NativeClientError, 'math': math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'native-selection.py', 'exec'), ns)
    worker = SimpleNamespace(_fasth3=None, _lora_sigma_schedules={}, _base_schedule_by_partition={})
    worker._sigma_schedule_for_request = lambda sampling, task: ns['_sigma_schedule_for_request'](worker, sampling, task)
    worker._base_schedule_for_task = lambda task: ns['_base_schedule_for_task'](worker, task)
    for task in ('fl2va', 'ref2va'):
        sample = SimpleNamespace(lora_request=SimpleNamespace(lora_int_id=7), lora_scale=1, num_inference_steps=8)
        require(ns['_resolve_sigma_positions'](worker, task, sample) == (None, 8),
                'Turbo no longer selects its eight-interval uniform schedule')
    return {'status': 'PASS_STATIC_SELECTION_ONLY', 'num_inference_steps': 8,
            'base_schedule': None, 'tasks': ['fl2va', 'ref2va']}


def check_native(lora, pipeline, *, ref_shift):
    report = native_contract.check(lora, pipeline, expected_ref_shift=ref_shift)
    ns = native_contract.native_functions(lora, pipeline)
    worker = SimpleNamespace(partition='combined', supported_tasks={'t2va', 'fl2va', 'ref2va'}, _fasth3=None)
    for name in (native_contract.ARTIFACT, 'minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors'):
        spec = ns['parse_turbo_filename'](name)
        try:
            ns['_resolve_task'](worker, 'audio', turbo_spec=spec)
        except native_contract.NativeClientError:
            report['checks'].append({'check': name + ':reject-audio-task', 'passed': True, 'accepted': False})
        else:
            raise ValueError('Unsupported audio task was accepted')
    return report


def local_check(lock):
    original = (HERE / 'fixtures/lora.py').read_bytes()
    pipeline = (HERE / 'fixtures/pipeline_minimax_h3.py').read_bytes()
    return {**identity(), 'status': 'PASS_LOCAL_CONTRACT_ONLY', 'errors': [],
            'before': check_native(original, pipeline, ref_shift=6),
            'after': check_native(transform(original), pipeline, ref_shift=12),
            'request_path': schedule_gate.check_static_path(pipeline),
            'selection': check_schedule_selection(pipeline),
            'native_cpu_torch_executed': False, 'gpu_validated': False,
            'limits': ['Local source/AST checks only, no installed Linux Torch or pipeline execution']}


def installed_check(lock):
    require(sys.platform == 'linux' and os.geteuid() != 0,
            'Final installed gate requires Linux non-root execution')
    sources, report = installed_sources(lock, patched=True)
    applied = read_json('/opt/dynamo/build-info/ref8-patch.json')
    require(applied.get('status') == 'PASS_PATCH_APPLIED_CPU_ONLY' and applied.get('errors') == [],
            'Missing successful build-time patch evidence')
    require(all(applied.get(k) == v for k, v in identity().items()) and
            applied.get('changed_files') == [TARGET] and applied.get('source_sha256_before') == BEFORE,
            'Build-time patch identity mismatch')
    expected_before = {SOURCE_MAP[name]: digest for name, digest in lock['fixtures'].items()}
    require(applied.get('before_source_report', {}).get('sha256') == expected_before,
            'Original six-source evidence does not match fixed 0.31 source')
    require(applied.get('after_source_report', {}).get('sha256') == report['sha256'],
            'Installed six-source evidence changed since patch')
    native = check_native(sources['lora.py'], sources['pipeline_minimax_h3.py'], ref_shift=12)
    selection = check_schedule_selection(sources['pipeline_minimax_h3.py'])
    numeric = schedule_gate.run()  # Real installed CPU torch, no fake tensors.
    return {**identity(), 'status': 'PASS_INSTALLED_CONTRACT_CPU_ONLY', 'errors': [],
            'source_report': report, 'native_contract': native, 'selection': selection,
            'native_cpu_torch_executed': True, 'sigma_gate': numeric, 'gpu_validated': False,
            'limits': ['Real CPU sigma math plus native AST contract; no full H3 model or weights loaded',
                       'No HTTP, GPU kernels, adapter binding, HSDP or generated media acceptance']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--local', action='store_true')
    modes.add_argument('--installed', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    try:
        lock = load_lock()
        result = local_check(lock) if args.local else installed_check(lock)
    except Exception as exc:
        result = {'status': 'FAIL', 'errors': [type(exc).__name__ + ': ' + str(exc)]}
    emit(result, args.report)
    return 1 if result['status'] == 'FAIL' else 0


if __name__ == '__main__':
    sys.exit(main())

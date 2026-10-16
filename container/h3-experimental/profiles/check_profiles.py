"""Check all six native worker profiles, asset binding and separate Base modes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'assets'))
from verify_assets import load_lock
import base_verify

EXPECTED = {'base-fl50': (50, 12, 3, 'none'), 'base-ref50': (50, 12, 3, 'none'),
            'turbo-fl8': (8, 6, 3, 'request_lora'), 'turbo-ref8': (8, 12, 3, 'request_lora'),
            'turbo-fl4': (4, 6, 3, 'request_lora'), 'fasth3-dense4': (4, 12, 3, 'startup_fusion')}


def require(test, message):
    if not test:
        raise ValueError(message)


def validate(profiles=None):
    data = profiles or json.loads((HERE / 'profiles.json').read_bytes())
    assets = load_lock(ROOT / 'assets/assets.lock.json')
    by_id = {a['id']: a for a in assets['assets']}
    require(len(by_id) == len(assets['assets']) == 4, 'Need all four unique adapters')
    require(data['schema'] == 'h3-experimental-profiles/1', 'Wrong profiles schema')
    require(len(data['profiles']) == 6 and {p['id'] for p in data['profiles']} == set(EXPECTED), 'Need all six unique profiles')
    require(data['versions'] == {'vllm': '0.31.0', 'vllm-omni': '0.31.0rc1'}, 'Wrong versions')
    require(data['common']['gpu_count'] == 4 and data['common']['offload'] is False
            and data['common']['cache'] == 'none' and data['common']['quantization'] == 'none', 'Qualification baseline changed')
    require(not any('offload' in arg or 'quantization' in arg for arg in data['common_argv']), 'Unexpected offload or quantization')
    base_lock = base_verify.load_lock(HERE / 'fasth3_base_lock.json')
    sys.path.insert(0, str(ROOT / 'ref8'))
    import apply_patch
    import native_contract
    repair = apply_patch.load_lock()
    native = native_contract.native_functions(apply_patch.transform((ROOT / 'ref8/fixtures/lora.py').read_bytes()),
                                             (ROOT / 'ref8/fixtures/pipeline_minimax_h3.py').read_bytes())
    for p in data['profiles']:
        require((p['steps'], p['video_shift'], p['audio_shift'], p['load_mode']) == EXPECTED[p['id']], 'Profile contract drift: ' + p['id'])
        r = p['request_contract']
        require((r['num_inference_steps'], r['flow_shift'], r['extra_params']['audio_flow_shift']) == EXPECTED[p['id']][:3], 'Request and profile disagree')
        require(r['guidance_scale'] == 1 and r['extra_params']['task'] in p['allowed_tasks'], 'Guidance/task mismatch')
        require(p['expected_sigma_boundaries'] == p['steps'] + 1 and p['expected_denoiser_forwards'] == p['steps'], 'Wrong step units')
        require(p['base_path'] == '/userdata/mengsz/MiniMax-H3/' + ('Ref2VA' if p['base_partition'] == 'ref2va' else 'FL2VA'), 'Wrong Base path')
        if p['load_mode'] == 'none':
            require(p['adapter_path'] is None and p['extra_start_args'] == [] and 'lora' not in r, 'Base must have no adapter')
            continue
        a = by_id[p['id']]
        require((a['steps'], a['video_shift'], a['audio_shift']) == EXPECTED[p['id']][:3], 'Asset metadata and profile disagree')
        require(p['sha256'] == a['sha256'] and p['revision'] == a['revision']
                and p['adapter_path'] == '/opt/h3-adapters/' + a['relative_path'], 'Adapter identity mismatch')
        if p['load_mode'] == 'request_lora':
            require(p['extra_start_args'] == ['--lora-backend', 'peft', '--lora-path', p['adapter_path'], '--lora-scale', '1.0'], 'Wrong Turbo startup arguments')
            require(r.get('lora') == {'name': p['id'], 'path': p['adapter_path'], 'scale': 1.0}, 'Turbo request must activate exact LoRA')
            spec = native['parse_turbo_filename'](a['filename'])
            require((spec.denoise_steps, spec.video_shift, spec.audio_shift) == EXPECTED[p['id']][:3], 'Native patched Turbo parser disagrees')
            require(sorted(spec.supported_tasks) == sorted(p['allowed_tasks']), 'Native task contract differs')
        else:
            require(p['id'] == 'fasth3-dense4' and p['allowed_tasks'] == ['t2va'], 'FastH3 is T2VA only')
            require('lora' not in r and p['extra_start_args'] == ['--lora-path', p['adapter_path']], 'FastH3 startup fusion cannot use request LoRA')
            require(p['base_revision'] == a['base_revision_required'] == base_lock['revision'], 'Wrong FastH3 teacher revision')
            require(p['base_schedule'] == [0.999, 0.749, 0.5, 0.25, 0.0], 'Wrong FastH3 schedule')
    return {'schema': 'h3-experimental-profiles-check/1', 'status': 'PASS_LOCAL_PROFILES_ONLY',
            'errors': [], 'profiles': [p['id'] for p in data['profiles']],
            'versions': data['versions'], 'assets_lock_sha256': hashlib.sha256((ROOT / 'assets/assets.lock.json').read_bytes()).hexdigest(),
            'ref8_implementation': repair['implementation_id'], 'base_payload_verified': False,
            'gpu_validated': False, 'limits': ['Configuration and native parser AST only; no adapter binding or model load']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--installed', action='store_true')
    modes.add_argument('--local', action='store_true', help='Explicit spelling for the default local-only check')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = validate()
        if args.installed:
            require(sys.platform == 'linux' and os.geteuid() != 0, 'Installed gate needs Linux non-root')
            versions = {name: md.version(name) for name in report['versions']}
            require(versions == report['versions'], 'Installed version mismatch')
            require(shutil.which('vllm-omni') is not None, 'Missing official vllm-omni CLI')
            entries = [e for e in md.distribution('vllm-omni').entry_points
                       if e.group == 'console_scripts' and e.name == 'vllm-omni']
            require(len(entries) == 1 and entries[0].value == 'vllm_omni.entrypoints.cli.main:main',
                    'Unexpected official Omni CLI entry point')
            report['status'] = 'PASS_INSTALLED_PROFILES_CPU_ONLY'
            source = os.environ.get('H3_EXPERIMENTAL_SOURCE_COMMIT', '')
            base = os.environ.get('H3_EXPERIMENTAL_BASE_IMAGE', '')
            require(bool(re.fullmatch(r'[0-9a-f]{40}', source)), 'Missing source commit')
            require(bool(re.fullmatch(r'[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}', base)), 'Base image must use a digest')
            inputs = {'schema': 'h3-experimental-inputs/1', 'source_commit': source, 'base_image': base,
                      'versions': versions, 'assets_lock_sha256': report['assets_lock_sha256'],
                      'dynamo_gateway_source_sha': os.environ.get('DYNAMO_GATEWAY_SOURCE_SHA'),
                      'dynamo_vllm_base_image': os.environ.get('DYNAMO_VLLM_BASE_IMAGE')}
            require(args.output is not None, 'Installed gate needs evidence output')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            (args.output.parent / 'image-inputs.json').write_text(json.dumps(inputs, indent=2) + '\n', encoding='utf-8')
    except Exception as exc:
        report = {'schema': 'h3-experimental-profiles-check/1', 'status': 'FAIL', 'errors': [str(exc)]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report['status'] == 'FAIL'


if __name__ == '__main__':
    sys.exit(main())

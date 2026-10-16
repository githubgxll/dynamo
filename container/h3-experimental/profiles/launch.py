"""Launch one pinned native worker; no Kubernetes or download operations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

import base_verify
from check_profiles import HERE, ROOT, require, validate


def command(profile, data, model_path=None):
    selected = model_path or profile['base_path']
    require(isinstance(selected, str) and PurePosixPath(selected).is_absolute()
            and '\\' not in selected and '\x00' not in selected, 'Base path must be an absolute Linux path')
    return ['vllm-omni', 'serve', selected, '--served-model-name', profile['served_model_name'],
            '--task-type', profile['base_partition'], *data['common_argv'], *profile['extra_start_args']]


def configure(work):
    settings = {'TMPDIR': str(work / 'tmp'), 'TMP': str(work / 'tmp'), 'TEMP': str(work / 'tmp'),
                'HF_HOME': str(work / 'cache/hf'), 'XDG_CACHE_HOME': str(work / 'cache/xdg'),
                'TORCH_HOME': str(work / 'cache/torch'), 'TRITON_CACHE_DIR': str(work / 'cache/triton'),
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONNOUSERSITE': '1',
                'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'VLLM_NO_USAGE_STATS': '1',
                'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'DINGO_ENABLE_MINIMAX_H3_PATCHES': '0',
                'OMNI_DIFFUSION_PROMPT_EMBED_CACHE': '0', 'VLLM_OMNI_VAE_LEGACY_TEMPORAL': '0',
                'VLLM_OMNI_VAE_ENCODE_LEGACY_PREP': '0', 'VLLM_OMNI_VIDEO_SYNC_TIMEOUT': '10800',
                'VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT': '10800', 'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT': '10800',
                'NCCL_DEBUG': 'WARN', 'TZ': 'Asia/Shanghai'}
    os.environ.update(settings)
    for relative in ('tmp', 'cache/hf', 'cache/xdg', 'cache/torch', 'cache/triton', 'results'):
        (work / relative).mkdir(parents=True, exist_ok=True)


def preflight(profile, data, model, evidence):
    require(sys.platform == 'linux', 'Worker startup requires Linux image')
    require({name: md.version(name) for name in data['versions']} == data['versions'], 'Wrong installed framework')
    require(model.is_dir(), 'Missing read-only Base directory: ' + str(model))
    require(bool(os.statvfs(model).f_flag & os.ST_RDONLY), 'Base must be mounted read-only')
    index = json.loads((model / 'model_index.json').read_bytes())
    release = index.get('_minimax_h3') or {}
    require(release.get('partition') == profile['base_partition'], 'Base partition mismatch')
    require(release.get('base_schedule') is None and release.get('sigma_shift_scales') == {'video': 12.0, 'audio': 3.0}, 'Expected original Base metadata')
    for component in ('transformer', 'text_encoder', 'video_vae', 'audio_vae', 'processor', 'tokenizer'):
        require((model / component).is_dir(), 'Incomplete Base partition: ' + component)
    for tool in ('vllm-omni', 'ffmpeg', 'ffprobe'):
        require(shutil.which(tool) is not None, 'Missing tool: ' + tool)
    subprocess.run([sys.executable, '-B', str(ROOT / 'ref8/check_contract.py'), '--installed',
                    '--report', str(evidence / 'ref8-contract.json')], check=True)
    if profile['adapter_path']:
        adapter = Path(profile['adapter_path'])
        digest = hashlib.sha256()
        with adapter.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(chunk)
        require(digest.hexdigest() == profile['sha256'], 'Adapter SHA256 mismatch')
    if profile['load_mode'] == 'startup_fusion':
        print('FastH3: verifying complete read-only Base payload before fusion; this may take several minutes.', flush=True)
        try:
            report = base_verify.verify_model(model, HERE / 'fasth3_base_lock.json',
                progress=lambda row, i, n: print(f'Base identity {i}/{n}: {row["path"]}', flush=True))
        except base_verify.BaseVerificationError as exc:
            (evidence / 'fasth3-base-check.json').write_text(json.dumps(exc.evidence, indent=2) + '\n', encoding='utf-8')
            raise
        (evidence / 'fasth3-base-check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    # Use a child process so its CUDA contexts exit before real model loading.
    gpu_code = 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==4; assert all(torch.cuda.get_device_capability(i)==(9,0) and "H100" in torch.cuda.get_device_name(i) for i in range(4)); print("GPU visibility checked; model execution not yet validated")'
    subprocess.run([sys.executable, '-c', gpu_code], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--model-path', help='Optional preexisting read-only Base partition; FastH3 exact-byte gate remains mandatory')
    parser.add_argument('--work-dir', type=Path, default=Path('/work'))
    parser.add_argument('--start', action='store_true', help='Explicitly run preflight then native worker; omitted prints command only')
    args = parser.parse_args()
    validate()
    data = json.loads((HERE / 'profiles.json').read_bytes())
    found = [p for p in data['profiles'] if p['id'] == args.profile]
    require(len(found) == 1, 'Unknown profile')
    profile = found[0]
    argv = command(profile, data, args.model_path)
    print(json.dumps({'argv': argv, 'request_contract': profile['request_contract'], 'execute': args.start}, indent=2), flush=True)
    if not args.start:
        return
    work = args.work_dir.resolve()
    require(work.is_dir() and os.access(work, os.W_OK), 'Need writable mounted work directory')
    configure(work)
    evidence = work / ('startup-' + args.profile + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    evidence.mkdir(exist_ok=False)
    try:
        preflight(profile, data, Path(args.model_path or profile['base_path']), evidence)
        (evidence / 'launch.json').write_text(json.dumps({'argv': argv, 'profile': profile,
            'status': 'PREFLIGHT_PASS_MODEL_STARTING', 'gpu_media_validated': False}, indent=2) + '\n', encoding='utf-8')
        os.chdir(work)
        os.execvpe(argv[0], argv, os.environ)
    except Exception as exc:
        (evidence / 'startup-error.json').write_text(json.dumps({'status': 'FAIL_BEFORE_MODEL_READY', 'error': str(exc)}, indent=2) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()

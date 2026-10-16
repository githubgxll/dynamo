"""Apply one source-locked Ref8 correction during a Linux image build only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

HERE = Path(__file__).resolve().parent
IMPLEMENTATION = 'omni031rc1-ref8-shift12-v1'
VERSION = '0.31.0rc1'
COMMIT = '6dd0d1f9310f7598b773c796b434c8000c2816ec'
WHEEL_SHA = '93ff995e7fcde7a7132d460eb21102e727f68068ebd58744526ee3178149350c'
TARGET = 'diffusion/models/minimax_h3/lora.py'
ARTIFACT = 'minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors'
BEFORE = 'f811b4d754d295eebb0d6940f60af307e401da54f1adb12ae2dd97b942e24c2f'
AFTER = '4628f4b90b47aa824bfae141ea5c3c17713ea13f73c88135215afac556d774cf'
OLD = b'        video_shift=_TURBO_VIDEO_SHIFT_768P if match.group("res") else _TURBO_VIDEO_SHIFT_544P,\n'
NEW = b'''        # Ref8 v1.0 768p is an artifact-specific exception, not an FL change.
        # https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/51
        video_shift=(
            12.0
            if name == "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"
            else (_TURBO_VIDEO_SHIFT_768P if match.group("res") else _TURBO_VIDEO_SHIFT_544P)
        ),
'''
SOURCE_MAP = {
    name: 'diffusion/models/minimax_h3/' + name for name in
    ('lora.py', 'pipeline_minimax_h3.py', 'time_request.py', 'denoise_loop.py', 'sampling.py')
}
SOURCE_MAP['sigma_schedule.py'] = 'diffusion/sched/sigma_schedule.py'
TOOLS = {'apply_patch.py', 'check_contract.py', 'native_contract.py', 'schedule_gate.py'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def transform(raw):
    require(sha(raw) == BEFORE and raw.count(OLD) == 1,
            'Refusing patch: original lora.py SHA256 or exact replacement differs')
    result = raw.replace(OLD, NEW, 1)
    compile(result, TARGET, 'exec')
    require(sha(result) == AFTER, 'Patched output differs from reviewed SHA256')
    return result


def checked_read(path, root):
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            'Unsafe source path: ' + str(path))
    require(path.is_file(), 'Missing source: ' + str(path))
    return path.read_bytes()


def load_lock():
    lock = read_json(HERE / 'patch.lock.json')
    require(lock.get('schema') == 'h3-ref8-omni031/1' and
            lock.get('implementation_id') == IMPLEMENTATION and
            lock.get('omni_version') == VERSION and lock.get('omni_commit') == COMMIT and
            lock.get('wheel_sha256') == WHEEL_SHA and
            lock.get('source_sha256_before') == BEFORE and
            lock.get('source_sha256_after') == AFTER and
            lock.get('changed_files') == [TARGET], 'Unexpected patch/version identity')
    require(set(lock.get('fixtures', {})) == set(SOURCE_MAP), 'Incomplete native source inventory')
    require(set(lock.get('tools', {})) == TOOLS, 'Incomplete patch tool inventory')
    for name, digest in lock['tools'].items():
        require(sha(checked_read(HERE / name, HERE)) == digest, 'Changed tool: ' + name)
    for name, digest in lock['fixtures'].items():
        require(sha(checked_read(HERE / 'fixtures' / name, HERE)) == digest, 'Changed fixture: ' + name)
    require(sha(transform((HERE / 'fixtures/lora.py').read_bytes())) == AFTER,
            'Fixture is not the original reviewed source')
    return lock


def identity():
    return {'implementation_id': IMPLEMENTATION, 'omni_version': VERSION,
            'omni_commit': COMMIT, 'wheel_sha256': WHEEL_SHA,
            'patch_lock_sha256': sha((HERE / 'patch.lock.json').read_bytes()),
            'source_sha256_after': AFTER}


def distribution(name):
    canonical = lambda value: value.lower().replace('_', '-').replace('.', '-')
    found = {str(getattr(item, '_path', '')): item for item in md.distributions()
             if canonical(item.metadata.get('Name', '')) == name}
    require(len(found) == 1, 'Missing or ambiguous installed distribution: ' + name)
    return next(iter(found.values()))


def installed_sources(lock, *, patched):
    require(sys.platform == 'linux', 'Installed source validation requires Linux')
    omni = distribution('vllm-omni')
    require(omni.version == VERSION, 'Expected exactly vllm-omni ' + VERSION)
    vllm = distribution('vllm')
    require(vllm.version.split('+', 1)[0] == '0.31.0', 'Expected vLLM 0.31.0 stable')
    spec = importlib.util.find_spec('vllm_omni')
    require(spec is not None and spec.submodule_search_locations is not None,
            'Installed vllm_omni package was not found')
    roots = list(spec.submodule_search_locations)
    require(len(roots) == 1, 'Ambiguous effective vllm_omni package root')
    root = Path(roots[0]).resolve()
    require(root == Path(omni.locate_file('vllm_omni')).resolve(),
            'Effective package root does not belong to the selected wheel')
    owned = {str(item).replace('\\', '/') for item in (omni.files or ())}
    sources, hashes = {}, {}
    for name, relative in SOURCE_MAP.items():
        require('vllm_omni/' + relative in owned, 'Wheel RECORD does not own: ' + relative)
        raw = checked_read(root / relative, root)
        expected = AFTER if patched and name == 'lora.py' else lock['fixtures'][name]
        require(sha(raw) == expected, 'Installed source hash mismatch: ' + relative)
        sources[name], hashes[relative] = raw, expected
    return sources, {'status': 'PASS_INSTALLED_SOURCES_ONLY', 'root': str(root),
                     'versions': {'vllm': vllm.version, 'vllm-omni': omni.version},
                     'sha256': hashes, 'patched': patched}


def apply_installed(lock):
    require(sys.platform == 'linux' and os.geteuid() == 0,
            'Patch application is restricted to Linux image construction as root')
    sources, before = installed_sources(lock, patched=False)
    updated = transform(sources['lora.py'])
    target = Path(before['root']) / TARGET
    fd, temporary = tempfile.mkstemp(prefix='.ref8-shift12-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(updated)
        os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    # Force Python to compile the edited source rather than use an old cache.
    for cached in (target.parent / '__pycache__').glob('lora.*.pyc'):
        require(not cached.is_symlink(), 'Unexpected symbolic compiled cache')
        cached.unlink()
    _, after = installed_sources(lock, patched=True)
    return {**identity(), 'status': 'PASS_PATCH_APPLIED_CPU_ONLY', 'errors': [],
            'source_sha256_before': BEFORE, 'changed_files': [TARGET],
            'before_source_report': before, 'after_source_report': after,
            'gpu_validated': False,
            'limits': ['Only six relevant source files are validated; this is not a whole-package audit',
                       'Asset content is independently verified by the image asset gate',
                       'No model loading, GPU execution or media generation']}


def emit(report, destination):
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding='utf-8')
    print(payload, end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply-installed', action='store_true', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = apply_installed(load_lock())
    except Exception as exc:
        result = {'status': 'FAIL', 'errors': [type(exc).__name__ + ': ' + str(exc)]}
    emit(result, args.report)
    return 1 if result['status'] == 'FAIL' else 0


if __name__ == '__main__':
    sys.exit(main())

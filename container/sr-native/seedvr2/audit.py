"""Offline, non-root image checks. This does not qualify GPU inference."""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE = Path('/opt/sources/SeedVR')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source=SOURCE):
    manifest = json.loads((HERE / 'source-tree.json').read_text())
    for relative, expected in manifest['files'].items():
        path = source / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'Official source differs: {relative}')
    return manifest['commit']


def audit(output):
    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'FAIL', 'backend': 'seedvr2', 'euid': os.geteuid(),
              'source_commit': os.environ.get('DYNAMO_COMMIT_SHA'),
              'gpu_validation_status': 'NOT_RUN'}
    try:
        report['upstream_commit'] = verify_source()
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError('Expected Python 3.10')
        versions = {'torch': '2.3.0+cu121', 'torchvision': '0.18.0+cu121',
                    'diffusers': '0.29.1', 'transformers': '4.38.2',
                    'numpy': '1.26.4', 'av': '12.0.0', 'flash-attn': '2.5.9.post1'}
        report['versions'] = {name: metadata.version(name) for name in versions}
        if report['versions'] != versions:
            raise RuntimeError('Pinned core dependency versions differ')
        check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
        (output / 'pip-check.txt').write_text(check.stdout + check.stderr)
        if check.returncode:
            raise RuntimeError('pip check failed; see pip-check.txt')
        # Load libtorch/libc10 before importing directly linked extension modules.
        import torch
        if torch.version.cuda != '12.1':
            raise RuntimeError('Expected Torch built against CUDA12.1')
        sys.path.insert(0, str(SOURCE))
        for name in ('fused_layer_norm_cuda', 'apex.normalization', 'flash_attn_2_cuda',
                     'flash_attn', 'models.dit.nadit', 'models.dit_v2.nadit',
                     'projects.video_diffusion_sr.infer'):
            importlib.import_module(name)
        import numpy as np
        import mediapy
        from torchvision.io import read_video
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            video = Path(temporary) / 'probe.mp4'
            pixels = np.zeros((17, 64, 64, 3), dtype=np.uint8)
            pixels[..., 1] = np.arange(17, dtype=np.uint8)[:, None, None] * 10
            mediapy.write_video(str(video), pixels, fps=24)
            decoded, _, info = read_video(str(video), output_format='TCHW')
            if tuple(decoded.shape) != (17, 3, 64, 64) or abs(info['video_fps'] - 24) > 1e-6:
                raise RuntimeError('FFmpeg/PyAV/torchvision media roundtrip failed')
        report['status'] = 'PASS_CPU_ONLY'
        report['scope'] = 'Exact source/import/dependencies/media reader only; CUDA kernels and models not executed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (output / 'runtime-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    audit(parser.parse_args().output_dir)

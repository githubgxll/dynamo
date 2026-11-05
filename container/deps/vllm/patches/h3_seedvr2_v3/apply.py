# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply the tested H3/SeedVR2 overlay to its source-locked Omni 0.32 base."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent


def prepare(site_packages: Path):
    site_packages = site_packages.resolve(strict=True)
    lock = json.loads((ROOT / 'source-lock.json').read_text())
    versions = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in distributions(path=[str(site_packages)])}
    for name, key in [('vllm', 'vllm_version'), ('vllm-omni', 'vllm_omni_version')]:
        if versions.get(name) != lock[key]:
            raise ValueError(f'{name}: expected {lock[key]}, found {versions.get(name)}')
    plans = []
    for entry in lock['files']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid source path')
        source = ROOT / 'sources' / relative
        target = site_packages / relative
        if target.is_symlink() or not target.resolve().is_relative_to(site_packages):
            raise ValueError(f'unsafe target: {relative}')
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['after_sha256']:
            raise ValueError(f'overlay checksum mismatch: {relative}')
        compile(data, str(relative), 'exec')
        actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
        allowed = {entry['before_sha256'], entry['after_sha256'], *entry.get('compatible_parent_sha256', [])}
        if actual not in allowed:
            raise ValueError(f'base checksum mismatch: {relative}')
        if actual != entry['after_sha256']:
            plans.append((target, data))
    return plans


def apply(site_packages: Path, *, check: bool = False):
    plans = prepare(site_packages)  # Validate every source before writing any.
    if not check:
        for target, data in plans:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix='.dingo-sr-', dir=target.parent)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temp, target.stat().st_mode & 0o777 if target.exists() else 0o644)
                os.replace(temp, target)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
    return len(plans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site_packages', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(json.dumps({'check': args.check, 'files_to_replace': apply(args.site_packages, check=args.check)}))

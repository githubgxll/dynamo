"""Bind Linux preparation locks to the reviewed build inputs, excluding locks themselves."""
import argparse
import hashlib
import json
from pathlib import Path


def collect(vbench_root, compliance_root):
    files = {}
    for prefix, root in [('container/vbench', Path(vbench_root)), ('container/compliance', Path(compliance_root))]:
        for p in sorted(root.rglob('*')):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if any(x in {'locks', 'tests', '__pycache__', '.git'} for x in rel.parts) or p.suffix in {'.pyc', '.zip', '.whl', '.pth', '.pt', '.safetensors', '.mp4'} or p.name.startswith('test_'):
                continue
            files[f'{prefix}/{rel.as_posix()}'] = hashlib.sha256(p.read_bytes()).hexdigest()
    if not files or not any(p.endswith('/requirements.in') for p in files):
        raise ValueError('Missing required build inputs')
    return {'schema_version': 1, 'files': files}


def verify(expected, current):
    if expected != current:
        old, new = expected.get('files', {}), current.get('files', {})
        changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        raise ValueError('Build inputs changed after preparation; rerun preparation: ' + ', '.join(changed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['prepare', 'publish'], required=True)
    p.add_argument('--vbench-root', type=Path, required=True)
    p.add_argument('--compliance-root', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    a = p.parse_args()
    doc = collect(a.vbench_root, a.compliance_root)
    if a.phase == 'publish':
        expected = json.loads((a.vbench_root / 'locks/input-fingerprint.json').read_text(encoding='utf-8'))
        verify(expected, doc)
    a.evidence.mkdir(parents=True, exist_ok=True)
    (a.evidence / 'input-fingerprint.json').write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

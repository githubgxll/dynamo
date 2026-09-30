"""Record public, explicit build identifiers only; never serialize environment variables."""
import argparse
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['prepare', 'publish'], required=True)
    p.add_argument('--base-image', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not re.fullmatch(r'[a-z0-9./:_-]+@sha256:[a-f0-9]{64}', a.base_image):
        raise SystemExit('Base image must be a resolved digest reference')
    if not re.fullmatch(r'[a-f0-9]{40}', a.source_commit):
        raise SystemExit('DYNAMO_COMMIT_SHA must be the full source commit')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({
        'schema_version': 1, 'phase': a.phase, 'base_image': a.base_image,
        'dynamo_source_commit': a.source_commit,
        'vbench_commit': 'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490',
        'protocol': 'custom_input-six-dimensions-longer-r1',
        'platform': 'linux/amd64', 'gpu_validation': 'NOT_RUN',
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

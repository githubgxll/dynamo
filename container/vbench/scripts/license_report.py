"""Use the repository policy and add assets that package metadata cannot describe.

Preparation exports violations for review. A runtime image cannot pass with violations.
No policy exceptions or assertions of legal permission are manufactured here.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.metadata as metadata
import json
import re
import subprocess
import sys
from pathlib import Path

PYIQA_REVIEWED_SPDX = 'CC-BY-NC-SA-4.0 AND LicenseRef-S-Lab-1.0'


def correct_pyiqa_notice(path):
    """Patch only the reviewed component header; preserve all original notice bytes.

    The generic generator reads pyIQA's Apache classifier but retains its bundled
    LICENSE text. Keep that text and every other component intact, and point to the
    separately exported pair of LICENSE and LICENSE-S-Lab files.
    """
    path = Path(path)
    original = path.read_bytes()
    headers = [(b'## pyiqa 0.1.13' + nl + nl + b'License: Apache-2.0' + nl, nl)
               for nl in (b'\n', b'\r\n')]
    matches = [(header, nl) for header, nl in headers if original.count(header) == 1]
    headings = re.findall(rb'(?m)^## pyiqa(?:\s|$)', original)
    if len(headings) != 1 or len(matches) != 1:
        raise ValueError('Unexpected pyIQA NOTICES component/version/license format; review before changing its declaration')
    expected, newline = matches[0]
    replacement = (b'## pyiqa 0.1.13' + newline + newline + b'License: ' +
                   PYIQA_REVIEWED_SPDX.encode('ascii') + newline +
                   b'Bundled license evidence: pyiqa-bundled-licenses.txt (LICENSE and LICENSE-S-Lab)' + newline)
    path.write_bytes(original.replace(expected, replacement, 1))


def require_pass(path):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    if doc.get('schema_version') != 1 or doc.get('policy_pass') is not True or doc.get('violations'):
        raise ValueError('License policy has unresolved findings. Review legal/license-status.json; '
                         'do not disable the gate or treat Harbor administrator access as license approval.')


def generate(policy_path, output):
    from compliance.generators.common import Component, write_cyclonedx, write_merged_csv, write_notices
    from compliance.policy.validate import load_policy, validate_row

    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'compliance.generators', '--ecosystem', 'python,dpkg',
                    '--venv', '/opt/venv', '--output-dir', str(output), '--policy', str(policy_path)], check=True)
    with (output / 'osrb-deps.csv').open(encoding='utf-8', newline='') as f:
        components = [Component(r['ecosystem'], r['name'], r['version'], r['spdx'], r['source_url'])
                      for r in csv.DictReader(f)]
    corrected = []
    for item in components:
        if item.ecosystem == 'python' and item.name.lower().replace('_', '-') == 'pyiqa':
            if item.version != '0.1.13':
                raise ValueError('Re-review pyIQA file licenses for a changed version')
            item = dataclasses.replace(item, spdx=PYIQA_REVIEWED_SPDX)
        corrected.append(item)
    components = corrected
    source_root = Path('/opt/VBench')
    assets_root = Path('/opt/vbench-assets/vbench')
    extras = []
    for name, version, spdx, path, url in [
        ('VBench', 'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490', 'Apache-2.0', source_root/'LICENSE', 'https://github.com/Vchitect/VBench'),
        ('AMT-vendored', 'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490', 'CC-BY-NC-4.0', source_root/'vbench/third_party/amt/LICENSE', 'https://github.com/MCG-NKU/AMT'),
        ('RAFT-vendored', 'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490', 'BSD-3-Clause', source_root/'vbench/third_party/RAFT/LICENSE', 'https://github.com/princeton-vl/RAFT'),
        ('DINO-code', '7c446df5b9f45747937fb0d72314eb9f7b66930a', 'Apache-2.0', assets_root/'dino_model/facebookresearch_dino_main/LICENSE', 'https://github.com/facebookresearch/dino'),
    ]:
        text = path.read_text(encoding='utf-8')
        extras.append(Component('native', name, version, spdx, url, text))
    checkpoints = [
        ('dino-vitb16', 'dino_model/dino_vitbase16_pretrain.pth'),
        ('clip-vit-b32', 'clip_model/ViT-B-32.pt'),
        ('amt-s', 'amt_model/amt-s.pth'),
        ('raft-things', 'raft_model/models/raft-things.pth'),
        ('clip-vit-l14', 'clip_model/ViT-L-14.pt'),
        ('laion-linear-head', 'aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth'),
        ('musiq-spaq', 'pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'),
    ]
    for name, relative in checkpoints:
        digest = sha256_file(assets_root/relative)
        extras.append(Component('native', 'checkpoint-'+name, digest, 'UNKNOWN', None,
                                'Checkpoint-specific license applicability has not been recorded. '
                                'Code licenses alone do not establish checkpoint redistribution terms.'))
    pyiqa = metadata.distribution('pyiqa')
    bundled = []
    for f in pyiqa.files or []:
        if f.name in {'LICENSE', 'LICENSE-S-Lab'} and '.dist-info/' in str(f):
            bundled.append(str(f)+'\n'+Path(pyiqa.locate_file(f)).read_text(encoding='utf-8'))
    if len(bundled) < 2:
        raise ValueError('pyIQA bundled license evidence missing')
    (output/'pyiqa-bundled-licenses.txt').write_text('\n\n'.join(bundled), encoding='utf-8')
    correct_pyiqa_notice(output/'NOTICES-Python.txt')
    write_notices('VBench-assets', extras, output)
    components.extend(extras)
    write_merged_csv(components, {}, output/'osrb-deps.csv')
    write_cyclonedx(components, output/'osrb.cdx.json')
    policy = load_policy(policy_path)
    findings = []
    for c in components:
        v = validate_row(policy, c.ecosystem, c.name, c.version, c.spdx, image='vbench-runtime')
        if v is not None:
            findings.append(dataclasses.asdict(v))
    doc = {'schema_version': 1, 'image_scope': 'vbench-runtime', 'policy_pass': not findings,
           'components': len(components), 'violations': findings,
           'scope': 'Full installed Python/dpkg plus explicitly inventoried VBench source and checkpoints; no vLLM baseline subtraction',
           'policy_sha256': sha256_file(policy_path)}
    (output/'license-status.json').write_text(json.dumps(doc, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'policy_pass': not findings, 'violations': len(findings), 'report': str(output/'license-status.json')}))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--require-pass', type=Path)
    a = p.parse_args()
    if a.require_pass:
        require_pass(a.require_pass)
    elif a.policy and a.output:
        generate(a.policy, a.output)
    else:
        p.error('Use --require-pass or both --policy and --output')


if __name__ == '__main__':
    main()

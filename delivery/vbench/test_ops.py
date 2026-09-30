"""Operator boundary checks; no Git push, Docker, or Kubernetes calls."""
import argparse
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


OPS = module('ops_under_test', Path(__file__).with_name('ops.py'))
FP = module('fingerprint_under_test', OPS.ROOT/'container/vbench/scripts/fingerprint.py')
LICENSE = module('license_under_test', OPS.ROOT/'container/vbench/scripts/license_report.py')


@pytest.mark.parametrize('name', ['../outside.txt', '/absolute.txt', 'C:/outside.txt'])
def test_import_rejects_unsafe_archive(tmp_path, name):
    path = tmp_path/'input.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(name, 'not allowed')
    with zipfile.ZipFile(path) as z, pytest.raises(ValueError, match='Unsafe'):
        OPS.safe_zip_members(z)


def test_wrong_commit_artifact_cannot_write_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(OPS, 'LOCKS', tmp_path/'locks')
    monkeypatch.setattr(OPS, 'DEST', tmp_path/'evidence')
    monkeypatch.setattr(OPS, 'current_fingerprint', lambda: {'files': {}})
    monkeypatch.setattr(OPS, 'git', lambda *args: 'a'*40)
    path = tmp_path/'wrong-commit.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('input-fingerprint.json', json.dumps({'files': {}}))
        z.writestr('build-identity.json', json.dumps({'phase': 'prepare', 'dynamo_source_commit': 'b'*40}))
    with pytest.raises(ValueError, match='current HEAD'):
        OPS.import_preparation(path)
    assert not OPS.LOCKS.exists() and not OPS.DEST.exists()


def test_fingerprint_binds_code_but_not_generated_locks(tmp_path):
    source = tmp_path/'vbench'
    compliance = tmp_path/'compliance'
    source.mkdir(); compliance.mkdir()
    (source/'requirements.in').write_text('torch==2.3.1\n')
    (source/'score.py').write_text('PROTOCOL = 1\n')
    original = FP.collect(source, compliance)
    (source/'locks').mkdir()
    (source/'locks/requirements.lock').write_text('generated\n')
    (source/'test_example.py').write_text('local tests only\n')
    assert FP.collect(source, compliance) == original
    (source/'score.py').write_text('PROTOCOL = 2\n')
    with pytest.raises(ValueError, match='score.py'):
        FP.verify(original, FP.collect(source, compliance))


def test_license_gate_does_not_accept_string_true_or_remaining_findings(tmp_path):
    path = tmp_path/'license-status.json'
    for report in [{'schema_version': 1, 'policy_pass': 'true', 'violations': []},
                   {'schema_version': 1, 'policy_pass': True, 'violations': ['unresolved']}]:
        path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match='unresolved'):
            LICENSE.require_pass(path)


def test_pod_requires_digest_and_preserves_node_selector(tmp_path):
    args = argparse.Namespace(image=OPS.IMAGE_REPO+':latest', name='jirx-vbench-test',
                              namespace='token-factory', gpus=1, pull_secret='existing-pull-secret',
                              output=str(tmp_path/'pod.json'))
    with pytest.raises(ValueError, match='digest'):
        OPS.pod_manifest(args)
    assert not Path(args.output).exists()
    args.image = OPS.IMAGE_REPO+'@sha256:'+'a'*64
    OPS.pod_manifest(args)
    spec = json.loads(Path(args.output).read_text())['spec']
    assert 'nodeName' not in spec
    assert spec['nodeSelector'] == {'kubernetes.io/hostname': 'hd04-gpu1-0063'}
    assert spec['containers'][0]['resources']['limits']['nvidia.com/gpu-h100-80gb-hbm3'] == '1'
    assert spec['imagePullSecrets'] == [{'name': 'existing-pull-secret'}]

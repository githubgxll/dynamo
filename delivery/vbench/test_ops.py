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
CI = module('vbench_ci_ops_test', OPS.ROOT/'.github/scripts/vbench_ci.py')


def test_prepare_local_check_rejects_invalid_base_before_external_commands(tmp_path, monkeypatch):
    config = tmp_path/'configuration.json'
    config.write_text(json.dumps({'vbench': {'phase': 'prepare'}}))
    monkeypatch.setattr(OPS, 'CONFIG', config)
    monkeypatch.setattr(OPS, 'DEST', tmp_path/'evidence')

    def invalid_definition():
        raise ValueError('invalid reviewed base definition')

    monkeypatch.setattr(CI, 'load_base_definition', invalid_definition)
    monkeypatch.setattr(OPS, 'base_helper', lambda: CI)
    monkeypatch.setattr(OPS, 'run', lambda *args, **kwargs: pytest.fail('Invalid base must stop before external commands'))
    with pytest.raises(ValueError, match='invalid reviewed base definition'):
        OPS.check()
    assert not OPS.DEST.exists()


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


def artifact_documents(base):
    """An otherwise complete artifact whose declared identity matches its lock."""
    return {
        'input-fingerprint.json': {'files': {}},
        'build-identity.json': {'phase': 'prepare', 'dynamo_source_commit': 'a'*40,
                                'base_image': base['image'], 'platform': 'linux/amd64'},
        'base.lock.json': base,
        'assets.lock.json': {'schema_version': 1},
        'legal/license-status.json': {'schema_version': 1, 'policy_pass': True, 'violations': []},
    }


@pytest.mark.parametrize('change', [
    {'source_tag': 'nvidia/cuda:latest'},
    {'digest': 'sha256:'+'a'*64, 'image': 'nvidia/cuda@sha256:'+'a'*64},
    {'schema_version': 999},
])
def test_import_rejects_base_drift_even_when_artifact_identity_matches(tmp_path, monkeypatch, change):
    monkeypatch.setattr(OPS, 'LOCKS', tmp_path/'locks')
    monkeypatch.setattr(OPS, 'DEST', tmp_path/'evidence')
    monkeypatch.setattr(OPS, 'current_fingerprint', lambda: {'files': {}})
    monkeypatch.setattr(OPS, 'git', lambda *args: 'a'*40)
    base = {**CI.load_base_definition(), **change}
    path = tmp_path/'wrong-base.zip'
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in artifact_documents(base).items():
            z.writestr(name, json.dumps(data))
        z.writestr('requirements.lock', 'example==1.0 --hash=sha256:'+'b'*64)
    with pytest.raises(ValueError):
        OPS.import_preparation(path)
    assert not OPS.LOCKS.exists() and not OPS.DEST.exists()


def test_valid_reviewed_base_can_be_imported_without_docker_or_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(OPS, 'LOCKS', tmp_path/'locks')
    monkeypatch.setattr(OPS, 'DEST', tmp_path/'evidence')
    monkeypatch.setattr(OPS, 'current_fingerprint', lambda: {'files': {}})
    monkeypatch.setattr(OPS, 'git', lambda *args: 'a'*40)
    monkeypatch.setattr(OPS, 'run', lambda *args, **kwargs: pytest.fail('Import must not call an external command'))
    base = CI.load_base_definition()
    path = tmp_path/'valid-base.zip'
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in artifact_documents(base).items():
            z.writestr(name, json.dumps(data))
        z.writestr('requirements.lock', 'example==1.0 --hash=sha256:'+'b'*64)
    OPS.import_preparation(path)
    assert CI.validate_base_lock(json.loads((OPS.LOCKS/'base.lock.json').read_text())) == base['image']
    assert len(list(OPS.DEST.iterdir())) == 1


def test_publish_phase_rejects_mismatched_base_before_changing_configuration(tmp_path, monkeypatch):
    lockdir = tmp_path/'locks'
    lockdir.mkdir()
    monkeypatch.setattr(OPS, 'LOCKS', lockdir)
    monkeypatch.setattr(OPS, 'current_fingerprint', lambda: {'files': {}})
    base = {**CI.load_base_definition(), 'digest': 'sha256:'+'a'*64,
            'image': 'nvidia/cuda@sha256:'+'a'*64}
    for name, data in {'base.lock.json': base, 'assets.lock.json': {'schema_version': 1},
                       'input-fingerprint.json': {'files': {}},
                       'preparation-license-status.json': {'policy_pass': True, 'violations': []}}.items():
        (lockdir/name).write_text(json.dumps(data))
    (lockdir/'requirements.lock').write_text('example==1.0\n')
    config = tmp_path/'configuration.json'
    config.write_text(json.dumps({'vbench': {'phase': 'prepare'}}))
    before = config.read_bytes()
    monkeypatch.setattr(OPS, 'CONFIG', config)
    with pytest.raises(ValueError):
        OPS.verify_imported_locks()
    with pytest.raises(ValueError):
        OPS.set_phase('publish')
    assert config.read_bytes() == before


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


def test_reviewed_base_definition_is_a_fingerprinted_input(tmp_path):
    source = tmp_path/'vbench'
    compliance = tmp_path/'compliance'
    source.mkdir(); compliance.mkdir()
    (source/'requirements.in').write_text('torch==2.3.1\n')
    definition = source/'base-image.json'
    definition.write_text(json.dumps(CI.load_base_definition()))
    original = FP.collect(source, compliance)
    assert 'container/vbench/base-image.json' in original['files']
    changed = {**CI.load_base_definition(), 'digest': 'sha256:'+'a'*64}
    definition.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='base-image.json'):
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

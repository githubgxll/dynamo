"""Verify one downloaded Actions backend artifact without contacting the registry."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile
import ops

ROOT = ops.ROOT
spec = importlib.util.spec_from_file_location('native_ci', ROOT/'.github/scripts/sr_native_ci.py')
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


def verify(directory):
    record = ci.read(directory/'image.json')
    lock = ops.load_lock()
    ci.validate_lock(lock)
    if record.get('schema') != 'sr-native-publication/1' or record.get('status') != 'PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED':
        raise ValueError('Not a completed native SR publication')
    backend = record.get('backend')
    if backend not in lock['models'] or record.get('branch') != lock['remote_branch']:
        raise ValueError('Unexpected model or branch')
    commit = record['source_commit']
    environ = {'GITHUB_SHA': commit, 'GITHUB_REF_NAME': record['branch'], 'GITHUB_EVENT_NAME': 'push',
               'GITHUB_RUN_ID': record.get('run_id', ''), 'GITHUB_RUN_ATTEMPT': record.get('run_attempt', '')}
    ci.identity(environ)
    if commit != ops.git('rev-parse', 'HEAD'):
        raise ValueError('Use the artifact from this checkout HEAD; do not combine recipe revisions')
    if record.get('base_image') != lock['models'][backend]['base_image']:
        raise ValueError('Base image differs')
    digest = record.get('digest', '')
    if not ci.re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
        raise ValueError('Invalid registry digest')
    reference = lock['harbor_repository'] + '@' + digest
    if record.get('deployment_image') != reference or (directory/'deployment-image.txt').read_text().strip() != reference:
        raise ValueError('Deployment image identity mismatch')
    expected_tag = ':'.join([lock['harbor_repository'], lock['models'][backend]['tag_prefix']]) + '-' + commit[:12] + '-' + record['run_id'] + '-' + record['run_attempt']
    if record.get('image') != expected_tag or record.get('gpu_validation_status') != 'NOT_RUN':
        raise ValueError('Unexpected tag or qualification level')
    if record.get('actions_url') != 'https://github.com/githubgxll/dynamo/actions/runs/' + record['run_id']:
        raise ValueError('Actions URL and run identity differ')
    hashes = record.get('evidence_sha256', {})
    required = {'inputs.json', 'candidate-image-inspect.json', 'installed-inventory.json',
                'packages.cdx.json', 'pip-check.txt', 'final/runtime-validation.json',
                'base-resolved.json', 'base-image-inspect.json', 'final-container-state.json',
                'recipe/delivery/sr-native/source-lock.json', 'recipe/delivery/sr-native/input-seal.json'}
    if not required.issubset(hashes):
        raise ValueError('Required publication evidence is missing')
    if hashes != ci.evidence_hashes(directory):
        raise ValueError('Incomplete or modified publication evidence manifest')
    sealed = ci.read(directory/'recipe/delivery/sr-native/input-seal.json')
    recipe_names = {ops.SEAL, *sealed['shared_inputs'],
                    *(name for name, item in sealed['delta'].items() if item['status'] != 'D')}
    if {name.removeprefix('recipe/') for name in hashes if name.startswith('recipe/')} != recipe_names:
        raise ValueError('Exported recipe does not contain exactly the sealed inputs')
    for name, expected in hashes.items():
        relative = ops.LEGACY.safe_relative(name)
        path = directory.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Evidence changed: ' + name)
        if name.startswith('recipe/'):
            source = name.removeprefix('recipe/')
            if path.read_bytes() != ops.git('show', commit + ':' + source, binary=True):
                raise ValueError('Recipe differs from submitted Git source: ' + source)
    values = ci.read(directory/'inputs.json')
    if values != ci.expected_inputs(backend, environ):
        raise ValueError('Inputs and publication differ')
    base = ci.read(directory/'base-image-inspect.json')
    ci.validate_base(base, values['base_image'])
    base_record = ci.read(directory/'base-resolved.json')
    if (base_record.get('status') != 'PASS_IMMUTABLE_BASE_PULL' or base_record.get('backend') != backend
            or base_record.get('base_image') != values['base_image'] or base_record.get('image_id') != base['Id']):
        raise ValueError('Archived base pull and source lock differ')
    state = ci.read(directory/'final-container-state.json')
    if state.get('exit_code') != 0 or state.get('oom_killed') is not False:
        raise ValueError('Final offline audit process did not finish successfully')
    ci.validate_report(ci.read(directory/'final/runtime-validation.json'), values)
    ci.validate_inventory(ci.read(directory/'installed-inventory.json'), backend, commit)
    image = ci.read(directory/'candidate-image-inspect.json')
    ci.validate_image(image, values)
    if image['Id'] != record['image_id']:
        raise ValueError('Tested image ID differs from publication')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    destination = ROOT/'.h3-experimental-evidence/sr-native/imports'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    destination.mkdir(parents=True)
    with zipfile.ZipFile(args.archive) as bundle:
        for entry in bundle.infolist():
            relative = ops.LEGACY.safe_relative(entry.filename.rstrip('/'))
            if ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Evidence cannot contain symlinks')
            path = destination.joinpath(*relative.parts)
            if entry.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, path.open('xb') as output:
                    shutil.copyfileobj(source, output)
    records = list(destination.rglob('image.json'))
    if len(records) != 1:
        raise ValueError('Import one completed backend artifact at a time')
    record = verify(records[0].parent)
    print(json.dumps({'status':'PASS_IMPORTED_PUBLICATION_RECORD_ONLY', 'backend':record['backend'],
                      'deployment_image':record['deployment_image'], 'directory':str(destination),
                      'not_verified':['Independent registry pull', 'GPU inference', 'Human quality']}, indent=2))


if __name__ == '__main__':
    main()

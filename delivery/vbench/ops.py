"""Windows operator commands. Nothing publishes or touches K8s unless explicitly invoked.

No credentials are accepted here. Git uses its existing credential helper.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(os.environ.get('VBENCH_DELIVERY', r'D:\AI\_workspace\VBench_Image_20260930'))
IMAGE_REPO = 'registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-testing/vench'
CONTEXT = 'server.teleport.hd-04.zetyun.cn-hd04-cci-k8s'
BRANCH = 'DingoRouter-vbench-20260930'
CONFIG = ROOT / '.github/dingo-images.json'
LOCKS = ROOT / 'container/vbench/locks'
ALLOWED = ['.github/dingo-images.json', '.github/scripts/prepare_dingo_image_matrix.py',
           '.github/scripts/vbench_ci.py', '.github/scripts/test_vbench_ci.py',
           '.github/workflows/dingo-router-ci.yml', 'container/vbench', 'delivery/vbench', '.gitignore',
           'container/compliance/policy/licenses.toml']


def run(args, **kw):
    return subprocess.run([str(x) for x in args], cwd=ROOT, check=True, **kw)


def git(*args):
    return run(['git', *args], capture_output=True, text=True, encoding='utf-8').stdout.strip()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def current_fingerprint():
    m = load_module('vbench_fingerprint', ROOT/'container/vbench/scripts/fingerprint.py')
    return m.collect(ROOT/'container/vbench', ROOT/'container/compliance')


def check():
    import yaml
    files = list((ROOT/'container/vbench').rglob('*.py')) + list((ROOT/'delivery/vbench').glob('*.py'))
    files += [ROOT/'.github/scripts/prepare_dingo_image_matrix.py', ROOT/'.github/scripts/vbench_ci.py']
    for f in files:
        ast.parse(f.read_text(encoding='utf-8'), filename=str(f))
    yaml.load((ROOT/'.github/workflows/dingo-router-ci.yml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    matrix = load_module('vbench_matrix', ROOT/'.github/scripts/prepare_dingo_image_matrix.py')
    rows = matrix.build_matrix(cfg, git('rev-parse', 'HEAD'), 'configured')
    if len(rows) != 1 or rows[0]['framework'] != 'vbench' or not rows[0]['image'].startswith(IMAGE_REPO+':'):
        raise ValueError('Configured matrix must contain exactly the approved VBench repository')
    if rows[0]['builder_image']:
        raise ValueError('VBench must not use the Dynamo native builder')
    if cfg['vbench']['phase'] == 'publish':
        verify_imported_locks()
    run(['git', 'diff', '--check'])
    run(['git', 'diff', '--cached', '--check'])
    tests = ([ROOT/'.github/scripts/test_vbench_ci.py']
             + sorted((ROOT/'container/vbench').rglob('test_*.py'))
             + sorted((ROOT/'delivery/vbench').glob('test_*.py')))
    existing = [p for p in tests if p.exists()]
    if not existing:
        raise ValueError('Verification tests missing')
    run([sys.executable, '-B', '-m', 'pytest', '-q', '--noconftest', '-c', ROOT/'delivery/vbench/pytest.ini', *existing])
    DEST.mkdir(parents=True, exist_ok=True)
    evidence = {'status': 'PASS_LOCAL_ONLY', 'phase': cfg['vbench']['phase'],
                'git_head': git('rev-parse', 'HEAD'), 'selected_image': rows[0]['image'],
                'not_run': ['Docker build', 'Linux dependency resolve', 'model loading', 'Harbor push/pull', 'GPU scoring']}
    out = DEST / ('local-check-'+stamp()+'.json')
    out.write_text(json.dumps(evidence, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(evidence, indent=2))
    print('Evidence:', out)


def safe_zip_members(z):
    if sum(i.file_size for i in z.infolist()) > 256*1024*1024:
        raise ValueError('Preparation artifact too large: expected metadata, not model weights')
    seen = set()
    for i in z.infolist():
        p = Path(i.filename.replace('\\', '/'))
        if p.is_absolute() or '..' in p.parts or ':' in i.filename or i.filename.startswith(('/', '\\')):
            raise ValueError('Unsafe ZIP path')
        key = '/'.join(p.parts).casefold()
        if key in seen:
            raise ValueError('Duplicate ZIP member')
        seen.add(key)
        if (i.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('Symlink in evidence ZIP')
    return z.infolist()


def import_preparation(archive):
    archive = Path(archive).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='vbench-evidence-') as td:
        staging = Path(td)
        with zipfile.ZipFile(archive) as z:
            safe_zip_members(z)
            if z.testzip() is not None:
                raise ValueError('ZIP CRC validation failed')
            z.extractall(staging)
        candidates = list(staging.rglob('input-fingerprint.json'))
        if len(candidates) != 1:
            raise ValueError('Expected one preparation output root')
        src = candidates[0].parent
        expected = json.loads(candidates[0].read_text(encoding='utf-8'))
        if expected != current_fingerprint():
            raise ValueError('Artifact build inputs differ from this checkout; use matching source commit')
        identity = json.loads((src/'build-identity.json').read_text(encoding='utf-8'))
        if identity.get('phase') != 'prepare' or identity.get('dynamo_source_commit') != git('rev-parse', 'HEAD'):
            raise ValueError('Artifact is not a preparation for current HEAD')
        base = json.loads((src/'base.lock.json').read_text(encoding='utf-8'))
        if base.get('image') != identity.get('base_image') or base.get('platform') != 'linux/amd64':
            raise ValueError('Base identity mismatch')
        asset = json.loads((src/'assets.lock.json').read_text(encoding='utf-8'))
        if not asset:
            raise ValueError('Empty asset lock')
        report = json.loads((src/'legal/license-status.json').read_text(encoding='utf-8'))
        required = ['base.lock.json', 'requirements.lock', 'assets.lock.json', 'input-fingerprint.json']
        for name in required:
            if not (src/name).is_file() or not (src/name).stat().st_size:
                raise ValueError('Missing or empty lock '+name)
        evidence = DEST / ('preparation-'+stamp())
        shutil.copytree(src, evidence)
        (evidence/'downloaded-artifact.json').write_text(json.dumps({'sha256': sha(archive), 'source_commit': identity['dynamo_source_commit']}, indent=2)+'\n', encoding='utf-8')
        LOCKS.mkdir(parents=True, exist_ok=True)
        for name in required:
            shutil.copy2(src/name, LOCKS/name)
        shutil.copy2(src/'legal/license-status.json', LOCKS/'preparation-license-status.json')
        print('Imported locks. Full evidence:', evidence)
        print('License policy pass:', report.get('policy_pass'))
        print('This import does not commit, push, or change phase.')


def verify_imported_locks():
    for name in ['base.lock.json', 'requirements.lock', 'assets.lock.json', 'input-fingerprint.json', 'preparation-license-status.json']:
        if not (LOCKS/name).is_file():
            raise ValueError('Missing '+name+'; run prepare and import its artifact first')
    if json.loads((LOCKS/'input-fingerprint.json').read_text(encoding='utf-8')) != current_fingerprint():
        raise ValueError('Build inputs changed; prepare again')
    report = json.loads((LOCKS/'preparation-license-status.json').read_text(encoding='utf-8'))
    if report.get('policy_pass') is not True or report.get('violations'):
        raise ValueError('Unresolved repository license policy findings. Read imported legal/license-status.json. '
                         'After reviewed policy corrections, prepare again; no automatic policy bypass is provided.')


def set_phase(phase):
    if phase == 'publish':
        verify_imported_locks()
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    cfg['vbench']['phase'] = phase
    CONFIG.write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')
    print('Phase set to', phase, '; not pushed.')


def submit(phase):
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    if cfg['vbench']['phase'] != phase:
        raise ValueError('Use set-phase '+phase+' before submitting this phase')
    if git('remote', 'get-url', 'origin').rstrip('/') != 'https://github.com/githubgxll/dynamo.git':
        raise ValueError('Unexpected origin; refusing to push')
    branch = git('branch', '--show-current')
    if branch != 'vbench/six-dimensions':
        raise ValueError('Expected local VBench branch; refusing to submit '+branch)
    check()
    changed = set(git('diff', '--name-only', 'HEAD').splitlines()) | set(git('ls-files', '--others', '--exclude-standard').splitlines())
    for file in changed:
        if not any(file == p or file.startswith(p+'/') for p in ALLOWED):
            raise ValueError('Unrelated change must be handled first: '+file)
    paths = [p for p in ALLOWED if (ROOT/p).exists()]
    run(['git', 'add', '--', *paths])
    run(['git', 'diff', '--cached', '--check'])
    staged = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode
    if staged == 1:
        run(['git', 'commit', '-m', 'Prepare VBench six-dimension '+phase+' image workflow'])
    elif staged != 0:
        raise ValueError('Could not inspect staged changes')
    # Non-force push; normal Git rejects a conflicting remote branch.
    run(['git', '-c', 'http.proxy=http://127.0.0.1:17890', 'push', 'origin', 'HEAD:refs/heads/'+BRANCH])
    print('Submitted', git('rev-parse', 'HEAD'), 'phase', phase)
    print('https://github.com/githubgxll/dynamo/actions')


def pod_manifest(args):
    if not re.fullmatch(re.escape(IMAGE_REPO)+r'@sha256:[a-f0-9]{64}', args.image):
        raise ValueError('Use the published VBench digest, not a tag or another repository')
    if not re.fullmatch(r'jirx-[a-z0-9-]{1,57}[a-z0-9]', args.name):
        raise ValueError('A valid jirx- Pod name is required')
    doc = {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': args.name, 'namespace': args.namespace,
           'labels': {'app.kubernetes.io/name': 'jirx-vbench', 'app.kubernetes.io/component': 'offline-evaluator'}},
           'spec': {'restartPolicy': 'Never', 'automountServiceAccountToken': False, 'enableServiceLinks': False,
           'nodeSelector': {'kubernetes.io/hostname': 'hd04-gpu1-0063'},
           'tolerations': [{'key':'dc.com/osm-nodepool.business-pool','operator':'Equal','value':'exclusive','effect':'NoSchedule'}],
           'containers': [{'name':'evaluator', 'image':args.image, 'imagePullPolicy':'IfNotPresent',
           'command':['sleep','infinity'], 'env':[{'name':'XDG_CACHE_HOME','value':'/data/cache'}, {'name':'HF_HOME','value':'/data/cache/huggingface'}, {'name':'VBENCH_IMAGE_REFERENCE','value':args.image}],
           'resources': {'requests':{'cpu':'8' if args.gpus == 1 else '32', 'memory':'32Gi' if args.gpus == 1 else '128Gi', 'ephemeral-storage':'40Gi', 'nvidia.com/gpu-h100-80gb-hbm3':str(args.gpus)},
                         'limits':{'cpu':'16' if args.gpus == 1 else '64','memory':'64Gi' if args.gpus == 1 else '256Gi','ephemeral-storage':'120Gi','nvidia.com/gpu-h100-80gb-hbm3':str(args.gpus)}},
           'volumeMounts':[{'name':'data','mountPath':'/data'}, {'name':'shm','mountPath':'/dev/shm'}]}],
           'volumes':[{'name':'data','emptyDir':{'sizeLimit':'100Gi'}},{'name':'shm','emptyDir':{'medium':'Memory','sizeLimit':'16Gi'}}]}}
    if args.pull_secret:
        if not re.fullmatch(r'[a-z0-9]([-a-z0-9.]*[a-z0-9])?', args.pull_secret):
            raise ValueError('Invalid secret name')
        doc['spec']['imagePullSecrets'] = [{'name':args.pull_secret}]
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
        f.write('\n')
    print('Saved only; no Kubernetes action:', out)


def kubectl(namespace, pod, *command):
    if namespace not in {'token-factory', 'elm-test'} or not pod.startswith('jirx-'):
        raise ValueError('Refusing a non-user target')
    return ['kubectl', '--context', CONTEXT, '-n', namespace, 'exec', '-i', pod, '-c', 'evaluator', '--', *command]


def upload(args):
    source = Path(args.file).resolve(strict=True)
    if source.suffix.lower() != '.mp4':
        raise ValueError('Smoke input must be an original MP4')
    # The remote path is fixed. Exclusive creation prevents accidental replacement.
    code = "import pathlib,sys,hashlib;p=pathlib.Path('/data/input');p.mkdir(exist_ok=True);f=p/'sample.mp4';h=hashlib.sha256();o=f.open('xb');\nfor b in iter(lambda:sys.stdin.buffer.read(1048576),b''): o.write(b);h.update(b)\no.close();print(h.hexdigest())"
    with source.open('rb') as f:
        result = run(kubectl(args.namespace, args.pod, '/opt/venv/bin/python', '-c', code), stdin=f, capture_output=True)
    if result.stdout.decode().strip() != sha(source):
        raise ValueError('Uploaded MP4 checksum mismatch')
    print('Uploaded unchanged, SHA256 verified: /data/input/sample.mp4')


def export_smoke(args):
    out = Path(args.output).resolve()
    if out.exists() or out.with_suffix(out.suffix+'.part').exists():
        raise ValueError('Output or partial file already exists; choose a fresh filename')
    out.parent.mkdir(parents=True, exist_ok=True)
    # A unique remote archive allows retry after an interrupted transfer without
    # overwriting earlier evidence or requiring deletion inside the Pod.
    remote = '/data/smoke-export-'+stamp()+'.zip'
    code = ("import pathlib,zipfile,sys,hashlib;p=pathlib.Path('/data/smoke');assert p.is_dir();"
            "zpath=pathlib.Path("+repr(remote)+");\n"
            "with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED) as z:\n"
            " for root,prefix in [(p,'smoke'),(pathlib.Path('/opt/vbench-build-info'),'build-info'),(pathlib.Path('/legal'),'legal')]:\n"
            "  for f in sorted(root.rglob('*')):\n"
            "   if f.is_file():z.write(f,prefix+'/'+str(f.relative_to(root)))\n"
            " for f in sorted(pathlib.Path('/data').glob('gpu-check*')):\n"
            "  if f.is_file():z.write(f,'gpu-check/'+f.name)\n"
            "print(hashlib.sha256(zpath.read_bytes()).hexdigest())")
    digest = run(kubectl(args.namespace, args.pod, '/opt/venv/bin/python', '-c', code), capture_output=True, text=True).stdout.strip()
    partial = out.with_suffix(out.suffix+'.part')
    with partial.open('xb') as f:
        run(kubectl(args.namespace, args.pod, '/opt/venv/bin/python', '-c',
                    "import sys,shutil;shutil.copyfileobj(open("+repr(remote)+",'rb'),sys.stdout.buffer)"), stdout=f)
    if sha(partial) != digest:
        raise ValueError('Result archive checksum mismatch; partial file retained')
    with zipfile.ZipFile(partial) as z:
        safe_zip_members(z)
        if z.testzip():
            raise ValueError('Result ZIP CRC failed')
    partial.rename(out)
    print('Downloaded and verified; no Pod was deleted:', out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='action', required=True)
    s.add_parser('check')
    a = s.add_parser('import-preparation'); a.add_argument('zip')
    a = s.add_parser('set-phase'); a.add_argument('phase', choices=['prepare','publish'])
    a = s.add_parser('submit'); a.add_argument('phase', choices=['prepare','publish'])
    a = s.add_parser('pod'); a.add_argument('--image', required=True); a.add_argument('--name', required=True)
    a.add_argument('--namespace', choices=['token-factory','elm-test'], default='token-factory')
    a.add_argument('--gpus', type=int, choices=[1,4], default=1); a.add_argument('--output', required=True)
    secret = a.add_mutually_exclusive_group(required=True)
    secret.add_argument('--pull-secret'); secret.add_argument('--use-serviceaccount-pullsecrets', action='store_true')
    for command, field in [('upload-smoke','file'),('export-smoke','output')]:
        a = s.add_parser(command); a.add_argument('--'+field, required=True); a.add_argument('--pod', required=True)
        a.add_argument('--namespace', choices=['token-factory','elm-test'], default='token-factory')
    a = p.parse_args()
    if a.action == 'check': check()
    elif a.action == 'import-preparation': import_preparation(a.zip)
    elif a.action == 'set-phase': set_phase(a.phase)
    elif a.action == 'submit': submit(a.phase)
    elif a.action == 'pod': pod_manifest(a)
    elif a.action == 'upload-smoke': upload(a)
    elif a.action == 'export-smoke': export_smoke(a)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as e:
        print('STOP:', e, file=sys.stderr)
        raise SystemExit(1)

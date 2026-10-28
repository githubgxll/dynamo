"""Offline recipe/regression checks. Does not build images or claim GPU execution."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    import yaml
    lock = json.loads((HERE/'source-lock.json').read_text(encoding='utf-8'))
    dependency_counts = {}
    for backend, model in lock['models'].items():
        tag = model['base_tag_reviewed'].split(':', 1)[1]
        manifest = (HERE/'base-evidence'/(tag+'.manifest.json')).read_bytes()
        digest = 'sha256:' + hashlib.sha256(manifest).hexdigest()
        if model['base_image'] != 'nvidia/cuda@' + digest:
            raise ValueError('Archived base manifest differs from fixed identity: ' + backend)
        index = json.loads((HERE/'base-evidence'/(tag+'.index.json')).read_text())
        matches = [item for item in index['manifests'] if item['platform']['os'] == 'linux' and item['platform']['architecture'] == 'amd64']
        if len(matches) != 1 or matches[0]['digest'] != digest:
            raise ValueError('Base manifest is not the reviewed amd64 child: ' + backend)
        requirements = (ROOT/'container/sr-native'/backend/'requirements.lock').read_text(encoding='utf-8')
        dependency_counts[backend] = len(re.findall(r'(?m)^[a-zA-Z0-9_.-]+==', requirements))
    workflow = yaml.load((ROOT / '.github/workflows/dingo-router-ci.yml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    job = workflow['jobs']['sr-native']
    blocks = [step['run'] for step in job['steps'] if 'run' in step]
    out = ROOT / '.h3-experimental-evidence/sr-native/local' / datetime.now(timezone.utc).strftime('checks-%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    env = dict(os.environ, TEMP=str(out), TMP=str(out), TMPDIR=str(out))
    tests = []
    for name, path in [('shared', ROOT/'container/sr-native/shared/tests'),
                       ('seedvr2', ROOT/'container/sr-native/seedvr2'),
                       ('flashvsr-plus', ROOT/'container/sr-native/flashvsr-plus/tests'),
                       ('submission', HERE/'tests')]:
        tests.append((name, [sys.executable, '-B', '-X', 'utf8', '-m', 'unittest', 'discover', '-s', str(path), '-p', 'test_*.py', '-v']))
    tests.append(('ci-publication', [sys.executable, '-B', '-X', 'utf8', str(ROOT/'.github/scripts/test_sr_native_ci.py')]))
    counts, skipped = {}, {}
    for name, command in tests:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, encoding='utf-8', errors='replace')
        output = result.stdout + result.stderr
        (out/(name+'.txt')).write_text(output, encoding='utf-8', newline='\n')
        if result.returncode:
            print(output[-16000:])
            raise RuntimeError(name + ' failed; evidence: ' + str(out))
        counts[name] = sum(map(int, re.findall(r'Ran (\d+) tests?', output)))
        skipped[name] = sum(map(int, re.findall(r'OK \(skipped=(\d+)\)', output)))
        if not counts[name]:
            raise RuntimeError(name + ' ran zero tests')
        print('PASS ' + name)
    files = list((ROOT/'container/sr-native').rglob('*.py')) + list(HERE.rglob('*.py'))
    files += [ROOT/'.github/scripts/sr_native_ci.py', ROOT/'.github/scripts/test_sr_native_ci.py']
    for path in files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 10))
    shell_files = list((ROOT/'container/sr-native').rglob('*.sh'))
    for index, block in enumerate(blocks):
        path = out/f'workflow-{index}.sh'
        path.write_text(block, encoding='utf-8', newline='\n')
        shell_files.append(path)
    bash = r'D:\workingtools\Git\bin\bash.exe' if os.name == 'nt' else '/bin/bash'
    for path in shell_files:
        result = subprocess.run([bash, '-n', str(path)], cwd=ROOT, env=env, capture_output=True, encoding='utf-8', errors='replace')
        if result.returncode:
            raise RuntimeError('Shell syntax check failed: ' + str(path) + '\n' + result.stderr)
    report = {'status':'PASS_LOCAL_RECIPE_AND_REGRESSIONS_ONLY', 'regression_counts':counts,
              'regression_tests':sum(counts.values()), 'skipped':sum(skipped.values()),
              'python_syntax_files':len(files), 'shell_files':len(shell_files), 'evidence_directory':str(out),
              'resolved_linux_python310_hash_locked_packages':dependency_counts,
              'not_run':['Linux package installation', 'CUDA extension compilation', 'Docker build',
                         'Harbor push/pull', 'H100 kernel/17-frame inference', 'Native-vs-Omni quality comparison']}
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

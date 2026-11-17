# SPDX-License-Identifier: Apache-2.0
"""Exercise the builder's certificate bootstrap with unavailable mirrorlists."""

import os
from pathlib import Path
import re
import subprocess

import pytest

pytestmark = [
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.skipif(os.name != "posix", reason="Docker RUN uses a POSIX shell"),
]
ROOT = Path(__file__).resolve().parents[2]


def bootstrap(tmp_path, failures):
    template = (ROOT / "container/templates/wheel_builder.Dockerfile").read_text()
    blocks = [part.split("\n\n", 1)[0] for part in template.split("\nRUN ")[1:]]
    command = next(block for block in blocks if "ca-certificates >= " in block)
    # Docker drops full-line comments before joining continued RUN lines.
    command = "".join(
        line for line in command.splitlines(keepends=True)
        if not line.lstrip().startswith("#")
    ).replace("\\\n", " ")
    command = re.sub(r"^(?:--mount=\S+\s*)+", "", command)
    repos = tmp_path / "repos"
    repos.mkdir()
    repo = repos / "almalinux.repo"
    repo.write_text(
        "[baseos]\nmirrorlist=https://mirrors.almalinux.org/mirrorlist/8/baseos\n"
        "# baseurl=https://repo.almalinux.org/almalinux/8/BaseOS/x86_64/os/\n"
        "gpgcheck=1\nsslverify=1\ngpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-AlmaLinux\n"
        "[extras]\nmirrorlist=https://mirrors.almalinux.org/mirrorlist/8/extras\n"
        "# baseurl=https://repo.almalinux.org/almalinux/8/extras/x86_64/os/\n"
        "gpgcheck=1\nsslverify=1\n"
    )
    command = command.replace("/etc/yum.repos.d", str(repos))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    dnf = bin_dir / "dnf"
    # Model an inaccessible public mirrorlist and an independent Extras outage.
    # All repository/configuration decisions are made by the actual RUN shell.
    dnf.write_text(
        "#!/usr/bin/python3\n"
        "import os,pathlib,sys\n"
        "root=pathlib.Path(os.environ['TEST_STATE']); args=sys.argv[1:]\n"
        "if 'install' not in args: sys.exit(0)\n"
        "if any(arg.startswith('ca-certificates') for arg in args):\n"
        "    counter=root/'attempts'\n"
        "    count=int(counter.read_text())+1 if counter.exists() else 1\n"
        "    counter.write_text(str(count))\n"
        "    repo=(root/'repos/almalinux.repo').read_text()\n"
        "    if any(line.startswith('mirrorlist=') for line in repo.splitlines()): sys.exit(1)\n"
        "    if '--disablerepo=*' not in args or '--enablerepo=baseos' not in args: sys.exit(1)\n"
        "    if count<=int(os.environ['TEST_FAILURES']): sys.exit(1)\n"
        "    (root/'certificates-installed').touch()\n"
        "else:\n"
        "    if not (root/'certificates-installed').exists(): sys.exit(2)\n"
        "    (root/'dependencies-installed').touch()\n"
    )
    dnf.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        TEST_STATE=str(tmp_path),
        TEST_FAILURES=str(failures),
    )
    result = subprocess.run(
        ["/bin/sh", "-c", command], env=env, capture_output=True, text=True, timeout=10
    )
    return result, repo


def test_certificate_bootstrap_does_not_depend_on_extras_or_mirrorlist(tmp_path):
    result, repo = bootstrap(tmp_path, failures=0)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "certificates-installed").exists()
    assert (tmp_path / "dependencies-installed").exists()
    assert "gpgcheck=1\nsslverify=1\n" in repo.read_text()
    assert "gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-AlmaLinux\n" in repo.read_text()


def test_transient_certificate_download_failure_recovers(tmp_path):
    result, _ = bootstrap(tmp_path, failures=2)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "attempts").read_text() == "3"
    assert (tmp_path / "dependencies-installed").exists()


def test_persistent_certificate_failure_blocks_later_installations(tmp_path):
    result, _ = bootstrap(tmp_path, failures=100)
    assert result.returncode != 0
    assert (tmp_path / "attempts").read_text() == "5"
    assert not (tmp_path / "certificates-installed").exists()
    assert not (tmp_path / "dependencies-installed").exists()

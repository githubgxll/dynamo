# SPDX-License-Identifier: Apache-2.0
"""Exercise the certificate-bootstrap shell without modifying host APT state."""

import os
from pathlib import Path
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


@pytest.fixture(params=["dynamo_base", "sglang_runtime"])
def bootstrap_stage(request):
    return request.param


def bootstrap(tmp_path, failures, install_status=0, *, stage_name):
    template = (ROOT / f"container/templates/{stage_name}.Dockerfile").read_text()
    stage_marker = "pre_runtime" if stage_name == "sglang_runtime" else "dynamo_base"
    stage = template.split(f" AS {stage_marker}", 1)[1]
    command = stage.split("RUN ", 1)[1].split("\n\n", 1)[0].replace("\\\n", "\n")
    sources = tmp_path / "sources"
    sources.mkdir()
    legacy = sources / "sources.list"
    legacy.write_text("deb http://archive.ubuntu.com/ubuntu noble main multiverse\n")
    parts = sources / "parts"
    parts.mkdir()
    deb822 = parts / "ubuntu.sources"
    deb822.write_text(
        "Types: deb\nURIs: http://security.ubuntu.com/ubuntu\n"
        "Suites: noble-security\nComponents: main multiverse\n"
        "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n"
    )
    vendor = parts / "vendor.list"
    vendor.write_text("deb https://developer.download.nvidia.com/example /\n")
    lists = tmp_path / "lists"
    lists.mkdir()
    command = command.replace("/etc/apt/sources.list.d", str(parts))
    command = command.replace("/etc/apt/sources.list", str(legacy))
    command = command.replace("/var/lib/apt/lists", str(lists))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    apt = bin_dir / "apt-get"
    apt.write_text(
        "#!/bin/sh\n"
        "case \" $* \" in\n"
        "  *' update '*)\n"
        "    count=0; [ ! -f \"$TEST_STATE/attempts\" ] || count=$(cat \"$TEST_STATE/attempts\");\n"
        "    count=$((count+1)); echo $count > \"$TEST_STATE/attempts\";\n"
        "    [ $count -gt \"$TEST_FAILURES\" ] || exit 100;;\n"
        "  *' install '*) touch \"$TEST_STATE/installed\"; exit \"$TEST_INSTALL_STATUS\";;\n"
        "  *) exit 2;;\n"
        "esac\n"
    )
    apt.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", TEST_STATE=str(tmp_path),
               TEST_FAILURES=str(failures), TEST_INSTALL_STATUS=str(install_status))
    result = subprocess.run(["/bin/sh", "-c", command], env=env, capture_output=True, text=True)
    return result, legacy, deb822, vendor


def test_temporary_index_failure_recovers_without_changing_signing_policy(tmp_path, bootstrap_stage):
    result, legacy, deb822, vendor = bootstrap(tmp_path, failures=2, stage_name=bootstrap_stage)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "installed").exists()
    assert (tmp_path / "attempts").read_text().strip() == "3"
    assert legacy.read_text() == "deb http://mirrors.huaweicloud.com/ubuntu noble main multiverse\n"
    assert deb822.read_text() == (
        "Types: deb\nURIs: http://mirrors.huaweicloud.com/ubuntu\n"
        "Suites: noble-security\nComponents: main multiverse\n"
        "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n"
    )
    assert vendor.read_text() == "deb https://developer.download.nvidia.com/example /\n"


def test_persistent_index_failure_never_installs_from_stale_lists(tmp_path, bootstrap_stage):
    result, *_ = bootstrap(tmp_path, failures=100, stage_name=bootstrap_stage)
    assert result.returncode != 0
    assert (tmp_path / "attempts").read_text().strip() == "5"
    assert not (tmp_path / "installed").exists()


def test_certificate_install_failure_is_not_hidden(tmp_path, bootstrap_stage):
    result, *_ = bootstrap(tmp_path, failures=0, install_status=100, stage_name=bootstrap_stage)
    assert result.returncode == 100

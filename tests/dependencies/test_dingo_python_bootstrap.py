# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the Dingo Python startup bootstrap installer.

container/deps/sglang/install_dingo_python_bootstrap.py writes a .pth hook
(instead of a sitecustomize.py, which base-image apport hooks shadow) plus a
bootstrap module into a site-packages directory. These tests install it into a
throwaway venv and observe real interpreter startup behavior:

* fresh interpreters and multiprocessing spawn children get logging.config,
* mixed-environment opt-in (DINGO_SITE_PACKAGES) pins dynamo to the selected
  copy while keeping the base dependency stack first on sys.path,
* a dangling or dynamo-less DINGO_SITE_PACKAGES aborts startup instead of
  silently falling back to a stale dynamo (the ABI-mismatch failure mode seen
  in the bj11 deployment).
"""

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "container/deps/sglang/install_dingo_python_bootstrap.py"


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        candidate = venv_dir / "Scripts" / "python.exe"
    else:
        candidate = venv_dir / "bin" / "python3"
        if not candidate.exists():
            candidate = venv_dir / "bin" / "python"
    assert candidate.exists(), f"venv interpreter missing: {candidate}"
    return candidate


def _purelib(venv_dir: Path) -> Path:
    """The venv's own purelib directory, resolved by the venv interpreter."""
    out = subprocess.run(
        [
            str(_venv_python(venv_dir)),
            "-c",
            'import sysconfig; print(sysconfig.get_path("purelib"))',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


@pytest.fixture(scope="module")
def bootstrapped_venv(tmp_path_factory) -> dict:
    """A throwaway venv with the real bootstrap installed by the real installer.

    dynamo is laid out as a namespace package (no top-level __init__.py) in
    both the base site-packages and the selected Dingo site-packages, matching
    how the dynamo runtime wheels install.
    """
    venv_dir = tmp_path_factory.mktemp("bootstrap-venv")
    venv.create(venv_dir, with_pip=False)
    python = _venv_python(venv_dir)
    site_packages = _purelib(venv_dir)

    install = subprocess.run(
        [sys.executable, str(INSTALLER), "--site-packages", str(site_packages)],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, f"installer failed: {install.stderr}"

    # Base image stand-ins: a namespace dynamo plus regular dependencies.
    base_dynamo = site_packages / "dynamo"
    (base_dynamo / "common").mkdir(parents=True)
    (base_dynamo / "common" / "__init__.py").write_text(
        "ORIGIN = 'base'\n", encoding="utf-8"
    )
    (base_dynamo / "legacy_only.py").write_text(
        "ORIGIN = 'base'\n", encoding="utf-8"
    )
    (site_packages / "base_only_dep.py").write_text(
        "ORIGIN = 'base'\n", encoding="utf-8"
    )
    (site_packages / "shared_dep.py").write_text("ORIGIN = 'base'\n", encoding="utf-8")

    # The selected Dingo site-packages: newer dynamo plus its extra deps.
    dingo_site = tmp_path_factory.mktemp("dingo-site")
    selected_dynamo = dingo_site / "dynamo"
    (selected_dynamo / "common").mkdir(parents=True)
    (selected_dynamo / "common" / "__init__.py").write_text(
        "ORIGIN = 'selected'\n", encoding="utf-8"
    )
    (dingo_site / "shared_dep.py").write_text(
        "ORIGIN = 'selected'\n", encoding="utf-8"
    )

    return {
        "python": python,
        "site_packages": site_packages,
        "dingo_site": dingo_site,
    }


def _run(
    bootstrapped_venv: dict,
    code: str,
    env_extra: dict | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Isolate from any host-level opt-in so the default cases are real.
    env.pop("DINGO_SITE_PACKAGES", None)
    env.update(env_extra or {})
    return subprocess.run(
        [str(bootstrapped_venv["python"]), "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )


def test_fresh_interpreter_configures_logging(bootstrapped_venv):
    code = """
import logging
logging.config.dictConfig({"version": 1, "root": {"level": "DEBUG", "handlers": []}})
assert logging.getLogger("sglang").getEffectiveLevel() == logging.DEBUG
"""
    result = _run(bootstrapped_venv, code)
    assert result.returncode == 0, result.stderr


def test_spawn_child_uses_logging_and_selected_sdk(bootstrapped_venv, tmp_path):
    parent = tmp_path / "spawn_parent.py"
    parent.write_text(
        "import multiprocessing as mp\n"
        "def child(q):\n"
        "    import logging, dynamo.common, shared_dep\n"
        "    logging.config.dictConfig({'version': 1, 'root': {'level': 'INFO', 'handlers': []}})\n"
        "    q.put((logging.getLogger('sglang').getEffectiveLevel(), dynamo.common.ORIGIN, shared_dep.ORIGIN))\n"
        "if __name__ == '__main__':\n"
        "    ctx = mp.get_context('spawn')\n"
        "    q = ctx.SimpleQueue()\n"
        "    p = ctx.Process(target=child, args=(q,))\n"
        "    p.start()\n"
        "    p.join(30)\n"
        "    assert p.exitcode == 0, p.exitcode\n"
        "    assert q.get() == (20, 'selected', 'base')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(bootstrapped_venv["python"]), str(parent)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DINGO_SITE_PACKAGES": str(bootstrapped_venv["dingo_site"]),
        },
    )
    assert result.returncode == 0, result.stderr


def test_default_environment_uses_base_dynamo(bootstrapped_venv):
    """Without DINGO_SITE_PACKAGES the interpreter keeps the base copy."""
    code = """
import dynamo.common as common
assert common.ORIGIN == "base"
"""
    result = _run(bootstrapped_venv, code)
    assert result.returncode == 0, result.stderr


def test_selected_environment_pins_dynamo_to_dingo_site(bootstrapped_venv):
    """Select the new SDK without exposing stale base-only SDK modules."""
    code = """
import dynamo.common, importlib.util
assert dynamo.common.ORIGIN == "selected"
assert importlib.util.find_spec("dynamo.legacy_only") is None
"""
    result = _run(
        bootstrapped_venv,
        code,
        env_extra={"DINGO_SITE_PACKAGES": str(bootstrapped_venv["dingo_site"])},
    )
    assert result.returncode == 0, result.stderr


def test_selected_environment_keeps_base_stack_priority(bootstrapped_venv):
    """Keep the base dependencies ahead of the selected SDK's dependencies."""
    code = """
import base_only_dep, shared_dep
assert shared_dep.ORIGIN == "base"
assert base_only_dep.ORIGIN == "base"
"""
    result = _run(
        bootstrapped_venv,
        code,
        env_extra={"DINGO_SITE_PACKAGES": str(bootstrapped_venv["dingo_site"])},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("missing-dir", "no-such-dir"),
        ("dynamo-less-dir", "empty-dingo-site"),
    ],
)
def test_broken_selection_aborts_startup(bootstrapped_venv, tmp_path, label, value):
    """A selection without a dynamo/ directory must abort startup rather than
    silently run the stale base dynamo (ABI-mismatch failure mode)."""
    if label == "dynamo-less-dir":
        target = tmp_path / value
        target.mkdir()
    else:
        target = bootstrapped_venv["site_packages"] / value
    result = _run(
        bootstrapped_venv,
        "print('should not get here')",
        env_extra={"DINGO_SITE_PACKAGES": str(target)},
    )
    assert result.returncode != 0
    assert "DINGO_SITE_PACKAGES" in result.stderr
    assert "should not get here" not in result.stdout


def test_relative_selection_aborts_startup(bootstrapped_venv):
    """The selection is normalized and joined onto submodule imports, so a
    relative path is rejected outright."""
    result = _run(
        bootstrapped_venv,
        "print('should not get here')",
        env_extra={"DINGO_SITE_PACKAGES": "relative/path"},
    )
    assert result.returncode != 0
    assert "DINGO_SITE_PACKAGES" in result.stderr
    assert "should not get here" not in result.stdout

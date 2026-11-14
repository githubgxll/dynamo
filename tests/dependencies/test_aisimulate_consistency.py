# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Dynamo on one published AISimulate release."""

from __future__ import annotations
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from importlib import metadata
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from tests.wheels.smoke_install import AISIMULATE_FIND_LINKS

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

ROOT = Path(__file__).resolve().parents[2]
AISIMULATE_REQUIREMENTS = ROOT / "container/deps/requirements.aisimulate.txt"
LOCKFILES = (
    ROOT / "Cargo.lock",
    ROOT / "lib/bindings/python/Cargo.lock",
    ROOT / "lib/bindings/kvbm/Cargo.lock",
)


def _root_configs() -> tuple[dict, dict]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    with (ROOT / "Cargo.toml").open("rb") as handle:
        cargo = tomllib.load(handle)
    return pyproject, cargo


def _python_requirement(pyproject: dict) -> Requirement:
    matches = [
        Requirement(requirement)
        for requirement in pyproject["project"]["dependencies"]
        if canonicalize_name(Requirement(requirement).name) == "aisimulate"
    ]
    assert len(matches) == 1, "ai-dynamo must declare one AISimulate dependency"
    return matches[0]


def _requirements_file_aisimulate_requirement(path: Path) -> Requirement:
    matches: list[Requirement] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement or requirement.startswith("--"):
            continue
        parsed = Requirement(requirement)
        if canonicalize_name(parsed.name) == "aisimulate":
            matches.append(parsed)
    assert len(matches) == 1, f"{path} must declare one AISimulate dependency"
    return matches[0]


def _exact_version(requirement: Requirement) -> Version:
    assert requirement.url is None, "AISimulate must resolve from PyPI"
    specifiers = list(requirement.specifier)
    assert len(specifiers) == 1 and specifiers[0].operator == "==", (
        "AISimulate must use one exact PyPI version"
    )
    return Version(specifiers[0].version)


def _locked_cargo_version(path: Path) -> Version:
    with path.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    matches = [package for package in packages if package["name"] == "aisimulate-core"]
    assert len(matches) == 1, f"expected one aisimulate-core package in {path}"

    package = matches[0]
    assert package.get("source") == (
        "registry+https://github.com/rust-lang/crates.io-index"
    ), f"aisimulate-core must resolve from crates.io in {path}"
    assert re.fullmatch(r"[0-9a-f]{64}", str(package.get("checksum", ""))), (
        f"aisimulate-core must have a registry checksum in {path}"
    )
    return Version(str(package["version"]))


def test_dynamo_pins_matching_published_aisimulate_releases() -> None:
    pyproject, cargo = _root_configs()
    python_requirement = _python_requirement(pyproject)
    python_version = _exact_version(python_requirement)
    container_requirement = _requirements_file_aisimulate_requirement(
        AISIMULATE_REQUIREMENTS
    )

    assert python_requirement.marker is not None
    environment = default_environment()
    environment["python_version"] = "3.10"
    assert not python_requirement.marker.evaluate(environment)
    environment["python_version"] = "3.11"
    assert python_requirement.marker.evaluate(environment)
    environment["python_version"] = "3.12"
    assert python_requirement.marker.evaluate(environment)
    environment["python_version"] = "3.13"
    assert python_requirement.marker.evaluate(environment)
    environment["python_version"] = "3.14"
    assert not python_requirement.marker.evaluate(environment)
    assert container_requirement.marker is None
    assert _exact_version(container_requirement) == python_version
    with (ROOT / "benchmarks/pyproject.toml").open("rb") as handle:
        benchmarks = tomllib.load(handle)
    assert _exact_version(_python_requirement(benchmarks)) == python_version
    assert "aisimulate" not in pyproject.get("tool", {}).get("uv", {}).get(
        "sources", {}
    )

    cargo_dependency = cargo["workspace"]["dependencies"]["aisimulate-core"]
    assert not {"path", "git", "rev", "branch", "tag"} & cargo_dependency.keys()
    cargo_requirement = str(cargo_dependency["version"])
    assert cargo_requirement.startswith("="), (
        "aisimulate-core must use one exact crates.io version"
    )
    cargo_version = Version(cargo_requirement.removeprefix("="))

    assert cargo_version == python_version
    assert all(_locked_cargo_version(path) == cargo_version for path in LOCKFILES)


def _staging_module():
    spec = importlib.util.spec_from_file_location(
        "stage_aisimulate", ROOT / "container/deps/stage_aisimulate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rust_source_staging_preserves_transitive_lock(tmp_path, monkeypatch):
    staging = _staging_module()
    root = tmp_path / "build"
    (root / ".cargo").mkdir(parents=True)
    (root / "Cargo.toml").write_text((ROOT / "Cargo.toml").read_text())
    for original in LOCKFILES:
        destination = root / original.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(original.read_text())
    (root / ".cargo/config.toml").write_text("[build]\njobs = 2\n")
    checkout = tmp_path / "source"
    checkout.mkdir()
    (checkout / "LICENSE").write_text("test license")
    monkeypatch.setattr(staging, "prepare_source", lambda *args: checkout)
    before = [tomllib.loads(path.read_text()) for path in LOCKFILES]
    staging.stage_rust_source(root, tmp_path / "deps")
    for original, expected in zip(LOCKFILES, before):
        after = tomllib.loads((root / original.relative_to(ROOT)).read_text())
        for package in expected["package"]:
            if package["name"] == "aisimulate-core":
                package.pop("source")
                package.pop("checksum")
        assert after == expected
    config = tomllib.loads((root / ".cargo/config.toml").read_text())
    assert config["build"]["jobs"] == 2
    assert config["patch"]["crates-io"]["aisimulate-core"]["path"] == str(
        checkout / "crates/core"
    )
    assert len(list((root / "rust-licenses").glob("*/LICENSE"))) == 1
    with pytest.raises(SystemExit, match="existing Cargo source patches"):
        staging.stage_rust_source(root, tmp_path / "deps")


@pytest.mark.parametrize("matching_version", [True, False])
def test_container_stages_the_published_aisimulate_wheel(
    tmp_path, monkeypatch, matching_version
) -> None:
    pyproject, _ = _root_configs()
    version = _exact_version(_python_requirement(pyproject))
    assert (
        _exact_version(
            _requirements_file_aisimulate_requirement(AISIMULATE_REQUIREMENTS)
        )
        == version
    )
    wheel_builder = (ROOT / "container/templates/wheel_builder.Dockerfile").read_text()
    assert "COPY container/deps/requirements.aisimulate.txt" in wheel_builder
    assert (
        "COPY container/deps/stage_aisimulate.py container/deps/aisimulate-sources.json"
        in wheel_builder
    )
    assert "python /opt/dynamo/container/deps/stage_aisimulate.py" in wheel_builder
    assert "COPY aisimulate" not in wheel_builder
    assert "/opt/dynamo/aisimulate" not in wheel_builder
    assert not (ROOT / "aisimulate").exists()
    staging = _staging_module()
    calls = []

    def download(command, **kwargs):
        calls.append(command)
        assert command[:4] == [sys.executable, "-m", "pip", "download"]
        for flag in ("--only-binary=:all:", "--no-deps", "--no-index"):
            assert flag in command
        assert command[command.index("--requirement") + 1] == str(
            AISIMULATE_REQUIREMENTS
        )
        assert command[command.index("--find-links") + 1] == AISIMULATE_FIND_LINKS
        wheel_version = str(version) if matching_version else "0.0.0"
        wheel = Path(command[command.index("--dest") + 1]) / "aisimulate-test.whl"
        with zipfile.ZipFile(wheel, "w") as package:
            package.writestr(
                "aisimulate.dist-info/METADATA",
                f"Name: aisimulate\nVersion: {wheel_version}\n",
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(staging.subprocess, "run", download)
    output = tmp_path / "wheels"
    if matching_version:
        staging.stage(ROOT, output)
        assert [p.name for p in output.iterdir()] == ["aisimulate-test.whl"]
    else:
        with pytest.raises(SystemExit, match="metadata does not match"):
            staging.stage(ROOT, output)
        assert not list(output.iterdir())
    assert len(calls) == 1


def test_aisimulate_source_fallback_rejects_checksum_mismatch(tmp_path, monkeypatch):
    staging = _staging_module()
    calls = []

    def unavailable(command, **kwargs):
        calls.append(command)
        assert command[1:4] == ["-m", "pip", "download"]
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(staging.subprocess, "run", unavailable)
    monkeypatch.setattr(
        staging.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(b"invalid archive"),
    )
    output = tmp_path / "wheels"
    with pytest.raises(SystemExit, match="source archive checksum mismatch"):
        staging.stage(ROOT, output)
    assert len(calls) == 1  # No build may run after the checksum fails.
    assert not list(output.iterdir())


def _aisim_source_archive(tmp_path, version, monkeypatch):
    """Build a real codeload-style aisim tarball for the reviewed commit."""
    commit = json.loads(
        (ROOT / "container/deps/aisimulate-sources.json").read_text()
    )[version]["commit"]
    staging = _staging_module()
    content = tmp_path / "aisim-src"
    root = content / f"aisim-{commit}"
    (root / "scripts").mkdir(parents=True)
    (root / "python/aisimulate").mkdir(parents=True)
    (root / "crates/core").mkdir(parents=True)
    (root / "python/aisimulate/pyproject.toml").write_text(
        '[project]\nname = "aisimulate"\nversion = "0.13.0"\n'
    )
    (root / "crates/core/Cargo.toml").write_text(
        '[package]\nname = "aisimulate-core"\nversion = "0.13.0"\n'
    )
    (root / "Cargo.lock").write_text(
        '[[package]]\nname = "aisimulate-core"\nversion = "0.13.0"\n'
    )
    (root / "scripts/apply_dev_version.py").write_text(
        """
import sys
from pathlib import Path
root = Path(sys.argv[2])
tag = sys.argv[1]
for relative in ("python/aisimulate/pyproject.toml", "crates/core/Cargo.toml"):
    path = root / relative
    text = path.read_text()
    suffix = tag if not relative.startswith("crates") else tag.replace(".dev", "-dev.")
    old = '"0.13.0"'
    path.write_text(text.replace(old, '"' + old[1:-1] + suffix + '"', 1))
"""
    )
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.GNU_FORMAT) as tar:
        tar.add(content, arcname=".")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    deps = tmp_path / "deps"
    deps.mkdir()
    (deps / "aisimulate-sources.json").write_text(
        json.dumps({version: {"commit": commit, "sha256": digest}})
    )
    monkeypatch.setattr(
        staging.urllib.request, "urlopen", lambda *args, **kwargs: open(archive, "rb")
    )
    return staging, deps


def test_aisimulate_source_fallback_extracts_and_verifies_real_archive(
    tmp_path, monkeypatch
):
    version = str(_exact_version(_python_requirement(_root_configs()[0])))
    rust_version = version.replace(".dev", "-dev.")
    staging, deps = _aisim_source_archive(tmp_path, version, monkeypatch)
    work = tmp_path / "work"
    work.mkdir()
    checkout = staging.prepare_source(deps, version, work)
    commit = json.loads((deps / "aisimulate-sources.json").read_text())[version][
        "commit"
    ]
    assert checkout == work / f"aisim-{commit}"
    assert (
        tomllib.loads(
            (checkout / "python/aisimulate/pyproject.toml").read_text()
        )["project"]["version"]
        == version
    )
    assert (
        tomllib.loads((checkout / "crates/core/Cargo.toml").read_text())["package"][
            "version"
        ]
        == rust_version
    )
    assert f'version = "{rust_version}"' in (checkout / "Cargo.lock").read_text()
    with pytest.raises(SystemExit, match="No reviewed source fallback"):
        staging.prepare_source(
            deps, "0.13.0.dev209901010000000001", tmp_path / "work2"
        )


def test_aisimulate_source_fallback_rejects_tampered_archive(tmp_path, monkeypatch):
    version = str(_exact_version(_python_requirement(_root_configs()[0])))
    staging, deps = _aisim_source_archive(tmp_path, version, monkeypatch)
    archive = tmp_path / "source.tar.gz"
    data = bytearray(archive.read_bytes())
    data[-6] ^= 0x01
    archive.write_bytes(data)
    recorded = json.loads((deps / "aisimulate-sources.json").read_text())[version][
        "sha256"
    ]
    with archive.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() != recorded
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(SystemExit, match="source archive checksum mismatch"):
        staging.prepare_source(deps, version, work)
    commit = json.loads((deps / "aisimulate-sources.json").read_text())[version][
        "commit"
    ]
    assert not (work / f"aisim-{commit}").exists()

def test_installed_aisimulate_matches_the_declared_release() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISimulate supports Python 3.11 through 3.13")
    pyproject, _ = _root_configs()
    expected = _exact_version(_python_requirement(pyproject))

    assert Version(metadata.version("aisimulate")) == expected



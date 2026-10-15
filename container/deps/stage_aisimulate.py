# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage exact AISimulate wheels or reviewed Rust source for image builds."""

import argparse
import email
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import tomllib


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def prepare_source(deps, version, work):
    """Download only the reviewed commit, verify its digest, and stamp the pin."""
    rust_version = version.replace(".dev", "-dev.")
    sources = json.loads((deps / "aisimulate-sources.json").read_text())
    if version not in sources:
        raise SystemExit(f"No reviewed source fallback for AISimulate {version}")
    source = sources[version]
    commit = source["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit("AISimulate source must be an immutable commit")
    print(f"Building AISimulate {version} from {commit}", flush=True)
    archive = work / "source.tar.gz"
    url = f"https://codeload.github.com/ai-dynamo/aisimulate/tar.gz/{commit}"
    for attempt in range(3):
        try:
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                archive.open("wb") as dest,
            ):
                shutil.copyfileobj(response, dest)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != source["sha256"]:
        raise SystemExit("AISimulate source archive checksum mismatch")
    with tarfile.open(archive) as tar:
        # Agent/editor links are unrelated to the build and some point
        # outside the checkout. Preserve package/license links only.
        members = [
            m
            for m in tar.getmembers()
            if not set(Path(m.name).parts) & {".agents", ".claude"}
        ]
        tar.extractall(work, members=members, filter="data")
    checkout = work / f"aisimulate-{commit}"
    run(
        sys.executable,
        str(checkout / "scripts/apply_dev_version.py"),
        ".dev" + version.split(".dev", 1)[1],
        str(checkout),
    )
    project = checkout / "python/aisimulate"
    py_version = tomllib.loads((project / "pyproject.toml").read_text())[
        "project"
    ]["version"]
    core_version = tomllib.loads(
        (checkout / "crates/core/Cargo.toml").read_text()
    )["package"]["version"]
    if (py_version, core_version) != (version, rust_version):
        raise SystemExit(
            "Stamped source does not match the required Python/Rust versions"
        )
    # Stamping changes only the local workspace package in Cargo.lock.
    # Keep all transitive dependency versions locked during compilation.
    lock = checkout / "Cargo.lock"
    text, count = re.subn(
        r'(name = "aisimulate-core"\nversion = ")[^"]+("\n)',
        lambda match: match[1] + rust_version + match[2],
        lock.read_text(),
    )
    if count != 1:
        raise SystemExit("Expected one AISimulate core package in Cargo.lock")
    lock.write_text(text)
    return checkout


def stage(root, output):
    deps = root / "container/deps"
    requirements = deps / "requirements.aisimulate.txt"
    pins = [
        line.strip()
        for line in requirements.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(pins) != 1 or not re.fullmatch(
        r"aisimulate==[0-9]+\.[0-9]+\.[0-9]+(?:\.dev[0-9]+)?", pins[0]
    ):
        raise SystemExit("Expected one exact AISimulate version pin")
    version = pins[0].split("==", 1)[1]
    rust_version = version.replace(".dev", "-dev.")
    for manifest in (root / "Cargo.toml", root / "lib/bindings/python/Cargo.toml"):
        data = tomllib.loads(manifest.read_text())
        dependencies = (
            data["dependencies"]
            if "dependencies" in data
            else data["workspace"]["dependencies"]
        )
        if dependencies["aisimulate-core"]["version"] != "=" + rust_version:
            raise SystemExit(f"AISimulate Python/Rust version mismatch: {manifest}")

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aisimulate-wheel-") as temp:
        work = Path(temp)
        wheels = work / "wheels"
        wheels.mkdir()
        downloaded = (
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "--only-binary=:all:",
                    "--no-deps",
                    "--no-index",
                    "--retries",
                    "3",
                    "--timeout",
                    "60",
                    "--find-links",
                    "https://pypi.nvidia.com/aisimulate/",
                    "--dest",
                    str(wheels),
                    "--requirement",
                    str(requirements),
                ],
                check=False,
            ).returncode
            == 0
        )
        if not downloaded:
            checkout = prepare_source(deps, version, work)
            project = checkout / "python/aisimulate"
            # Use the existing manylinux builder toolchain; wheels are consumed
            # by this image family, with the same ABI floor as Dynamo wheels.
            run(
                sys.executable,
                "-m",
                "maturin",
                "build",
                "--release",
                "--locked",
                "--out",
                str(wheels),
                cwd=project,
            )

        candidates = list(wheels.glob("aisimulate-*.whl"))
        if len(candidates) != 1:
            raise SystemExit(
                f"Expected exactly one AISimulate wheel, found {candidates}"
            )
        wheel = candidates[0]
        with zipfile.ZipFile(wheel) as package:
            metadata_files = [
                name
                for name in package.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise SystemExit(
                    "Wheel must have exactly one distribution metadata file"
                )
            metadata = email.message_from_bytes(package.read(metadata_files[0]))
            if (metadata["Name"], metadata["Version"]) != ("aisimulate", version):
                raise SystemExit(
                    "Downloaded/built wheel metadata does not match the requested version"
                )
        shutil.copy2(wheel, output / wheel.name)
        print(f"Staged {wheel.name}", flush=True)


def stage_rust_source(root, output):
    """Use reviewed source when the exact nightly is not in the public index.

    This changes only the disposable build checkout: Cargo's manifest pins and
    all transitive lockfile entries stay fixed. --locked metadata/build verifies
    that the source has the same dependency requirements as the published crate.
    """
    data = tomllib.loads((root / "Cargo.toml").read_text())
    pin = data["workspace"]["dependencies"]["aisimulate-core"]["version"]
    if not re.fullmatch(r"=\d+\.\d+\.\d+-dev\.\d+", pin):
        raise SystemExit("Rust source staging requires an exact reviewed nightly pin")
    version = pin[1:].replace("-dev.", ".dev")
    output.mkdir(parents=True, exist_ok=True)
    checkout = prepare_source(root / "container/deps", version, output)
    config = root / ".cargo/config.toml"
    text = config.read_text()
    if "[patch.crates-io]" in text:
        raise SystemExit("Refusing to replace existing Cargo source patches")
    # The runtime and KVBM bindings are independent Cargo workspaces.
    # Validate every lockfile before writing any of the build-only overrides.
    updated_locks = []
    for relative in (
        "Cargo.lock",
        "lib/bindings/python/Cargo.lock",
        "lib/bindings/kvbm/Cargo.lock",
    ):
        lock = root / relative
        lock_text, count = re.subn(
            r'(name = "aisimulate-core"\nversion = "' + re.escape(pin[1:])
            + r'"\n)source = "registry\+https://github.com/rust-lang/crates.io-index"\n'
            + r'checksum = "[0-9a-f]{64}"\n',
            r"\1",
            lock.read_text(),
        )
        if count != 1:
            raise SystemExit(f"Expected one locked registry AISimulate core package: {relative}")
        updated_locks.append((lock, lock_text))
    config.write_text(
        text + "\n[patch.crates-io]\naisimulate-core = { path = "
        + json.dumps(str(checkout / "crates/core")) + " }\n"
    )
    for lock, lock_text in updated_locks:
        lock.write_text(lock_text)
    licenses = root / "rust-licenses" / ("aisimulate-core-" + pin[1:])
    licenses.mkdir(parents=True, exist_ok=True)
    for name in ("LICENSE", "NOTICE"):
        if (checkout / name).is_file():
            shutil.copy2(checkout / name, licenses / name)
    print(f"Staged Rust source: aisimulate-core {pin[1:]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/dynamo"))
    parser.add_argument("--output", type=Path, default=Path("/opt/dynamo/dist"))
    parser.add_argument("--rust-source-only", action="store_true")
    args = parser.parse_args()
    if args.rust_source_only:
        stage_rust_source(args.root.resolve(), args.output.resolve())
    else:
        stage(args.root.resolve(), args.output.resolve())

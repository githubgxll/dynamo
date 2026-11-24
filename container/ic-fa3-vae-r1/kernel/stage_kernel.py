"""Materialize or verify the pinned H100 FA3 kernel in the kernels 0.16 cache layout."""
from __future__ import annotations

import argparse
import base64
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from tempfile import TemporaryDirectory
import zipfile

LOCK_PATH = Path(__file__).with_name("lock.json")
LIBRARY = "_flash_attn3_cuda_0f52d9d.abi3.so"

# The parent image enables Python safe-path mode. Load the shipped helper by
# its exact file path, regardless of the working directory or PYTHONPATH.
spec = util.spec_from_file_location("_ic_fa3_download", Path(__file__).resolve().with_name("download.py"))
if spec is None or spec.loader is None:
    raise ImportError("Cannot load the bundled FA3 downloader")
download = util.module_from_spec(spec)
spec.loader.exec_module(download)


def require(value, message):
    if not value:
        raise ValueError(message)


def destination(lock, cache):
    return cache / lock["cache_repo"] / "snapshots" / lock["revision"] / "build" / lock["variant"]


def cache_directory(path):
    # The nested HF snapshot already contributes 160 characters. Let deep
    # Windows checkouts use the same byte-for-byte local install as Linux CI.
    path = path.resolve()
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        name = str(path)
        return Path("\\\\?\\UNC\\" + name[2:] if name.startswith("\\\\") else "\\\\?\\" + name)
    return path


def extract_archive(archive, directory, lock):
    source = lock["offline_archive"]
    require(download.valid(archive, source["sha256"], source["bytes"]), "Local FA3 ZIP differs from the H100 probe archive")
    with zipfile.ZipFile(archive) as package:
        infos = package.infolist()
        expected = lock["files"]
        require(sorted(i.filename for i in infos) == sorted(expected), "Unexpected kernel ZIP member list")
        for info in infos:
            require(info.file_size == expected[info.filename]["bytes"], "Kernel ZIP member size mismatch")
            with package.open(info) as src, (directory / info.filename).open("xb") as out:
                shutil.copyfileobj(src, out, length=4 * 1024 * 1024)


def verify_tree(path, lock):
    require(path.is_dir() and not path.is_symlink(), f"Missing real kernel directory: {path}")
    files = lock["files"]
    require({p.name for p in path.iterdir()} == set(files), "Unexpected files in kernel variant")
    for name, row in files.items():
        require(download.valid(path / name, row["sha256"], row["bytes"]), "Incorrect FA3 file size or SHA256: " + name)
    upstream = json.loads((path / "metadata.json").read_bytes())
    expected = {name: base64.b64encode(bytes.fromhex(row["sha256"])).decode("ascii")
                for name, row in files.items() if name not in {"metadata.json", "metadata.json.sigstore"}}
    require(upstream["digest"] == {"algorithm": "sha256", "files": expected},
            "Pinned binary differs from the published digest list")
    with (path / LIBRARY).open("rb") as lib:
        elf = lib.read(20)
    require(elf[:6] == b"\x7fELF\x02\x01" and int.from_bytes(elf[18:20], "little") == 62,
            "Expected a little-endian 64-bit x86 ELF shared library")


def install(cache, lock, archive):
    target = destination(lock, cache)
    if target.exists():
        verify_tree(target, lock)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fa3-", dir=target.parent) as temporary:
        staged = Path(temporary) / lock["variant"]
        staged.mkdir(mode=0o755)
        if archive is None:
            for name, row in lock["files"].items():
                download.acquire(lock["source_prefix"] + name, staged / name, row["sha256"], row["bytes"])
        else:
            extract_archive(archive, staged, lock)
        verify_tree(staged, lock)
        # The final runtime's non-root user must be able to read the kernel even
        # if the runner's build user had a restrictive umask while fetching.
        for f in staged.iterdir():
            f.chmod(0o644)
        for parent in (staged, *target.parents):
            if parent == cache.parent:
                break
            parent.chmod(0o755)
        try:
            staged.rename(target)
        except OSError:
            if not target.is_dir():
                raise
            verify_tree(target, lock)
    return target


def verify_runtime(cache, lock):
    require(sys.platform.startswith("linux") and platform.machine() == lock["linux_platform"],
            "FA3 runtime requires Linux x86_64")
    require(os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("KERNELS_CACHE") == str(cache),
            "Set HF_HUB_OFFLINE=1 and KERNELS_CACHE to the immutable snapshot cache")
    import torch

    observed = {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "kernels_version": metadata.version("kernels")}
    require(all(v == lock[k] for k, v in observed.items()), f"Parent Torch/CUDA/kernels differ: {observed}")
    return observed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "verify"))
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--archive", type=Path, help="Use the original verified offline archive instead of HTTPS")
    parser.add_argument("--runtime", action="store_true", help="Check the installed Linux runtime without GPU calls")
    args = parser.parse_args(argv)
    if args.action == "verify" and args.archive:
        parser.error("--archive applies to install only")
    raw = LOCK_PATH.read_bytes()
    lock = json.loads(raw)
    cache = cache_directory(args.cache)
    target = destination(lock, cache)
    if args.action == "install":
        install(cache, lock, args.archive)
    else:
        verify_tree(target, lock)
    report = {"status": "PASS_PINNED_FA3_FILES", "repo": lock["repo"], "revision": lock["revision"],
              "variant": lock["variant"], "build": str(target), "files": len(lock["files"]),
              "binary_sha256": lock["files"][LIBRARY]["sha256"], "lock_sha256": hashlib.sha256(raw).hexdigest(),
              "gpu_executed": False}
    if args.runtime:
        report.update(verify_runtime(cache, lock), status="PASS_PINNED_FA3_FILES_AND_RUNTIME_METADATA")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        sys.exit(f"FA3 kernel validation failed: {exc}")

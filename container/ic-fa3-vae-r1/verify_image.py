"""Install the frozen source archive or check the assembled image without a GPU."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
from zipfile import ZipFile


ROOT = Path("/opt/ic-r1")
PARENT = "registry.hd-04.alayanew.com:8443/openclaw/vllm-omni@sha256:9387d41fca398d0061cc36be37a2ef4527e56cbbb679727e7494b93add9f9658"
CACHE = "/opt/ic-fa3-cache"
ID = "ic-fa3-vae-r1"
PAYLOAD_DIRS = ("source", "inputs", "worker")
MODEL_SUFFIXES = {".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".onnx"}
METADATA = {"vllm": "0.33.0", "torch": "2.14.0+cu130", "kernels": "0.16.1"}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_digest(rows):
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def manifest_rows(manifest):
    require(manifest["schema"] == "ic-image-payload/1" and manifest["parent_image"] == PARENT
            and manifest["release_id"] == ID, "Unexpected release manifest identity")
    rows = {}
    for row in manifest["files"]:
        name = row["path"]
        path = PurePosixPath(name)
        require(len(path.parts) > 1 and path.parts[0] in PAYLOAD_DIRS and ".." not in path.parts
                and name == path.as_posix() and "\\" not in name and ":" not in name
                and path.suffix.lower() not in MODEL_SUFFIXES, f"Invalid image payload path: {name}")
        require(name not in rows and set(row) == {"path", "bytes", "sha256"}
                and type(row["bytes"]) is int and row["bytes"] >= 0
                and re.fullmatch("[0-9a-f]{64}", row["sha256"]), f"Invalid manifest entry: {name}")
        rows[name] = row
    worker = [rows[name] for name in sorted(rows) if name.startswith("worker/")]
    require(tree_digest(worker) == manifest["worker_tree_sha256"], "Worker tree differs from release")
    require("worker/run_suite.py" in rows and "source/vllm_omni/__init__.py" in rows,
            "Release lacks the worker or Omni package")
    require(rows["inputs/DEV-SOURCE-LOCK.json"]["sha256"] == manifest["source_manifest_sha256"],
            "Release source lock fields disagree")
    return rows


def unpack(archive_path, root, rows):
    """Only extract declared plain files under the two new image directories."""
    expected = {name for name in rows if not name.startswith("worker/")}
    require(all(not (root / name).exists() and not (root / name).is_symlink()
                for name in ("source", "inputs")), "Source or input destination already exists")
    with ZipFile(archive_path) as archive:
        entries = archive.infolist()
        require(len(entries) == len(expected) and {i.filename for i in entries} == expected,
                "ZIP contents differ from manifest")
        for item in entries:
            row = rows[item.filename]
            require(item.file_size == row["bytes"] and item.orig_filename == item.filename
                    and not item.is_dir() and (item.external_attr & 0x10) == 0
                    and stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG),
                    f"Wrong ZIP entry size or type: {item.filename}")
            destination = root / item.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            require(digest(destination) == row["sha256"], f"Wrong extracted content: {item.filename}")


def check_files(root, rows):
    actual = set()
    for folder in PAYLOAD_DIRS:
        for dirpath, dirs, files in os.walk(root / folder):
            base = Path(dirpath)
            require(not (base.is_symlink() or any((base / d).is_symlink() for d in dirs)),
                    f"Unexpected linked directory under {base}")
            for name in files:
                path = base / name
                relative = path.relative_to(root).as_posix()
                row = rows.get(relative)
                info = path.lstat()
                require(row is not None and stat.S_ISREG(info.st_mode) and info.st_size == row["bytes"]
                        and digest(path) == row["sha256"], f"Undeclared or invalid image file: {relative}")
                actual.add(relative)
    require(actual == set(rows), "Missing frozen payload files")
    lock = json.loads((root / "inputs" / "DEV-SOURCE-LOCK.json").read_bytes())
    sources = {"source/" + row["path"]: dict(row, path="source/" + row["path"]) for row in lock["files"]}
    require(len(sources) == len(lock["files"]) and sources ==
            {name: row for name, row in rows.items() if name.startswith("source/")},
            "Source lock and release payload differ")


def verify_runtime(root, manifest):
    require(platform.system() == "Linux" and platform.machine() == "x86_64"
            and sys.version_info[:2] == (3, 12), "Expected Linux/amd64 and Python 3.12")
    require(os.environ.get("KERNELS_CACHE") == CACHE and os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Expected offline image environment")
    versions = {name: importlib.metadata.version(name) for name in METADATA}
    require(versions == METADATA, f"Base package versions differ: {versions}")
    import torch
    require(torch.__version__ == METADATA["torch"] and not torch.cuda.is_initialized(), "Expected parent Torch on CPU")
    spec = importlib.util.find_spec("vllm_omni")
    require(spec and spec.origin == str(root / "source/vllm_omni/__init__.py"),
            f"Omni lookup escaped frozen source: {getattr(spec, 'origin', None)}")

    # Importing the GPU pipeline also runs upstream platform detection. Buildx
    # verifies import resolution and syntax; the real eight-rank import is a
    # deployment check on the target's torch/CUDA environment.
    for folder in ("source/vllm_omni", "inputs", "worker"):
        for path in (root / folder).rglob("*.py"):
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    kernel_dir = root / "kernel"
    require(digest(kernel_dir / "lock.json") == manifest["kernel_lock_sha256"], "FA3 lock differs")
    kernel = subprocess.run([sys.executable, "-B", str(kernel_dir / "stage_kernel.py"), "verify",
                             "--cache", CACHE, "--runtime"], stdout=subprocess.PIPE, text=True, check=True)
    kernel_receipt = json.loads(kernel.stdout.strip().splitlines()[-1])
    require(kernel_receipt.get("status") == "PASS_PINNED_FA3_FILES_AND_RUNTIME_METADATA"
            and kernel_receipt.get("lock_sha256") == manifest["kernel_lock_sha256"],
            "Offline kernel check did not validate the release lock")
    help_result = subprocess.run([sys.executable, "-B", str(root / "worker/run_suite.py"), "--help"],
                                 stdout=subprocess.PIPE, text=True, check=True)
    require(all(option in help_result.stdout for option in ("--plan", "--out")), "Worker release CLI differs")
    return {"status": "PASS_IMAGE_CPU_AND_OFFLINE_PAYLOAD_ONLY", "release_id": ID, "parent_image": PARENT,
            "release_manifest_sha256": digest(root / "release-manifest.json"),
            "source_manifest_sha256": manifest["source_manifest_sha256"], "worker_tree_sha256": manifest["worker_tree_sha256"],
            "kernel_lock_sha256": manifest["kernel_lock_sha256"], "kernel": kernel_receipt,
            "base_versions": versions, "omni_source": spec.origin, "gpu_validated": False, "inference_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unpack", type=Path, help="build-stage source archive")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "release-manifest.json").read_bytes())
    rows = manifest_rows(manifest)
    if args.unpack:
        require(digest(args.unpack) == manifest["source_archive_sha256"], "Source ZIP SHA256 differs from release")
        unpack(args.unpack, ROOT, rows)
    check_files(ROOT, rows)
    if not args.unpack:
        print(json.dumps(verify_runtime(ROOT, manifest), sort_keys=True))


if __name__ == "__main__":
    main()

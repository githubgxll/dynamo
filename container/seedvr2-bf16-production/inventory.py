"""Clean-process installed distribution metadata; never import ML packages."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import os
from pathlib import Path
import sys


def snapshot():
    packages, files = [], {}
    for dist in md.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        metadata = Path(str(getattr(dist, "_path", "")))
        if not metadata.is_absolute() or not metadata.exists() or metadata.is_symlink():
            raise ValueError("Distribution metadata is not an absolute regular path: " + str(metadata))
        packages.append({"name": name.lower().replace("_", "-"), "version": dist.version,
                         "metadata_path": str(metadata), "location": str(dist.locate_file(""))})
        paths = [metadata] if metadata.is_file() else sorted(metadata.rglob("*"))
        for path in paths:
            if path.is_symlink():
                raise ValueError("Symlink in installed distribution metadata: " + str(path))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                previous = files.setdefault(str(path), digest)
                if previous != digest:
                    raise ValueError("Metadata changed while scanning: " + str(path))
    return {"schema": "seedvr2-bf16-clean-metadata/1", "status": "PASS",
            "python": sys.executable, "euid": os.geteuid() if hasattr(os, "geteuid") else None,
            "cwd": os.getcwd(), "sys_path": list(sys.path),
            "packages": sorted(packages, key=lambda item: (item["name"], item["version"], item["metadata_path"])),
            "files": dict(sorted(files.items()))}


def require_preserved(before, after, *, record_path, record_sha256):
    """Allow exactly the reviewed Omni RECORD change; retain duplicate packages."""
    for report in (before, after):
        if report.get("schema") != "seedvr2-bf16-clean-metadata/1" or report.get("status") != "PASS":
            raise ValueError("Invalid clean metadata evidence")
    if before["packages"] != after["packages"]:
        raise ValueError("Installed package/version/location inventory changed")
    if record_path not in before["files"]:
        raise ValueError("Omni RECORD was absent from clean metadata evidence")
    expected = {**before["files"], record_path: record_sha256}
    if after["files"] != expected:
        raise ValueError("Metadata changed outside the exact Omni RECORD delta")
    return True

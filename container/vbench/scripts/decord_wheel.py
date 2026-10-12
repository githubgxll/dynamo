#!/usr/bin/env python3
"""Repair the reviewed upstream Decord wheel's inconsistent Python ABI tag.

The official file is named py3-none-manylinux2010_x86_64 but its WHEEL record
says cp36-cp36m. Its Python code uses ctypes and the bundled native libraries
are Linux x86_64 shared libraries, not CPython extensions. Keep every payload
byte; only correct distribution identity, WHEEL tag, and RECORD bookkeeping.
Actual import and video decoding remain separate Linux qualification gates.
"""
from __future__ import annotations

import base64
import csv
import difflib
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ORIGINAL_FILENAME = "decord-0.6.0-py3-none-manylinux2010_x86_64.whl"
ORIGINAL_SHA256 = "51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976"
PATCHED_FILENAME = "decord-0.6.0+vbench1-py3-none-manylinux2010_x86_64.whl"
OLD_INFO = "decord-0.6.0.dist-info"
NEW_INFO = "decord-0.6.0+vbench1.dist-info"
ORIGINAL_TAG = "cp36-cp36m-manylinux2010_x86_64"
CORRECTED_TAG = "py3-none-manylinux2010_x86_64"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _original_record_audit(files: dict[str, bytes]) -> dict:
    # Upstream's exact reviewed archive also has a stale top_level.txt RECORD
    # entry. Preserve that file's bytes and expose the mismatch in the evidence.
    mismatches = []
    rows = list(csv.reader(io.StringIO(files[f"{OLD_INFO}/RECORD"].decode())))
    for name, digest, size in rows:
        if digest and (name not in files or digest != _record_hash(files[name]) or str(len(files[name])) != size):
            mismatches.append(name)
    return {"rows": len(rows), "mismatched_entries": mismatches,
            "note": "Upstream input is pinned by whole-wheel SHA256; output RECORD is regenerated from unchanged payload bytes."}


def patch_decord(original: Path, output_dir: Path, evidence: Path) -> Path:
    """Produce a deterministic, identifiable local wheel with no code changes."""
    if sha256(original) != ORIGINAL_SHA256:
        raise ValueError("decord upstream wheel SHA256 does not match reviewed 0.6.0")
    with zipfile.ZipFile(original) as source:
        names = source.namelist()
        if len(names) != len(set(names)) or any(
            n.startswith("/") or "\\" in n or ".." in PurePosixPath(n).parts or ":" in n
            for n in names
        ) or any(stat.S_ISLNK(item.external_attr >> 16) for item in source.infolist()):
            raise ValueError("unsafe or duplicate decord wheel members")
        files = {n: source.read(n) for n in names if not n.endswith("/")}
    before_metadata = files[f"{OLD_INFO}/METADATA"].decode("utf-8")
    before_wheel = files[f"{OLD_INFO}/WHEEL"].decode("utf-8")
    if before_metadata.count("Name: decord\n") != 1 or before_metadata.count("Version: 0.6.0\n") != 1:
        raise ValueError("decord METADATA differs from reviewed input")
    tags = [line for line in before_wheel.splitlines() if line.startswith("Tag:")]
    if tags != [f"Tag: {ORIGINAL_TAG}"] or before_wheel.count(f"Tag: {ORIGINAL_TAG}\n") != 1:
        raise ValueError("decord WHEEL tag differs from reviewed input")
    original_record_audit = _original_record_audit(files)
    original_file_hashes = {n: hashlib.sha256(data).hexdigest() for n, data in sorted(files.items())}
    after_metadata = before_metadata.replace("Version: 0.6.0\n", "Version: 0.6.0+vbench1\n")
    after_wheel = before_wheel.replace(f"Tag: {ORIGINAL_TAG}\n", f"Tag: {CORRECTED_TAG}\n")
    files[f"{OLD_INFO}/METADATA"] = after_metadata.encode("utf-8")
    files[f"{OLD_INFO}/WHEEL"] = after_wheel.encode("utf-8")
    del files[f"{OLD_INFO}/RECORD"]
    files = {NEW_INFO + n[len(OLD_INFO):] if n.startswith(OLD_INFO + "/") else n: data
             for n, data in files.items()}
    changed = {f"{OLD_INFO}/{name}" for name in ("METADATA", "WHEEL", "RECORD")}
    unchanged_file_hashes = {n: digest for n, digest in original_file_hashes.items() if n not in changed}
    for old_name, digest in unchanged_file_hashes.items():
        new_name = NEW_INFO + old_name[len(OLD_INFO):] if old_name.startswith(OLD_INFO + "/") else old_name
        if hashlib.sha256(files[new_name]).hexdigest() != digest:
            raise ValueError(f"decord payload changed unexpectedly: {old_name}")
    rows = [[n, _record_hash(data), str(len(data))] for n, data in sorted(files.items())]
    rows.append([NEW_INFO + "/RECORD", "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[NEW_INFO + "/RECORD"] = record.getvalue().encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / PATCHED_FILENAME
    with tempfile.TemporaryDirectory(prefix="decord-repack-", dir=output_dir) as temporary:
        pending = Path(temporary) / PATCHED_FILENAME
        # A stored wheel avoids zlib/zlib-ng producing different compressed bytes
        # on the Windows audit host and Linux builder. Docker compresses the layer.
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_STORED) as wheel:
            for name, data in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                # Make ZIP metadata identical on the Windows audit host and Linux Runner.
                info.create_system = 3
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o644 << 16
                wheel.writestr(info, data)
        os.replace(pending, output)
    evidence.mkdir(parents=True, exist_ok=True)
    for label, before, after in (("METADATA", before_metadata, after_metadata), ("WHEEL", before_wheel, after_wheel)):
        diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                         fromfile=f"{OLD_INFO}/{label}", tofile=f"{NEW_INFO}/{label}"))
        (evidence / f"decord-{label}.patch").write_text(diff, encoding="utf-8")
    (evidence / "decord-patched-RECORD.csv").write_text(record.getvalue(), encoding="utf-8")
    (evidence / "decord-patch.json").write_text(json.dumps({
        "original_filename": ORIGINAL_FILENAME, "original_sha256": ORIGINAL_SHA256,
        "patched_filename": PATCHED_FILENAME, "patched_sha256": sha256(output),
        "version": "0.6.0+vbench1", "original_tag": ORIGINAL_TAG, "corrected_tag": CORRECTED_TAG,
        "code_changed": False, "native_binaries_changed": False, "licenses_preserved": True,
        "change": "Correct WHEEL tag to match official py3-none filename; local version; dist-info rename; RECORD regeneration",
        "original_record_audit": original_record_audit,
        "unchanged_file_sha256": unchanged_file_hashes,
        "payload_sha256": {n: digest for n, digest in unchanged_file_hashes.items() if not n.startswith(OLD_INFO + "/")},
    }, indent=2) + "\n", encoding="utf-8")
    return output


def acquire_original(wheels: Path, sources: dict) -> Path:
    """Download only the reviewed artifact; this is not a dependency install."""
    wheels.mkdir(parents=True, exist_ok=True)
    original = wheels / ORIGINAL_FILENAME
    if original.exists():
        if sha256(original) != ORIGINAL_SHA256:
            raise ValueError("cached decord original has wrong SHA256")
        return original
    with tempfile.TemporaryDirectory(prefix="decord-download-", dir=wheels) as temporary:
        root = Path(temporary)
        requirement = root / "original-wheel.txt"
        requirement.write_text(f"decord==0.6.0 --hash=sha256:{ORIGINAL_SHA256}\n", encoding="utf-8")
        destination = root / "download"
        destination.mkdir()
        subprocess.run([sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check", "download",
                        "--index-url", sources["python_index"], "--only-binary=:all:", "--no-deps", "--require-hashes",
                        "--retries", "1", "--timeout", "15", "--dest", str(destination), "-r", str(requirement)], check=True)
        downloaded = destination / ORIGINAL_FILENAME
        if {item.name for item in destination.iterdir()} != {ORIGINAL_FILENAME} or not downloaded.is_file():
            raise ValueError("decord download did not produce exactly the reviewed Linux x86_64 wheel")
        if sha256(downloaded) != ORIGINAL_SHA256:
            raise ValueError("downloaded decord wheel has wrong SHA256")
        os.replace(downloaded, original)
    return original


def prepare_decord(wheels: Path, evidence: Path, sources: dict) -> Path:
    return patch_decord(acquire_original(wheels, sources), wheels, evidence)

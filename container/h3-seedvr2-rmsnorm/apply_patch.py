# SPDX-License-Identifier: Apache-2.0
"""Apply only the reviewed upstream RMSNorm change to an installed wheel."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path

from patchlib import (atomic_write, check_sources, dump, inventory, lock_data,
                      locate_installed, patch_bytes, sha256, source_hashes, update_record)


def apply_installed(report: Path) -> dict:
    lock = lock_data()
    dist, module, record = locate_installed()
    before_sources = source_hashes(module)
    check_sources(before_sources, lock, patched=False)
    for name, expected in lock["required_versions"].items():
        if metadata.version(name) != expected:
            raise ValueError(f"Unexpected parent version: {name}")
    before_inventory = inventory()
    original = module.read_bytes()
    patched = patch_bytes(original, lock)
    original_record = record.read_bytes()
    changed_record = update_record(original_record, original, patched)
    original_direct_url = dist.read_text("direct_url.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    # Preserve exact original distribution metadata; parent wheel provenance stays intact.
    atomic_write(report.parent / "omni-RECORD.before.csv", original_record)
    dump(report.parent / "installed-versions.before.json", before_inventory)
    # A failure aborts the build; these writes are not a live hot-patching interface.
    atomic_write(module, patched)
    atomic_write(record, changed_record)
    # Avoid timestamp/size cached bytecode, including already compiled optimized variants.
    pycache = module.parent / "__pycache__"
    if pycache.is_dir():
        for cached in pycache.glob("nadit.*.pyc"):
            cached.unlink()
    after_sources = source_hashes(module)
    check_sources(after_sources, lock, patched=True)
    if inventory() != before_inventory or dist.read_text("direct_url.json") != original_direct_url:
        raise ValueError("Package inventory or original wheel provenance changed while patching")
    result = {
        "schema": "h3-seedvr2-rmsnorm-application/1", "status": "PASS",
        "upstream_commit": lock["upstream_commit"], "parent_image": lock["parent_image"],
        "module_path": str(module), "before_sha256": sha256(original), "after_sha256": sha256(patched),
        "record_path": str(record), "record_before_sha256": sha256(original_record),
        "record_after_sha256": sha256(changed_record),
        "direct_url_sha256": sha256(original_direct_url.encode("utf-8")) if original_direct_url is not None else None,
        "source_hashes_before": before_sources, "source_hashes_after": after_sources,
        "inventory_unchanged": True, "scope": "Exact installed source backport; numerical and GPU validation are separate",
    }
    dump(report, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-installed", action="store_true", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply_installed(args.report), indent=2))


if __name__ == "__main__":
    main()

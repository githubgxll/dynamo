# SPDX-License-Identifier: Apache-2.0
"""Verify the actual installed patched module without changing dependencies."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from patchlib import (HERE, MODULE_NAME, check_sources, dump, inventory, lock_data,
                      locate_installed, sha256, source_hashes)


def check_pytest_xml(path: Path) -> int:
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    if len(cases) != 6 or any(case.find(kind) is not None for case in cases for kind in ("failure", "error", "skipped")):
        raise ValueError("Pristine upstream regression must report exactly six successful tests, none skipped")
    if any("test_rms_norm_large_finite_inputs" not in case.attrib.get("name", "") for case in cases):
        raise ValueError("Unexpected upstream numerical test identity")
    return len(cases)


def run_upstream_tests(output: Path, lock: dict) -> None:
    test = HERE / "upstream" / "test_seedvr2_norm.py"
    if sha256(test.read_bytes()) != lock["vendored_sha256"]["upstream/test_seedvr2_norm.py"]:
        raise ValueError("Upstream numerical regression bytes changed")
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(name, None)
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="srnorm-pytest-") as cwd:
        command = [sys.executable, "-I", "-B", "-m", "pytest", "-q", "-c", os.devnull,
                   "--rootdir", str(test.parent), "--confcutdir", str(test.parent),
                   "--import-mode=importlib", "-p", "no:cacheprovider", "-o", "addopts=",
                   "-o", "markers=core_model: upstream model check\ncpu: upstream CPU check",
                   "--junitxml", str(output / "upstream-pytest.xml"), str(test)]
        result = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False, timeout=180)
    (output / "upstream-pytest.txt").write_text(result.stdout, encoding="utf-8")
    print(result.stdout, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"Pristine upstream RMSNorm regression failed (exit {result.returncode})")
    check_pytest_xml(output / "upstream-pytest.xml")


def numeric_cases(norm_class, torch, device: str) -> tuple[list[dict], dict]:
    cases = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for affine in (False, True):
            x = torch.ones(2, 3, 2560, dtype=dtype, device=device)
            x[0, 0, 0] = 256
            x[0, 1, 0] = 300
            x[0, 2, 0] = 60000
            x[1, 0].fill_(300)
            x[1, 1].zero_()
            x[1, 2].fill_(0.0001)
            norm = norm_class(2560, eps=1e-5, elementwise_affine=affine).to(device=device, dtype=dtype)
            with torch.no_grad():
                if affine:
                    norm.weight.copy_(torch.linspace(0.5, 1.5, 2560, dtype=dtype, device=device))
                high = x.double()
                reference = (high / (high.square().mean(-1, keepdim=True) + norm.eps).sqrt()).to(dtype)
                if affine:
                    reference = reference * norm.weight
                actual = norm(x)
            if actual.dtype != dtype or not actual.isfinite().all().item():
                raise AssertionError("RMSNorm changed dtype or produced nonfinite values")
            torch.testing.assert_close(actual, reference)
            cases.append({"dtype": str(dtype), "affine": affine, "device": device,
                          "finite": True, "allclose": True,
                          "max_absolute_error": (actual.double() - reference.double()).abs().max().item()})
    x = torch.full((2, 2560), 300.0, dtype=torch.float16, device=device)
    with torch.no_grad():
        actual = norm_class(2560, elementwise_affine=False).to(device=device, dtype=torch.float16)(x)
        old_formula = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(actual, torch.ones_like(actual))
    if not torch.equal(old_formula, torch.zeros_like(old_formula)):
        raise AssertionError("The FP16 all-300 regression did not reproduce the old zero-output failure")
    return cases, {"passed": True, "input": 300.0, "output_min": actual.min().item(),
                   "output_max": actual.max().item(), "old_formula_all_zero": True}


def validate(output: Path, device: str, application_report: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = lock_data()
    receipt = json.loads(application_report.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or receipt.get("upstream_commit") != lock["upstream_commit"]:
        raise ValueError("Missing or unrecognized installed patch receipt")
    dist, expected_path, record = locate_installed()
    if receipt["before_sha256"] != lock["before_sha256"] or receipt["after_sha256"] != lock["after_sha256"]:
        raise ValueError("Patch receipt has unexpected source identity")
    if Path(receipt["module_path"]).resolve() != expected_path:
        raise ValueError("Patch receipt and installed module path differ")
    if sha256(record.read_bytes()) != receipt["record_after_sha256"]:
        raise ValueError("Patched installed distribution RECORD changed")
    before_record = application_report.parent / "omni-RECORD.before.csv"
    if sha256(before_record.read_bytes()) != receipt["record_before_sha256"]:
        raise ValueError("Original distribution RECORD evidence changed")
    direct_url = dist.read_text("direct_url.json")
    if (sha256(direct_url.encode("utf-8")) if direct_url is not None else None) != receipt["direct_url_sha256"]:
        raise ValueError("Original wheel origin evidence changed")
    before_inventory = json.loads((application_report.parent / "installed-versions.before.json").read_text(encoding="utf-8"))
    current_inventory = inventory()
    if current_inventory != before_inventory:
        raise ValueError("Installed distributions differ from the inherited parent inventory")
    dump(output / "installed-versions.json", current_inventory)
    versions = {name: metadata.version(name) for name in lock["required_versions"]}
    if versions != lock["required_versions"]:
        raise ValueError("Protected parent component versions changed")
    sources = source_hashes(expected_path)
    check_sources(sources, lock, patched=True)
    # This import is intentionally the real installed module, including its imports.
    # AST extraction can validate local syntax, but is not accepted by this gate.
    module = importlib.import_module(MODULE_NAME)
    imported_path = Path(module.__file__).resolve()
    function_path = Path(inspect.getsourcefile(module.RMSNorm.forward)).resolve()
    if imported_path != expected_path or function_path != expected_path:
        raise ValueError("Runtime import is shadowed by a different SeedVR2 source tree")
    if sha256(imported_path.read_bytes()) != lock["after_sha256"]:
        raise ValueError("Actual imported RMSNorm source is not the approved patch")
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA acceptance was explicitly requested but no CUDA device is available")
    cases, all_300 = numeric_cases(module.RMSNorm, torch, device)
    # The exact upstream file runs unmodified on CPU. The independent six-case
    # probe above also exercises CUDA when selected explicitly after publication.
    run_upstream_tests(output, lock)
    result = {
        "schema": "h3-seedvr2-rmsnorm-runtime/1", "status": "PASS", "device": device,
        "module_path": str(imported_path), "module_sha256": sha256(imported_path.read_bytes()),
        "upstream_commit": lock["upstream_commit"], "parent_image": lock["parent_image"],
        "versions": versions, "six_case_count": len(cases), "six_cases": cases, "all_300": all_300,
        "upstream_pytest_case_count": 6, "source_hashes": sources, "inventory_unchanged": True,
        "distribution_inventory_sha256": sha256((output / "installed-versions.json").read_bytes()),
        "euid": os.geteuid() if hasattr(os, "geteuid") else None,
        "source_commit": os.environ.get("H3_EXPERIMENTAL_SOURCE_COMMIT"),
        "scope": "Installed import, six numerical groups, original upstream regression and unchanged package inventory; no model/video quality claim",
    }
    dump(output / "runtime-validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--application-report", type=Path,
                        default=HERE / "build-info" / "patch-application.json")
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.output_dir, args.device, args.application_report), indent=2))
    except Exception as error:
        dump(args.output_dir / "runtime-validation.json", {"status": "FAIL", "device": args.device,
                                                          "error": str(error), "error_type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()

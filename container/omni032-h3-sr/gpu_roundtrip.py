#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bind the reviewed HSDP8 numerical gate to the new 0.32 source verifier.

Resolving this adapter does not execute CUDA. Running it under torchrun invokes
the unchanged gate, which requires eight real CUDA ranks and never CPU-skips.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "h3-hsdp-offload"
GATE_SHA256 = "e611fff462b6fbaa80a67872973e0032754be9a47845a308380a38594f9d5a61"
TORCH_CONTRACT_SHA256 = "57f96f29e51f5145aa059c8d29b3209c9fcde793f64443d14eda7c1f16ca4952"
TORCH_POLICY_SHA256 = "cef22d9996df716be30524a37bde4fce1e2a5a767da243ab6eb79f958e419c61"


def _checked(path, expected):
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Reviewed inherited GPU implementation differs: " + str(path))


def _bind(name, path):
    path = path.resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != path:
            raise ValueError("A different " + name + " verifier is already bound")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def resolve_gate():
    path = PARENT / "gpu_roundtrip.py"
    torch_path = PARENT / "torch_contract.py"
    _checked(path, GATE_SHA256)
    _checked(torch_path, TORCH_CONTRACT_SHA256)
    _checked(PARENT / "torch-contract.json", TORCH_POLICY_SHA256)
    # The old gate calls runtime.verify_installed() only for source provenance.
    # It must see the new full-wheel identity, never the 0.31 lineage verifier.
    _bind("runtime", HERE / "runtime.py")
    _bind("torch_contract", torch_path)
    return _bind("_omni032_inherited_gpu_gate", path)


if __name__ == "__main__":
    resolve_gate().main()

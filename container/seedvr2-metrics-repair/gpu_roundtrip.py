#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the unchanged eight-rank parent gate with exact child source identity.

Invoke through torchrun just like the inherited gate. This adapter adds no GPU
success path and changes none of its numerical, cycle or fastpath assertions.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = Path("/opt/h3-hsdp-offload")


def resolve_gate():
    lock = json.loads((HERE / "inputs/source-lock.json").read_text(encoding="utf-8"))
    path = PARENT / "gpu_roundtrip.py"
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != lock["parent_gpu_gate_sha256"]:
        raise ValueError("Inherited GPU gate differs from the exact reviewed parent")
    # The inherited gate deliberately imports `runtime` at provenance time.
    # Bind it explicitly to the child verifier, which checks the full parent
    # manifest plus the exact declared SR delta and the unchanged H3 sources.
    torch_path = PARENT / "torch_contract.py"
    if torch_path.is_symlink() or hashlib.sha256(torch_path.read_bytes()).hexdigest() != lock["parent_torch_contract_sha256"]:
        raise ValueError("Inherited Torch contract differs from the exact reviewed parent")
    for name, target in (("source_patches", HERE / "source_patches.py"),
                         ("runtime", HERE / "runtime.py"), ("torch_contract", torch_path)):
        existing = sys.modules.get(name)
        if existing is not None and Path(getattr(existing, "__file__", "")).resolve() != target:
            raise ValueError("A different " + name + " verifier is already bound")
        if existing is None:
            spec = importlib.util.spec_from_file_location(name, target)
            existing = importlib.util.module_from_spec(spec)
            sys.modules[name] = existing
            spec.loader.exec_module(existing)
    spec = importlib.util.spec_from_file_location("_sr_opt_inherited_gpu_gate", path)
    gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gate
    spec.loader.exec_module(gate)
    return gate


if __name__ == "__main__":
    resolve_gate().main()

# SPDX-License-Identifier: Apache-2.0
"""One exact source delta; keep H3 selection and component lifecycle intact."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = "vllm_omni/diffusion/offloader/sequential_backend.py"
HELPER = "vllm_omni/diffusion/offloader/hsdp_migration.py"


def transform(data):
    source = data.decode("utf-8")
    anchor = "from .plan_resolver import resolve_offload_plan\n"
    if source.count(anchor) != 1:
        raise ValueError("Unexpected offloader import anchor")
    source = source.replace(anchor, anchor + "from .hsdp_migration import capture_components, restore_components, move_component\n", 1)
    start = source.index("def _capture_tensor_devices(")
    end = source.index("\n\nclass SequentialOffloadHook", start)
    source = source[:start] + '''def _capture_tensor_devices(modules):
    # Capture owning components, not naked .data handles: FSDP owns shard refs.
    return capture_components(modules)


def _restore_tensor_devices(devices):
    restore_components(devices)
''' + source[end:]
    start = source.index('        """Move module parameters and buffers to device.')
    end = source.index("\n    def _to_cpu(", start)
    source = source[:start] + '''        """Move one idle component through native PyTorch/FSDP conversion."""
        return move_component(
            module, target_device, non_blocking=non_blocking, pin_memory=pin_memory
        )
''' + source[end:]
    # Normal execution still enters this exact hook at the original stages.
    # No H3 pipeline, HSDP wrapping, cache selection, or denoising loop is edited.
    compile(source, TARGET, "exec")
    return source.encode("utf-8")


def locked_changes(root):
    lock = json.loads((HERE / "source-patches.lock.json").read_text(encoding="utf-8"))
    before = (root / TARGET).read_bytes()
    after = transform(before)
    if hashlib.sha256(before).hexdigest() != lock["before_sha256"]:
        raise ValueError("Installed offloader differs from successful parent")
    if hashlib.sha256(after).hexdigest() != lock["after_sha256"]:
        raise ValueError("Offloader patch differs from reviewed output")
    helper = (HERE / "hsdp_migration.py").read_bytes()
    if hashlib.sha256(helper).hexdigest() != lock["helper_sha256"]:
        raise ValueError("Migration helper differs from reviewed lock")
    return TARGET, before, after, helper

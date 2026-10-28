# SPDX-License-Identifier: Apache-2.0
"""Exact migration and H3 entry deltas over the unchanged successful C14 base."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = "vllm_omni/diffusion/offloader/sequential_backend.py"
HELPER = "vllm_omni/diffusion/offloader/hsdp_migration.py"
PIPELINE = "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
PIPELINE_BEFORE = "e8d37d9e086b72930d33b419efadbf25ff695bcef295cba8f3c0c2b59edc85e0"


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
    # This transform preserves the prior migration patch byte for byte.
    # The pipeline entry fix below moves activation before FSDP pre-forward.
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


def transform_pipeline(before):
    if hashlib.sha256(before).hexdigest() != PIPELINE_BEFORE:
        raise ValueError("Source SHA256 differs from the locked original H3 pipeline")
    old = b"        with self._resident_dit_layers_on_device(enabled=transformer is self.transformer):\n"
    new = b'''        # Restore the whole component before native FSDP forward-pre hooks run.
        component_context = (
            self._component_on_device(transformer)
            if getattr(self, "_model_cpu_offload_modules", None)
            else nullcontext()
        )
        with (
            component_context,
            self._resident_dit_layers_on_device(enabled=transformer is self.transformer),
        ):
'''
    if before.count(old) != 1:
        raise ValueError("Expected exactly one diffuse residency-context anchor")
    after = before.replace(old, new, 1)
    compile(after, PIPELINE, "exec")
    return after


def locked_pipeline_change(root):
    policy = json.loads((HERE / "source-patches.lock.json").read_text(encoding="utf-8"))["pipeline"]
    if policy["target"] != PIPELINE or policy["before_sha256"] != PIPELINE_BEFORE:
        raise ValueError("H3 pipeline policy differs from reviewed input")
    before = (root / PIPELINE).read_bytes()
    after = transform_pipeline(before)
    if hashlib.sha256(after).hexdigest() != policy["after_sha256"]:
        raise ValueError("H3 pipeline patch differs from reviewed output")
    if hashlib.sha256((HERE / policy["patch_file"]).read_bytes()).hexdigest() != policy["patch_sha256"]:
        raise ValueError("Reviewed H3 pipeline diff changed")
    return PIPELINE, before, after

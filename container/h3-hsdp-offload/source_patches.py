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
    source = source.replace(anchor, anchor + "from .hsdp_migration import (capture_components, restore_components, move_component,\n"
                            "    offload_profile_scope, timed_offload, count_offload)\n"
                            "import logging\nimport os\nimport threading\n", 1)
    start = source.index("def _capture_tensor_devices(")
    end = source.index("\n\nclass SequentialOffloadHook", start)
    source = source[:start] + '''def _capture_tensor_devices(modules):
    # Capture owning components, not naked .data handles: FSDP owns shard refs.
    return capture_components(modules)


def _restore_tensor_devices(devices):
    restore_components(devices)


# One worker owns one serialized GPU action. A lease is deliberately process-
# wide: a ContextVar alone would hide the active GPU scope from other threads.
_DIT_SCOPE_LOCK = threading.RLock()
_ACTIVE_DIT_SCOPE = None


def _scope_signature(hook):
    return (hook.device, tuple(id(item) for item in hook.offload_targets),
            hook.pin_memory, hook.use_hsdp, hook.offload_after_context,
            hook.scope_fastpath)


def _check_dit_scope(hook, module=None):
    with _DIT_SCOPE_LOCK:
        scope = _ACTIVE_DIT_SCOPE
        if scope is None:
            return None
        if scope["hook"] is not hook or scope["thread"] != threading.get_ident():
            raise RuntimeError("Another component/thread cannot enter an active DiT residency scope")
        if module is not None and scope["module"] is not module:
            raise RuntimeError("DiT residency scope belongs to another module")
        if scope["signature"] != _scope_signature(hook):
            raise RuntimeError("Offload configuration changed inside active DiT residency scope")
        return scope


@contextmanager
def _dit_scope_lease(hook, module):
    global _ACTIVE_DIT_SCOPE
    with _DIT_SCOPE_LOCK:
        scope = _check_dit_scope(hook, module)
        outer = scope is None
        if outer:
            mode = os.environ.get("H3_OFFLOAD_FASTPATH", "1")
            if mode not in {"0", "1"}:
                raise ValueError("H3_OFFLOAD_FASTPATH must be 0 or 1")
            scope = {"hook": hook, "module": module, "thread": threading.get_ident(),
                     "signature": _scope_signature(hook), "ready": False,
                     "enabled": mode == "1", "depth": 0, "checking": False}
            _ACTIVE_DIT_SCOPE = scope
        scope["depth"] += 1
    try:
        yield scope, outer
    finally:
        with _DIT_SCOPE_LOCK:
            scope["depth"] -= 1
            if outer:
                # Cleanup errors must never leave a reusable residency proof.
                scope["ready"] = False
                _ACTIVE_DIT_SCOPE = None
''' + source[end:]
    start = source.index('        """Move module parameters and buffers to device.')
    end = source.index("\n    def _to_cpu(", start)
    source = source[:start] + '''        """Move one idle component through native PyTorch/FSDP conversion."""
        return move_component(
            module, target_device, non_blocking=non_blocking, pin_memory=pin_memory
        )
''' + source[end:]
    # Only DiT receives the lease optimization; other components retain their
    # existing activation/release lifecycle. Keep a same-image control switch.
    source = source.replace("        offload_after_context: bool = True,\n",
                            "        offload_after_context: bool = True,\n        scope_fastpath: bool = False,\n", 1)
    source = source.replace("        self.offload_after_context = offload_after_context\n",
                            "        self.offload_after_context = offload_after_context\n        self.scope_fastpath = scope_fastpath\n", 1)
    source = source.replace("                offload_after_context=id(dit_mod) in selected_dit_ids,\n",
                            "                offload_after_context=id(dit_mod) in selected_dit_ids,\n                scope_fastpath=True,\n", 1)
    source = source.replace("    def _to_cpu(self, module: nn.Module) -> None:\n",
                            "    def _to_cpu(self, module: nn.Module) -> None:\n"
                            "        scope = _check_dit_scope(self)\n"
                            "        if scope is not None and scope[\"ready\"] and not scope[\"checking\"]:\n"
                            "            raise RuntimeError(\"Cannot migrate a component inside an active DiT scope\")\n", 1)
    source = source.replace("            current_omni_platform.empty_cache()\n",
                            "            with timed_offload(\"allocator_empty_cache\"):\n                current_omni_platform.empty_cache()\n", 1)
    source = source.replace("    def _to_gpu(self, module: nn.Module) -> None:\n",
                            "    def _to_gpu(self, module: nn.Module) -> None:\n"
                            "        scope = _check_dit_scope(self)\n"
                            "        if scope is not None and scope[\"ready\"] and not scope[\"checking\"]:\n"
                            "            raise RuntimeError(\"Cannot migrate a component inside an active DiT scope\")\n", 1)
    for name, following in (("_to_cpu", "_to_gpu"), ("_to_gpu", "pre_forward")):
        start = source.index("    def " + name + "(")
        end = source.index("    def " + following + "(", start)
        method = source[start:end].rstrip("\n")
        declaration, body = method.split("\n", 1)
        source = source[:start] + declaration + "\n        with _DIT_SCOPE_LOCK:\n" + "\n".join(
            "    " + line if line else line for line in body.split("\n")) + "\n\n" + source[end:]
    start = source.index("    def pre_forward(self, module:")
    end = source.index("\n\ndef apply_sequential_offload(", start)
    source = source[:start] + '''    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple, dict]:
        scope = _check_dit_scope(self, module)
        if scope is not None and scope["ready"] and scope["enabled"]:
            # The outer scope already activated the entire DiT before native
            # FSDP pre-forward. No tensor traversal, copy or device sync here.
            count_offload("hook_fastpath")
            return args, kwargs
        with _DIT_SCOPE_LOCK, offload_profile_scope("hook:" + module.__class__.__name__):
            # Serialize the check plus the actual transition against lease
            # acquisition in another thread, including calls outside a scope.
            scope = _check_dit_scope(self, module)
            if scope is not None:
                scope["checking"] = True
            try:
                with timed_offload("hook_pre_forward"):
                    count_offload("hook_slowpath")
                    for target in self.offload_targets:
                        self._to_cpu(target)
                    self._to_gpu(module)
                    with timed_offload("hook_synchronize"):
                        current_omni_platform.synchronize()
                    if logger.isEnabledFor(logging.DEBUG):
                        with timed_offload("debug_memory_query"):
                            free_gb = current_omni_platform.get_free_memory() / 1024 / 1024 / 1024
                        logger.debug("Swapped: %s -> CPU, %s -> %s, free memory: %.4f GB",
                                     [t.__class__.__name__ for t in self.offload_targets],
                                     module.__class__.__name__, str(self.device), free_gb)
            finally:
                if scope is not None:
                    scope["checking"] = False
        return args, kwargs
''' + source[end:]
    start = source.index("@contextmanager\ndef sequential_offload_component(")
    end = source.index("\n\nclass ModelLevelOffloadBackend", start)
    source = source[:start] + '''@contextmanager
def sequential_offload_component(module: nn.Module) -> Iterator[None]:
    """Activate/release components; reuse DiT residency only inside its scope."""
    hook = _get_sequential_offload_hook(module)
    lease = _dit_scope_lease(hook, module) if hook.scope_fastpath else nullcontext((None, True))
    with offload_profile_scope("component:" + module.__class__.__name__), lease as (scope, outer):
        if not outer:
            count_offload("nested_dit_scope")
            yield
            return
        release_after = scope["signature"][4] if scope is not None else hook.offload_after_context
        try:
            hook.pre_forward(module)
            if scope is not None:
                scope["ready"] = True
            with timed_offload("component_body"):
                yield
            if scope is not None:
                _check_dit_scope(hook, module)
        except BaseException:
            if scope is not None:
                scope["ready"] = False
            try:
                if release_after:
                    hook._to_cpu(module)
            except BaseException:
                logger.exception("Failed to release %s after component failure", module.__class__.__name__)
            raise
        else:
            if scope is not None:
                scope["ready"] = False
            if release_after:
                hook._to_cpu(module)
''' + source[end:]
    source = source.replace("from contextlib import contextmanager\n", "from contextlib import contextmanager, nullcontext\n", 1)
    # Hook removal while leased would invalidate the proof without a transition.
    start = source.index("def remove_sequential_offload(")
    end = source.index("\n\ndef _get_sequential_offload_hook", start)
    body = source[start:end]
    marker = "    first_error: BaseException | None = None\n"
    if body.count(marker) != 1:
        raise ValueError("Unexpected hook removal guard anchor")
    before, remainder = body.split(marker, 1)
    body = before + '''    with _DIT_SCOPE_LOCK:
        if _ACTIVE_DIT_SCOPE is not None:
            raise RuntimeError("Cannot remove offload hooks while a DiT residency scope is active")
''' + "\n".join("    " + line if line else line for line in (marker + remainder).split("\n"))
    source = source[:start] + body + source[end:]
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

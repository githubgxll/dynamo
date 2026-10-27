# SPDX-License-Identifier: Apache-2.0
"""Component residency for the pinned PyTorch 2.13 FSDP2 inference runtime.

This is component parking, not FSDP CPUOffloadPolicy or layerwise offload.
FSDP's CUDA mesh/execution device stay unchanged while its sharded parameters
are parked on CPU. Every component must return before its next forward.

Use the native Module/FSDPModule conversion lifecycle: first reshard all FSDP
units, then call each module's nonrecursive _apply in graph postorder. Native
FSDP _apply refreshes its parameter/storage references after its children move.
No direct Parameter.data or DTensor._local_tensor assignment is used here.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule
from torch.distributed.tensor import DTensor


_CONVERSION_LOCK = threading.RLock()


@dataclass(frozen=True)
class ComponentPlacement:
    module: nn.Module
    device: torch.device | None


def _postorder(root: nn.Module) -> list[tuple[str, nn.Module]]:
    """Iterative, identity-deduplicated traversal, including Cache-DiT cycles."""
    result: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()
    pending = [("", root, False)]
    while pending:
        name, module, expanded = pending.pop()
        if expanded:
            result.append((name, module))
            continue
        if id(module) in seen:
            continue
        seen.add(id(module))
        pending.append((name, module, True))
        for child_name, child in reversed(tuple(module._modules.items())):
            if child is not None:
                path = f"{name}.{child_name}" if name else child_name
                pending.append((path, child, False))
    return result


def _slots(nodes: list[tuple[str, nn.Module]]):
    for path, module in nodes:
        for kind, values in (("parameter", module._parameters), ("buffer", module._buffers)):
            for name, tensor in values.items():
                if tensor is not None:
                    yield kind, f"{path}.{name}" if path else name, module, name, tensor


def _local(tensor: torch.Tensor) -> torch.Tensor:
    # Read-only inspection. Moving/rebinding this local tensor directly would
    # leave the DTensor wrapper and FSDP cached shard data inconsistent.
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor


def _describe(name: str, tensor: torch.Tensor) -> str:
    try:
        return (f"{name}: type={type(tensor).__name__}, device={tensor.device}, "
                f"local_device={_local(tensor).device}, shape={tuple(tensor.shape)}")
    except Exception:
        return f"{name}: type={type(tensor).__name__}, unreadable tensor metadata"


def _validate(nodes: list[tuple[str, nn.Module]], *, allow_mixed: bool = False):
    devices: set[torch.device] = set()
    seen: set[int] = set()
    for kind, name, _, _, tensor in _slots(nodes):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        local = _local(tensor)
        if tensor.device != local.device:
            raise RuntimeError("Inconsistent global/local tensor device: " + _describe(name, tensor))
        if tensor.device.type not in {"cpu", "cuda"} or tensor.layout != torch.strided:
            raise RuntimeError("Unsupported residency tensor: " + _describe(name, tensor))
        if isinstance(tensor, DTensor):
            if type(tensor) is not DTensor or type(local) is not torch.Tensor:
                raise RuntimeError("Unsupported DTensor/local subclass: " + _describe(name, tensor))
        elif type(tensor) not in {torch.Tensor, nn.Parameter}:
            # H3's TP text encoder owns vLLM ModelWeightParameter, RowvLLMParameter,
            # etc. They are dense Parameter subclasses with Python loading/axis
            # metadata, not arbitrary dispatching tensor wrappers. Preserve them
            # through the native shallow-copy conversion path below.
            if not (isinstance(tensor, nn.Parameter)
                    and type(tensor).__module__ == "vllm.model_executor.parameter"):
                raise RuntimeError("Unsupported Tensor subclass: " + _describe(name, tensor))
        if kind == "parameter" and tensor.grad is not None:
            raise RuntimeError("Residency migration is inference-only; gradient present: " + name)
        devices.add(tensor.device)
    if len(devices) > 1 and not allow_mixed:
        raise RuntimeError("Component must have homogeneous original residency; got "
                           + ", ".join(sorted(map(str, devices))))
    return devices


def _check_storage_aliases(nodes: list[tuple[str, nn.Module]]) -> None:
    """Allow tied objects, reject distinct views whose aliasing .to would split."""
    storage_owners: dict[tuple[str, int], tuple[int, str]] = {}
    seen: set[int] = set()
    for _, name, _, _, tensor in _slots(nodes):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        local = _local(tensor)
        storage = local.untyped_storage()
        if not storage.nbytes():
            continue
        key = (str(local.device), storage.data_ptr())
        previous = storage_owners.setdefault(key, (id(tensor), name))
        if previous[0] != id(tensor):
            raise RuntimeError("Distinct tensors share storage; migration would split aliases: "
                               f"{previous[1]} and {name}")


def _fsdp_units(nodes: list[tuple[str, nn.Module]], target: torch.device):
    units = []
    for path, module in nodes:
        if not isinstance(module, FSDPModule):
            continue
        state = module._get_fsdp_state()
        groups = state._fsdp_param_groups
        for group in groups:
            if group._training_state.name != "IDLE":
                raise RuntimeError(f"Cannot move active FSDP component {path}: {group._training_state}")
            if isinstance(group.offload_policy, CPUOffloadPolicy):
                raise RuntimeError("Component parking must not mix with FSDP CPUOffloadPolicy: " + path)
            if group.device.type != "cuda":
                raise RuntimeError("Only CUDA FSDP component residency is supported: " + path)
            if target.type == "cuda" and group.device != target:
                raise RuntimeError(f"FSDP execution device must remain {group.device}; got {target}")
            if group._all_gather_result is not None:
                raise RuntimeError("Cannot move FSDP with a pending all-gather: " + path)
        units.append((path, module))
    return units


@contextmanager
def _preserve_parameter_identity():
    # DTensor is always swapped by Torch's native _apply. Ordinary dense vLLM
    # Parameter subclasses must instead use native compatible shallow-copy:
    # forced swap would erase their Python class/weight-loader/axis metadata.
    # Convert() verifies compatibility before native _apply may update a slot.
    # Restore both documented switches even on failure. This lock serializes
    # our own conversions; the worker must exclude unrelated .to()/inference.
    old_swap = torch.__future__.get_swap_module_params_on_conversion()
    old_overwrite = torch.__future__.get_overwrite_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(False)
    torch.__future__.set_overwrite_module_params_on_conversion(False)
    try:
        yield
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(old_overwrite)
        torch.__future__.set_swap_module_params_on_conversion(old_swap)


def _move_component(module: nn.Module, target_device: torch.device, *,
                    non_blocking: bool, pin_memory: bool, allow_mixed: bool) -> bool:
    target = torch.device(target_device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    if target.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"Unsupported target device: {target}")
    nodes = _postorder(module)
    devices = _validate(nodes, allow_mixed=allow_mixed)
    if not devices or devices == {target}:
        # Native FSDP pre-forward can already be FORWARD when the sequential
        # wrapper asks to load an already-resident component. A no-op performs
        # no reshard/conversion and must not reject that valid hook ordering.
        return False
    units = _fsdp_units(nodes, target)

    # Synchronize the source CUDA execution stream before freeing or moving its
    # shard backing storage. HSDP movement is always blocking, independent of the
    # caller's non_blocking flag. No distributed collective is introduced here.
    for device in sorted((d for d in devices if d.type == "cuda"), key=str):
        torch.cuda.synchronize(device)
    if units:
        non_blocking = False

    # Mandatory first pass: native _apply starts with reshard, but doing that
    # only in postorder would let unwrapped children move UNSHARDED parameters
    # that the owning FSDP parent subsequently discards/re-registers.
    for _, unit in units:
        unit.reshard()
    nodes = _postorder(module)
    _validate(nodes, allow_mixed=allow_mixed)
    _check_storage_aliases(nodes)

    parameter_slots = [(owner, key, tensor, name)
                       for kind, name, owner, key, tensor in _slots(nodes) if kind == "parameter"]
    parameter_types = {id(tensor): type(tensor) for _, _, tensor, _ in parameter_slots}
    parameter_attributes = {id(tensor): dict(tensor.__dict__)
                            for _, _, tensor, _ in parameter_slots if not isinstance(tensor, DTensor)}
    buffer_ids = {id(tensor) for kind, _, _, _, tensor in _slots(nodes) if kind == "buffer"}
    parameter_ids = {id(tensor) for _, _, tensor, _ in parameter_slots}
    tensor_names = {id(tensor): name for _, name, _, _, tensor in _slots(nodes)}
    if buffer_ids & parameter_ids:
        raise RuntimeError("One tensor is registered as both parameter and buffer; unsupported")
    buffer_memo: dict[int, torch.Tensor] = {}
    current_name = "<component>"

    def convert(tensor):
        if id(tensor) in buffer_memo:
            return buffer_memo[id(tensor)]
        try:
            moved = tensor.to(target, non_blocking=non_blocking)
            if pin_memory and target.type == "cpu" and not isinstance(moved, DTensor):
                if not moved.is_pinned():
                    moved = moved.pin_memory()
            if id(tensor) in parameter_ids and not isinstance(tensor, DTensor):
                if not torch._has_compatible_shallow_copy_type(tensor, moved):
                    raise RuntimeError("Native dense conversion cannot preserve Parameter identity")
            if id(tensor) in buffer_ids:
                buffer_memo[id(tensor)] = moved
            return moved
        except Exception as exc:
            raise RuntimeError("Tensor migration failed at " + _describe(tensor_names.get(id(tensor), current_name), tensor)
                               + f", target={target}") from exc

    with torch.no_grad(), _preserve_parameter_identity():
        for name, child in nodes:
            current_name = name or "<root>"
            # Dispatch through the real FSDP override, not nn.Module._apply.
            # recurse=False avoids Cache-DiT cycles; postorder refreshes each
            # owning FSDP group's cached references after all owned slots move.
            child._apply(convert, recurse=False)

    for owner, key, before, name in parameter_slots:
        if owner._parameters[key] is not before:
            raise RuntimeError("Native conversion changed parameter identity: " + name)
        if type(before) is not parameter_types[id(before)]:
            raise RuntimeError("Native conversion changed parameter class: " + name)
        attributes = parameter_attributes.get(id(before))
        if attributes is not None and (before.__dict__.keys() != attributes.keys() or any(
            before.__dict__[key] is not value for key, value in attributes.items()
        )):
            raise RuntimeError("Native conversion changed parameter attributes: " + name)
    final_nodes = _postorder(module)
    if _validate(final_nodes) != {target}:
        raise RuntimeError(f"Component migration did not reach {target}")
    for name, unit in units:
        for group in unit._get_fsdp_state()._fsdp_param_groups:
            if not group.is_sharded:
                raise RuntimeError("FSDP did not remain sharded during migration: " + name)
            for param in group.fsdp_params:
                info = param._module_info
                if getattr(info.module, info.param_name) is not param.sharded_param:
                    raise RuntimeError("FSDP lost registered parameter identity: " + name)
                local = _local(param.sharded_param)
                if param._sharded_param_data.device != local.device:
                    raise RuntimeError("FSDP cached shard device differs: " + name)
                if local.untyped_storage().nbytes() and (
                    param._sharded_param_data.untyped_storage().data_ptr()
                    != local.untyped_storage().data_ptr()
                ):
                    raise RuntimeError("FSDP cached shard storage differs: " + name)
    return True


def move_component(module: nn.Module, target_device: torch.device, *,
                   non_blocking: bool = False, pin_memory: bool = False) -> bool:
    """Move an idle inference component via native FSDP-aware conversion.

Unsupported layouts fail rather than silently changing tensor aliasing. A
failed migration must abort generation; only initialization rollback may use
restore_components. This function does not make a partially failed worker safe
to resume. Pinning keeps the old backend policy: ordinary CPU tensors only.
"""
    with _CONVERSION_LOCK:
        return _move_component(module, target_device, non_blocking=non_blocking,
                               pin_memory=pin_memory, allow_mixed=False)


def capture_components(modules: Iterable[nn.Module]) -> list[ComponentPlacement]:
    """Record homogeneous original component placement before enable mutates it."""
    result = []
    seen: set[int] = set()
    with _CONVERSION_LOCK:
        for module in modules:
            if id(module) in seen:
                continue
            seen.add(id(module))
            devices = _validate(_postorder(module))
            result.append(ComponentPlacement(module, next(iter(devices)) if devices else None))
    return result


def restore_components(snapshot: Iterable[ComponentPlacement]) -> None:
    """Best-effort initialization rollback through the same FSDP lifecycle.

Rollback may encounter mixed devices left by a failed conversion. It restores
the captured homogeneous target, but reports any failure rather than claiming
that the partially initialized worker may continue inference.
"""
    errors: list[Exception] = []
    with _CONVERSION_LOCK:
        for placement in reversed(list(snapshot)):
            if placement.device is None:
                continue
            try:
                _move_component(placement.module, placement.device, non_blocking=False,
                                pin_memory=False, allow_mixed=True)
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise RuntimeError(f"Failed to restore {len(errors)} component placement(s); discard worker") from errors[0]

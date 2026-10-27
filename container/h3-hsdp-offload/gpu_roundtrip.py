#!/usr/bin/env python3
"""Real eight-rank FSDP2 migration gate; never a substitute for H3 media acceptance.

Run under torchrun in the final image. No downloads or model weights are needed.
CPU-only execution and fewer than eight ranks are failures, never skips.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
import uuid


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def module_graph(root):
    """Enumerate direct slots once even with a Cache-DiT-style back reference."""
    seen = set()
    pending = [("", root)]
    while pending:
        name, module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        yield name, module
        pending.extend((f"{name}.{key}".lstrip("."), child)
                       for key, child in reversed(list(module._modules.items())) if child is not None)


def slots(root):
    for path, module in module_graph(root):
        for kind, collection in (("parameter", module._parameters), ("buffer", module._buffers)):
            for name, tensor in collection.items():
                if tensor is not None:
                    yield (kind, f"{path}.{name}".lstrip(".")), tensor


def run(args, report):
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    rank, local_rank, world = (int(os.environ.get(key, "-1"))
                               for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 8 or not 0 <= rank < 8 or not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError("This gate requires torchrun with exactly eight CUDA ranks (mesh 1 x 8)")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A real CUDA BF16 runtime is required; CPU execution cannot pass")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    token = [datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
             if rank == 0 else None]
    dist.broadcast_object_list(token, src=0)
    directory = args.output_dir / token[0]
    report.update(run_id=token[0], rank=rank, local_rank=local_rank, world_size=world,
                  evidence_directory=str(directory), device=torch.cuda.get_device_name(local_rank),
                  torch=torch.__version__, cycles=args.cycles, phases=[])
    destination = directory / f"rank-{rank}.json"
    save(destination, report)

    def phase(name, operation):
        outcome = {"rank": rank, "phase": name, "status": "PASS"}
        result = None
        try:
            result = operation()
            torch.cuda.synchronize(device)
        except Exception:
            outcome.update(status="FAIL", traceback=traceback.format_exc())
        report["phases"].append(outcome)
        save(destination, report)
        collected = [None] * world
        dist.all_gather_object(collected, outcome)
        failures = [item for item in collected if item["status"] != "PASS"]
        if failures:
            raise RuntimeError(json.dumps({"phase": name, "rank_failures": failures}, indent=2))
        return result

    def provenance():
        import runtime
        import torch_contract
        from vllm_omni.diffusion.offloader import sequential_backend as backend
        report["image_identity"] = runtime.verify_installed()
        report["torch_contract"] = torch_contract.verify()
        path = Path(backend.__file__).resolve()
        report["installed_backend"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        report["package_versions"] = {name: importlib.metadata.version(name)
                                      for name in ("vllm", "vllm-omni", "torch")}
        return backend

    backend = phase("installed-image-and-torch-contract", provenance)

    def vllm_parameter_context():
        from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
        from vllm.model_executor import parameter
        init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank,
                                     backend="nccl", timeout=timedelta(seconds=180))
        initialize_model_parallel(tensor_model_parallel_size=8, backend="nccl")
        path = Path(parameter.__file__).resolve()
        report["installed_vllm_parameter"] = {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "class": "vllm.model_executor.parameter.ModelWeightParameter",
            "tensor_parallel_size": 8,
            "scope": "Real inherited parameter class and TP metadata; synthetic encoder computation",
        }
        return parameter.ModelWeightParameter

    ModelWeightParameter = phase("real-vllm-TP8-parameter-context", vllm_parameter_context)
    torch.manual_seed(20261027)
    torch.cuda.manual_seed_all(20261027)
    mesh = init_device_mesh("cuda", (1, 8), mesh_dim_names=("replicate", "shard"))
    width = 1024

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(width, width, bias=False, device=device, dtype=torch.bfloat16)
            self.right = nn.Linear(width, width, bias=False, device=device, dtype=torch.bfloat16)
            self.right.weight = self.left.weight
            self.register_buffer("bias", torch.full((width,), 0.125, device=device, dtype=torch.bfloat16))
            self.register_buffer("bias_alias", self.bias)

        def forward(self, value):
            return self.left(value) * 0.25 + self.right(value) * 0.25 + self.bias + self.bias_alias

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block(), Block()])
            self.out = nn.Linear(width, width, bias=False, device=device, dtype=torch.bfloat16)
            self.ignored_scale = nn.Parameter(torch.ones(width, device=device, dtype=torch.bfloat16),
                                              requires_grad=False)
            self.register_buffer("integer_buffer", torch.ones(width, device=device, dtype=torch.int64))

        def forward(self, value):
            for block in self.blocks:
                value = block(value)
            return self.out(value) * self.ignored_scale + self.integer_buffer

    model = Model().eval().requires_grad_(False)
    encoder = nn.Linear(width, width, bias=False, device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    def fixture_weight_loader(parameter, loaded_weight):
        with torch.no_grad():
            parameter.copy_(loaded_weight)

    encoder.weight = ModelWeightParameter(data=encoder.weight.detach(), input_dim=1, output_dim=0,
                                         weight_loader=fixture_weight_loader)
    encoder.weight.hsdp_test_marker = {"component": "encoder", "rank": rank}
    encoder.register_parameter("weight_alias", encoder.weight)
    if encoder.weight.tp_size != 8 or encoder.weight.tp_rank != rank:
        raise AssertionError("Real vLLM parameter did not inherit TP8 rank metadata")
    inputs = torch.randn(4, width, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        encoder_reference = encoder(inputs).detach().cpu().clone()
        unsharded_reference = model(inputs).detach().cpu().clone()
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    for block in model.blocks:
        fully_shard(block, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
    fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=policy,
                ignored_params={model.ignored_scale})
    units = [module for _, module in module_graph(model) if isinstance(module, FSDPModule)]
    if len(units) != 3:
        raise RuntimeError(f"Expected root and two nested FSDP groups, got {len(units)}")

    def reshard():
        for unit in units:
            unit.reshard()

    def forward_once():
        with torch.no_grad():
            result = model(inputs).detach().cpu().clone()
        reshard()
        return result

    resident_reference = phase("resident-fsdp-forward", forward_once)
    phase("resident-fsdp-vs-unsharded", lambda: torch.testing.assert_close(
        resident_reference, unsharded_reference, rtol=0, atol=0))
    # Introduce the cycle only after fully_shard/lazy initialization, as Cache-DiT
    # can wrap an already constructed graph. Do not call Module.train/to here.
    model.cache_backref = nn.Module()
    model.cache_backref.transformer = model

    def signature(tensor):
        result = {"shape": tuple(tensor.shape), "stride": tuple(tensor.stride()), "dtype": str(tensor.dtype),
                  "dtensor": isinstance(tensor, DTensor)}
        if isinstance(tensor, DTensor):
            result.update(mesh=tensor.device_mesh.mesh.tolist(), mesh_type=tensor.device_mesh.device_type,
                          mesh_names=tuple(tensor.device_mesh.mesh_dim_names or ()),
                          placements=tuple(str(item) for item in tensor.placements))
        return result

    def local(tensor):
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def capture(root):
        result = {}
        for key, value in slots(root):
            record = {"id": id(value), "class": type(value), "signature": signature(value),
                      "local_value": local(value).detach().cpu().clone()}
            if isinstance(value, ModelWeightParameter):
                record.update(custom_dict_id=id(value.__dict__), custom_attributes=dict(value.__dict__))
            result[key] = record
        return result

    expected_model, expected_encoder = capture(model), capture(encoder)
    if not any(item["signature"]["dtensor"] for item in expected_model.values()):
        raise RuntimeError("Fixture has no genuine DTensor; cannot validate this defect")

    def verify(root, expected, target):
        current = dict(slots(root))
        if current.keys() != expected.keys():
            raise AssertionError("Registered parameter/buffer slots changed")
        alias_map, reverse_map = {}, {}
        for key, value in current.items():
            original = expected[key]
            if key[0] == "parameter" and id(value) != original["id"]:
                raise AssertionError(f"Parameter object replaced: {key}")
            if type(value) is not original["class"]:
                raise AssertionError(f"Tensor subclass changed: {key}")
            if "custom_attributes" in original:
                attributes = original["custom_attributes"]
                if id(value.__dict__) != original["custom_dict_id"] or value.__dict__.keys() != attributes.keys():
                    raise AssertionError(f"vLLM parameter attribute dictionary replaced or changed: {key}")
                if any(value.__dict__[name] is not expected for name, expected in attributes.items()):
                    raise AssertionError(f"vLLM parameter custom attribute identity changed: {key}")
                if value.weight_loader is not fixture_weight_loader or value.hsdp_test_marker["rank"] != rank:
                    raise AssertionError(f"vLLM parameter loader or marker changed: {key}")
            old_identity = (key[0], original["id"])
            new_identity = (key[0], id(value))
            if alias_map.setdefault(old_identity, new_identity) != new_identity:
                raise AssertionError(f"Shared tensor alias split: {key}")
            if reverse_map.setdefault(new_identity, old_identity) != old_identity:
                raise AssertionError(f"Distinct tensors became aliases: {key}")
            if signature(value) != original["signature"]:
                raise AssertionError(f"Tensor shape/dtype/DTensor mesh/placements changed: {key}")
            if value.device != target or local(value).device != target or value.data.device != target:
                raise AssertionError(f"Wrapper/local/storage device mismatch: {key}")
            torch.testing.assert_close(local(value).detach().cpu(), original["local_value"], rtol=0, atol=0)
            if target.type == "cpu" and not isinstance(value, DTensor) and not value.is_pinned():
                raise AssertionError(f"Ordinary CPU tensor is not pinned: {key}")
        for _, unit in module_graph(root):
            if not isinstance(unit, FSDPModule):
                continue
            state = unit._get_fsdp_state()
            if state._device != device:
                raise AssertionError("FSDP execution device changed while shards were parked")
            for group in state._fsdp_param_groups:
                if not group.is_sharded or group.device != device:
                    raise AssertionError("FSDP group state/device is inconsistent")
                for param in group.fsdp_params:
                    registered = getattr(param._module_info.module, param._module_info.param_name)
                    if registered is not param.sharded_param or param._sharding_spec is not registered._spec:
                        raise AssertionError("FSDP cached parameter/spec no longer matches the module")
                    shard = registered.to_local()
                    if param._sharded_param_data.device != target:
                        raise AssertionError("FSDP cached shard remains on the previous device")
                    if (param._sharded_param_data.untyped_storage().data_ptr()
                            != shard.untyped_storage().data_ptr()):
                        raise AssertionError("FSDP all-gather backing points at stale storage")
                    for temporary in [*param.all_gather_outputs, *param._unsharded_inner_tensors]:
                        if temporary.untyped_storage().nbytes() != 0:
                            raise AssertionError("Reshard retained allocated unsharded weight storage")
                    info = param._module_info
                    for shared_module, shared_name in zip(info.shared_modules, info.shared_param_names):
                        if getattr(shared_module, shared_name) is not registered:
                            raise AssertionError("FSDP shared registration was broken")

    cpu = torch.device("cpu")

    def explicit_roundtrip():
        snapshot = backend._capture_tensor_devices([model, encoder])
        if not backend.SequentialOffloadHook._move_params(model, cpu, pin_memory=True):
            raise AssertionError("GPU to CPU movement incorrectly reported no change")
        verify(model, expected_model, cpu)
        if backend.SequentialOffloadHook._move_params(model, cpu, pin_memory=True):
            raise AssertionError("Repeated CPU offload is not a no-op")
        verify(model, expected_model, cpu)
        backend._restore_tensor_devices(snapshot)
        verify(model, expected_model, device)
        verify(encoder, expected_encoder, device)

    phase("installed-mover-capture-restore-and-repeated-cpu", explicit_roundtrip)

    def pending_all_gather():
        handle = model.unshard(async_op=True)
        if handle is None:
            raise AssertionError("Async native unshard did not expose its completion handle")
        try:
            try:
                backend.SequentialOffloadHook._move_params(model, cpu, pin_memory=True)
            except RuntimeError as exc:
                if "pending all-gather" not in str(exc):
                    raise AssertionError("Pending collective was rejected for an unexpected reason") from exc
            else:
                raise AssertionError("Migration accepted a pending FSDP all-gather")
        finally:
            handle.wait()
            reshard()
        verify(model, expected_model, device)

    phase("pending-all-gather-rejected-before-mutation", pending_all_gather)

    def initially_unsharded():
        # Exercise the mandatory first-pass reshard: moving ordinary child
        # slots first would otherwise move/discard temporary full parameters.
        for unit in units:
            unit.unshard(async_op=False)
        backend.SequentialOffloadHook._move_params(model, cpu, pin_memory=True)
        verify(model, expected_model, cpu)
        backend.SequentialOffloadHook._move_params(model, device)
        verify(model, expected_model, device)
        torch.testing.assert_close(forward_once(), resident_reference, rtol=0, atol=0)

    phase("unsharded-entry-is-resharded-before-child-migration", initially_unsharded)
    phase("register-hooks-and-initial-offload", lambda: backend.apply_sequential_offload(
        [model], [encoder], device, pin_memory=True, use_hsdp=True, offload_initial_dits=True))
    phase("initial-offload-integrity", lambda: verify(model, expected_model, cpu))

    for cycle in range(args.cycles):
        def conditioning():
            # Repeated conditioning invokes the original failure location twice.
            with torch.no_grad():
                for _ in range(2):
                    actual = encoder(inputs).detach().cpu()
                    torch.testing.assert_close(actual, encoder_reference, rtol=0, atol=0)
            verify(model, expected_model, cpu)
            verify(encoder, expected_encoder, device)

        phase(f"cycle-{cycle}-conditioning", conditioning)

        def denoising():
            # Multiple forwards while resident, then one component-level release.
            with backend.sequential_offload_component(model):
                verify(model, expected_model, device)
                verify(encoder, expected_encoder, cpu)
                for _ in range(3):
                    actual = forward_once()
                    torch.testing.assert_close(actual, resident_reference, rtol=0, atol=0)
                    verify(model, expected_model, device)
            verify(model, expected_model, cpu)
            if model.cache_backref.transformer is not model:
                raise AssertionError("Cache-DiT-style cycle was altered")

        phase(f"cycle-{cycle}-resident-forwards-reshard-and-offload", denoising)

    phase("remove-hooks", lambda: backend.remove_sequential_offload([model, encoder]))
    from vllm.distributed.parallel_state import destroy_model_parallel
    phase("destroy-vllm-fixture-model-parallel", destroy_model_parallel)
    report.update(status="PASS_SYNTHETIC_8GPU_HSDP_ROUNDTRIP_ONLY", mesh=[1, 8],
                  nested_fsdp_units=len(units), parameter_buffer_slots=len(expected_model),
                  forward_comparison="bitwise identical to resident BF16 reference",
                  boundaries=["No full H3 model or media generation", "No production peak-memory claim",
                              "No video-gateway service acceptance", "No deployment mutation"])
    save(destination, report)
    completed = [None] * world
    dist.all_gather_object(completed, {"rank": rank, "status": report["status"], "file": destination.name})
    if rank == 0:
        if {item["rank"] for item in completed} != set(range(8)) or any(
                item["status"] != report["status"] for item in completed):
            raise AssertionError("Cannot issue aggregate PASS without eight passing ranks")
        save(directory / "summary.json", {"status": report["status"], "run_id": token[0],
                                         "cycles": args.cycles, "ranks": completed,
                                         "boundaries": report["boundaries"]})
        print(json.dumps({"status": report["status"], "evidence_directory": str(directory)}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    if args.cycles < 3:
        parser.error("At least three complete offload cycles are required")
    report = {"status": "RUNNING", "scope": "Real synthetic FSDP2 GPU gate, not H3 media acceptance"}
    try:
        run(args, report)
    except Exception:
        report.update(status="FAIL", traceback=traceback.format_exc())
        directory = Path(report.get("evidence_directory", args.output_dir / "startup-failures"))
        rank = int(os.environ.get("RANK", "-1"))
        suffix = "" if "run_id" in report else "-" + uuid.uuid4().hex[:12]
        save(directory / f"rank-{rank}{suffix}.json", report)
        if rank == 0:
            save(directory / "failure-summary.json", {"status": "FAIL", "rank": rank,
                                                       "traceback": report["traceback"]})
        print(report["traceback"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

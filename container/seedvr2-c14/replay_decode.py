"""A2: torchrun single-node SP8; complete VAE decode chunk2 baseline then chunk5."""
from __future__ import annotations

import argparse
from datetime import timedelta
import os
import time
from pathlib import Path

from replay_common import (begin_output, compare, configure_numeric_context, load_tensor, load_window,
                           register_guard_process, require_guard, require_hash, save_failure, save_tensor, tolerance, validate_decode_context,
                           verify_sources, write_json)


def synchronized_local(dist, group, action, operation):
    """Rendezvous after non-collective work: a rank-local failure stops every rank.

    Never wrap a GPU collective in this helper: an asymmetric CUDA failure can
    leave peers inside the model. That failure exits the rank immediately and
    torchrun --max-restarts=0 reaps the complete group; NCCL also has a timeout.
    """
    value, error = None, None
    try:
        value = operation()
    except BaseException as exc:
        error = f"{action}: {type(exc).__name__}: {exc}"
    errors = [None] * 8
    dist.all_gather_object(errors, error, group=group)
    if any(errors):
        raise RuntimeError("; ".join(f"rank{i} {message}" for i, message in enumerate(errors) if message))
    return value


def load_vae(weights, device):
    """Checkpoint has bare VAE keys; native pipeline loader only prepends 'vae.'."""
    import torch
    from safetensors.torch import load_file
    from vllm_omni.diffusion.models.seedvr2.vae import SeedVR2VAE
    state = load_file(str(weights), device="cpu")
    if not state or any(t.is_floating_point() and t.dtype != torch.float16 for t in state.values()):
        raise ValueError("VAE checkpoint must contain FP16 floating weights")
    # No random parameter initialization or text/DiT/noise component is loaded.
    with torch.device("meta"):
        model = SeedVR2VAE()
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model = model.to(device=device).eval()
    model.use_tiling = True
    model.set_parallel_size(8, mode="spatial_shard_height")
    return model


def replay(window_dir: Path, output: Path, weights: Path, policy_path: Path | None = None) -> None:
    register_guard_process(require_guard(window_dir, output, "decode"))
    import torch
    import torch.distributed as dist
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel, get_sp_group, destroy_distributed_env,
    )

    world = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (world != 8 or rank not in range(8) or local_rank not in range(8)
            or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 8
            or "TORCHELASTIC_RUN_ID" not in os.environ):
        raise ValueError("A2 requires a supervised single-node eight-GPU torchrun group (--max-restarts=0)")
    if os.environ.get("SEEDVR2_C14_PROFILE"):
        raise ValueError("Unset SEEDVR2_C14_PROFILE for offline replay; it must not append to A0 capture")
    if os.environ.get("TORCHELASTIC_MAX_RESTARTS", "0") != "0":
        raise ValueError("Replay must not auto-restart or reuse an output ID after failure")
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    # Bootstrap with finite collective deadlines before Omni attaches to WORLD.
    dist.init_process_group("nccl", init_method="env://", world_size=world, rank=rank,
                            timeout=timedelta(seconds=300))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=300))
    owned_output, completed = False, False
    try:
        def preflight():
            window = load_window(window_dir)
            actual = validate_decode_context(window)
            policy = tolerance(window, policy_path)
            weight_hash = require_hash(weights, window.get("model_sha256", {}).get("ema_vae_fp16.safetensors"))
            sources = verify_sources(window, ("vllm_omni.diffusion.models.seedvr2.vae",
                                             "vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard",
                                             "vllm_omni.diffusion.distributed.parallel_state"))
            return window, actual, policy, weight_hash, sources
        window, actual, policy, weight_hash, sources = synchronized_local(dist, control, "preflight", preflight)
        autocast = configure_numeric_context(actual)
        init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank, backend="nccl")
        initialize_model_parallel(sequence_parallel_size=8, ulysses_degree=8, backend="nccl")
        def admit():
            if list(get_sp_group().ranks) != actual["sp_ranks"]:
                raise ValueError("Replay SP rank order differs from capture")
            if rank == 0:
                begin_output(output, "A2-replay-decode", window, {"chunks": [2, 5], "vae_weight_sha256": weight_hash,
                             "actual_config": actual, "source_hashes": sources, "tolerance": policy,
                             "dit": "not instantiated", "rng": "no sampling or random initialization",
                             "failure_supervision": "torchrun no-restart plus 300s WORLD/control timeout"})
        synchronized_local(dist, control, "output-admission", admit)
        owned_output = True
        model = synchronized_local(dist, control, "VAE checkpoint loading", lambda: load_vae(weights, device))
        def inputs():
            latent, latent_meta = load_tensor(window_dir, "vae_decode_input", device, restore_stride=True)
            if latent_meta.get("layout") != "BCTHW" or list(latent.shape) != [1, 16, 5, 138, 230]:
                raise ValueError("A2 requires the complete 17-frame/1840x1104 VAE input")
            if not torch.isfinite(latent).all():
                raise ValueError("Captured VAE latent contains nonfinite values")
            expected, expected_meta = load_tensor(window_dir, "decoded_rgb", device)
            if expected_meta.get("layout") != "BCTHW" or list(expected.shape) != [1, 3, 17, 1104, 1840]:
                raise ValueError("A2 requires complete clamp-before BCTHW decoded RGB")
            return latent, expected
        latent, expected = synchronized_local(dist, control, "input-loading", inputs)
        runs = []
        for chunk in (2, 5):
            torch.cuda.reset_peak_memory_stats(local_rank)
            dist.barrier(group=control)
            start = time.monotonic()
            # One native call = one fresh TemporalContext. Never clear internal
            # temporal history or manually recreate its individual chunks.
            with torch.inference_mode(), torch.autocast("cuda", **autocast):
                decoded = model.decode(latent, chunk_size=chunk)
            torch.cuda.synchronize(local_rank)
            timing = {"rank": rank, "elapsed_seconds": time.monotonic() - start,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(local_rank),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(local_rank)}
            check = synchronized_local(dist, control, "RGB comparison", lambda: compare(decoded, expected, policy))
            ranks = [None] * 8
            dist.all_gather_object(ranks, {"comparison": check, **timing}, group=control)
            def persist():
                write_json(output / f"chunk{chunk}-rank-{rank}.json", {"chunk_size": chunk, "comparison": check, **timing})
                if rank == 0:
                    evidence = save_tensor(output / f"decoded-chunk{chunk}.safetensors", decoded)
                    runs.append({"chunk_size": chunk, "comparison_to_A0": check, "output": evidence, "ranks": ranks})
                    write_json(output / f"chunk{chunk}.json", runs[-1])
            synchronized_local(dist, control, "evidence-write", persist)
            del decoded
            if chunk == 2 and not all(item["comparison"]["pass"] for item in ranks):
                raise RuntimeError("Original chunk2 replay differs on at least one rank; chunk5 was not run")
        def finish():
            if rank == 0:
                write_json(output / "COMPLETE.json", {"status": "PASS_BASELINE_AND_REPLAY_ONLY", "runs": runs,
                           "quality_acceptance": "Full-window and complete-shot quality review remain required"})
        synchronized_local(dist, control, "completion-write", finish)
        completed = True
    except BaseException as error:
        # Do not enter new CUDA collectives or teardown barriers after failure.
        # An uncaught rank exception makes torchrun terminate its peer ranks.
        if owned_output:
            try:
                write_json(output / f"FAILED-rank-{rank}.json", {"status": "FAIL_NO_CANDIDATE_CONCLUSION", "rank": rank,
                                                               "error_type": type(error).__name__, "error": str(error)})
                if rank == 0:
                    save_failure(output, error)
            except (OSError, ValueError):
                pass  # Original failure remains visible in torchrun's durable log.
        raise
    finally:
        if completed:
            dist.destroy_process_group(control)
            destroy_distributed_env()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vae-weights", type=Path, required=True, help="Mounted ema_vae_fp16.safetensors, same SHA as A0")
    parser.add_argument("--tolerance-policy", type=Path)
    args = parser.parse_args()
    replay(args.window, args.output, args.vae_weights, args.tolerance_policy)
    if os.environ.get("RANK") == "0":
        print("PASS A2 chunk2 baseline and chunk5 replay; C14 quality acceptance pending")


if __name__ == "__main__":
    main()

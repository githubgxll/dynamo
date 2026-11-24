"""One vLLM-Omni IC worker per torchrun rank, reused across all requests."""
import os
from pathlib import Path
import sys
import time
import uuid

from suite_contract import document, read, require, utc, verify_fa3


IMAGE_ROOT = Path("/opt/ic-r1")
SOURCE = IMAGE_ROOT / "source"
BACKEND = "FLASH_ATTN_3_HUB"
MODES = {"stock": (False, False, 1), "stream": (True, False, 1), "owner": (False, True, 1),
         "batch2": (False, False, 2), "combo": (True, True, 2)}


def release_info(plan):
    manifest = document(IMAGE_ROOT / "release-manifest.json")
    require(manifest["release_id"] == plan["release_id"], "Plan and image release differ")
    for key in ("parent_image", "source_manifest_sha256", "worker_tree_sha256", "kernel_lock_sha256"):
        require(manifest.get(key), f"Image release lacks {key}")
    return {key: manifest[key] for key in ("release_id", "parent_image", "source_manifest_sha256",
                                           "worker_tree_sha256", "kernel_lock_sha256")}


def verify_origins():
    for name, module in tuple(sys.modules.items()):
        if (name == "vllm_omni" or name.startswith("vllm_omni.")) and getattr(module, "__file__", None):
            require(Path(module.__file__).resolve().is_relative_to(SOURCE), f"Imported {name} outside release source")


def verify_vae(ev, row, rank, tile_work):
    stream, owner, batch = MODES[row["profile"]]
    require(ev.get("rank") == rank and ev.get("profile") == row["profile"] and
            ev.get("stream") is stream and ev.get("owner_only") is owner and ev.get("tile_batch") == batch,
            f"Rank {rank}: wrong VAE mode")
    calls = ev["decoder_calls_by_batch"]
    require(isinstance(calls, dict) and all(type(calls.get(k)) is int and calls[k] >= 0 for k in ("1", "2")) and
            ev["tile_forwards"] == calls["1"] + 2 * calls["2"] == tile_work["decode"] and ev["tile_forwards"] > 0,
            f"Rank {rank}: VAE tile-call evidence differs")
    require(batch == 2 or calls["2"] == 0, "Observed batch-two calls in a singleton mode")
    if row["profile"] == "stock":
        require(ev["full_rgb_chunks_received"] is None, "Stock decode does not instrument RGB broadcast count")
    require(ev["owner_output_frames"] == (row["frames"] if rank == 0 else 0), "VAE owner frame count differs")
    require((ev["sink_chunks"] > 0) == (stream and rank == 0), "VAE streaming chunk ownership differs")
    if owner and rank != 0:
        require(ev["full_rgb_chunks_received"] == 0, "Non-owner received full RGB under owner-only decode")


class NativeWorker:
    def __init__(self, plan, rank):
        initial = time.perf_counter()
        import torch
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor
        from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, DiffusionParallelConfig, OmniDiffusionConfig
        from vllm_omni.diffusion.distributed.parallel_state import get_fs_group, get_sp_group, get_world_group
        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

        self.torch = torch
        self.rank = rank
        parallel = DiffusionParallelConfig(data_parallel_size=1, tensor_parallel_size=1, pipeline_parallel_size=1,
            cfg_parallel_size=1, ulysses_degree=8, ring_degree=1, allgather_degree=1, sequence_parallel_size=8,
            text_encoder_tp_size=1, use_hsdp=True, hsdp_shard_size=8, hsdp_replicate_size=1,
            vae_patch_parallel_size=8, vae_parallel_mode="tile")
        self.config = OmniDiffusionConfig(model=plan["model_root"], model_class_name="MiniMaxH3RegenerateICPipeline",
            num_gpus=8, dtype=torch.bfloat16, parallel_config=parallel, enforce_eager=True,
            diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
            diffusion_offload_config={"mode":"layer", "components":["text_encoder"],
                                      "layer_options":{"text_encoder":{"weight_transfer":"rank-local"}}},
            vae_use_tiling=True, cache_backend="none", cache_strategy="none", diffusion_load_format="default",
            hsdp_weight_load_strategy="full", host_weight_runtime_mode="disabled",
            additional_config={"ic_audio_vae_path": plan["audio_vae_path"], "ic_dit_attention_backend": BACKEND,
                               "ic_vae_profile":"combo"})
        load_utc = utc()
        started = time.perf_counter()
        self.native = DiffusionWorker(local_rank=rank, rank=rank, od_config=self.config, distributed_init_method="env://",
                                      set_proc_title=False)
        self.device = self.native.device
        torch.cuda.synchronize(self.device)
        loaded = time.perf_counter()
        self.pipe = self.native.model_runner.pipeline
        pipe = self.pipe
        self.world = get_world_group()
        require(self.world.world_size == get_sp_group().world_size == get_fs_group().world_size == 8 and
                self.world.rank_in_group == rank, "Unexpected world/SP/HSDP topology")
        require(type(pipe).__name__ == "MiniMaxH3RegenerateICPipeline" and
                pipe.vae.distributed_executor.parallel_size == 8 and pipe.vae.is_distributed_enabled(),
                "Unexpected native pipeline or VAE topology")
        require(self.native.model_runner.offload_backend is not None and pipe.text_encoder.is_loaded == (rank == 0),
                "Resident text-encoder placement differs")
        require(rank != 0 or pipe.audio_vae is not None, "Missing long-request audio VAE")
        require(all(p.dtype == torch.float32 for p in pipe.vae.parameters()), "VAE must use FP32")
        shard_modules = sum(isinstance(m, FSDPModule) for m in pipe.transformer.modules())
        shards = sum(isinstance(p, DTensor) and p.to_local().numel() < p.numel() for p in pipe.transformer.parameters())
        require(shard_modules and shards, "HSDP sharding evidence absent")
        pipe.transformer.validate_precision()
        pipe._validate_effective_attention()
        verify_fa3(pipe.ic_fa3_evidence(), 0)
        require(callable(pipe.ic_vae_evidence), "Release VAE implementation missing")
        verify_origins()
        props = torch.cuda.get_device_properties(self.device)
        require("H100" in props.name, "Expected H100")
        self.identity = {"rank":rank, "pid":os.getpid(), "worker_id":uuid.uuid4().hex, "gpu_uuid":str(props.uuid)}
        self.startup = {**self.identity, "gpu":props.name, "load_count":1, "import_and_config_seconds":started-initial,
                        "native_load_started_utc":load_utc, "native_load_seconds":loaded-started,
                        "validation_seconds":time.perf_counter()-loaded,
                        "sharded_modules":shard_modules, "sharded_parameters":shards, "vae_dtype":"float32",
                        "world_size":8, "hsdp_size":8, "sequence_parallel_size":8, "vae_size":8,
                        "fa3":pipe.ic_fa3_evidence(), "source":str(SOURCE)}

    def gather(self, local):
        result = [None] * 8
        self.torch.distributed.all_gather_object(result, local, group=self.world.cpu_group)
        return result

    def owner(self, action):
        packet = [None]
        if self.rank == 0:
            try:
                packet[0] = (True, action())
            except Exception as error:
                packet[0] = (False, f"{type(error).__name__}: {error}")
        self.torch.distributed.broadcast_object_list(packet, src=0, group=self.world.cpu_group)
        ok, result = packet[0]
        if not ok:
            raise RuntimeError(result)
        return result

    def generate(self, run_id, row):
        from vllm.config import set_current_vllm_config
        from vllm_omni.diffusion.forward_context import set_forward_context
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        torch, pipe = self.torch, self.pipe
        prompt = {"prompt":read(row["prompt_path"]), "multi_modal_data":{"source_video":row["source"]}}
        sampling = OmniDiffusionSamplingParams(height=row["height"], width=row["width"], num_frames=row["frames"],
            fps=24, seed=row["seed"], num_inference_steps=3, guidance_scale=1., num_outputs_per_prompt=1,
            extra_args={"regenerate_sampler":"up3", "sharp":1., "ic_vae_profile":row["profile"]})
        req = OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id=run_id+"/"+row["id"])
        batch = DiffusionRequestBatch(requests=[req])
        pipe.vae.reset_tile_work_counts()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.distributed.barrier(group=self.world.device_group)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.no_grad(), set_current_vllm_config(self.native.vllm_config), set_forward_context(
                vllm_config=self.native.vllm_config, omni_diffusion_config=self.config):
            result = pipe(batch)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        record = {**self.identity, "pipeline_seconds":elapsed, "stage_seconds":dict(result.stage_durations),
                  "tile_work":pipe.vae.get_tile_work_counts(), "fa3":pipe.ic_fa3_evidence(), "vae":pipe.ic_vae_evidence(),
                  "peak_allocated_bytes":torch.cuda.max_memory_allocated(self.device),
                  "peak_reserved_bytes":torch.cuda.max_memory_reserved(self.device)}
        verify_fa3(record["fa3"], 3)
        verify_vae(record["vae"], row, self.rank, record["tile_work"])
        payload = None
        if self.rank == 0:
            output = result.output
            meta = output["metadata"]["minimax_h3_ic"]
            require(meta["effective_attention_backend"] == meta["attention_backend"] == BACKEND and meta["sampler"] == "up3" and
                    meta["vae_profile"] == row["profile"], "Owner runtime metadata differs from the request")
            require(output["metadata"]["video"] == {"fps":24, "frame_counts":[row["frames"]]}, "Output frame contract differs")
            payload = output["payload"]["video"][0]
            require(isinstance(payload, bytes) and len(payload) > 0, "Missing native MP4")
        else:
            require(result.output is None, "Non-owner retained native payload")
        return record, payload

    def close(self):
        self.native.shutdown()

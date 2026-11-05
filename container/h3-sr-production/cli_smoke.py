"""Actual installed parser verification of both production profiles, no dispatch."""
from __future__ import annotations
from contextlib import contextmanager
import copy
import importlib
import io
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


@contextmanager
def offline():
    def denied(*args, **kwargs):
        raise RuntimeError("Parser verification cannot connect to a network")
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), \
         patch.object(socket, "create_connection", denied):
        yield


def dingo_parser_profiles():
    """Run the actual installed Dingo parser; do not instantiate a worker."""
    args = importlib.import_module("dingo.vllm.omni.args")
    frozen = json.loads((HERE / "gateway-cli-cases.json").read_text(encoding="utf-8"))
    cases = frozen.get("cases", [])
    if (frozen.get("schema") != "h3sr-k8s-generated-parser-cases/1" or len(cases) != 2
            or {case.get("profile") for case in cases} != {"fl2va-sr", "ref2va-sr"}):
        raise ValueError("Both reviewed K8S-generated Dingo argv cases are required")
    results = []
    for case in cases:
        name = case["profile"]
        workflow = name.removesuffix("-sr")
        path = HERE / "profiles" / (name + ".json")
        profile = json.loads(path.read_text(encoding="utf-8"))
        model = profile["h3"]["native_argv"][2]
        # Parse the actual reviewed K8S generator output, never a guessed sample.
        profile_path = "/etc/h3-sr/profile.json"
        argv = ["dingo.vllm.omni", *case["argv"]]
        with patch.object(sys, "argv", argv), patch.object(args, "get_model_path", side_effect=AssertionError("Parser resolved a model")):
            config = args.parse_omni_args()
        if (config.h3_sr_profile != profile_path or config.request_adapter_workflow != workflow
                or config.model != model or config.diffusion.max_num_seqs != 1
                or config.diffusion.record_output_peak_memory is not False
                or config.output_modalities != ["video"] or config.component != "backend"
                or config.namespace != case["expected_namespace"]
                or config.endpoint != case["expected_endpoint"] or config.default_video_fps != 24
                or config.served_model_name != profile["gateway"]["request_model"]
                or config.detached_video_task_root != "/work/gateway/detached"):
            raise ValueError("Actual Dingo parser lost the production contract")
        invalid = list(argv)
        invalid[invalid.index("--request-adapter-workflow") + 1] = "invalid-profile"
        with patch.object(sys, "argv", invalid), patch.object(sys, "stderr", io.StringIO()):
            try:
                args.parse_omni_args()
            except (ValueError, SystemExit):
                pass
            else:
                raise ValueError("Actual Dingo parser accepted an invalid workflow")
        results.append({"profile": name, "argv": argv, "status": "PASS",
                        "invalid_workflow_rejected": True})
    return results


def run():
    from service import profile as profiles
    with offline():
        serve = importlib.import_module("vllm_omni.entrypoints.cli.serve")
        tracking = importlib.import_module("vllm_omni.utils.tracking_parser")
        offload = importlib.import_module("vllm_omni.diffusion.offloader.config")
        utils = importlib.import_module("vllm_omni.entrypoints.utils")
        data = importlib.import_module("vllm_omni.diffusion.data")
        serve._ensure_vllm_platform()  # Upstream CPU parser fallback, no CUDA execution.
        command = next(item for item in serve.cmd_init() if isinstance(item, serve.OmniServeCommand))
        parser = tracking.TrackingArgumentParser(prog="vllm-omni")
        command.subparser_init(parser.add_subparsers(required=True, dest="subparser"))
        results = []
        for name in ("fl2va-sr", "ref2va-sr"):
            frozen = profiles.load(HERE / "profiles" / (name + ".json"))
            for mode in ("off", "stages"):
                value = copy.deepcopy(frozen)
                value["monitoring"]["mode"] = mode
                for role in ("h3", "sr"):
                    argv = profiles.argv(value, role)
                    with patch.object(serve.OmniServeCommand, "cmd", side_effect=AssertionError("Parser dispatched an engine")), \
                         patch.object(serve, "omni_run_server", side_effect=AssertionError("Parser started a server")), \
                         patch.object(serve, "run_omni_api_server_worker_proc", side_effect=AssertionError("Parser spawned a worker")):
                        parsed = parser.parse_args(argv[1:])
                    if (getattr(parsed, "enable_sleep_mode", False) or getattr(parsed, "quantization", None)
                            or getattr(parsed, "max_num_seqs", None) != 1
                            or getattr(parsed, "num_gpus", None) != 8):
                        raise ValueError("Parser changed frozen execution/quantization scope")
                    if (getattr(parsed, "record_output_peak_memory", None) is not False
                            or getattr(parsed, "collect_stage_metrics", None) is not (mode == "stages")
                            or getattr(parsed, "disable_log_stats", None) is not True):
                        raise ValueError("Installed native parser lost default-off observation policy")
                    if role == "h3":
                        spec = getattr(parsed, "diffusion_quantization_config", None)
                        if isinstance(spec, str):
                            spec = json.loads(spec)
                        if spec != value["h3"]["quantization_config"]:
                            raise ValueError("DiT-only quantization parser differs")
                        configured_offload = offload.parse_diffusion_offload_config(getattr(parsed, "diffusion_offload_config", None))
                        if bool(configured_offload) != (name == "ref2va-sr"):
                            raise ValueError("FL/Ref module-offload parser scope differs")
                        if configured_offload and (configured_offload.mode is not offload.OffloadMode.MODULE
                                or configured_offload.components != frozenset({"dit", "text_encoder"})
                                or configured_offload.pin_memory is not True):
                            raise ValueError("Native typed offload changed scope or pinning")
                        cache = getattr(parsed, "cache_config", None)
                        cache = json.loads(cache) if isinstance(cache, str) else cache
                        typed_cache = data.DiffusionCacheConfig.from_dict(cache)
                        if any(getattr(typed_cache, key) != expected for key, expected in value["h3"]["cache_config"].items()):
                            raise ValueError("Native typed Cache-DiT differs from frozen policy")
                    else:
                        overrides = getattr(parsed, "stage_overrides", None)
                        overrides = utils.parse_stage_overrides(overrides)
                        selected = (overrides.get("0", overrides.get(0, {})) if isinstance(overrides, dict) else {})
                        if bool(selected.get("worker_extension_cls")) != (name == "ref2va-sr"):
                            raise ValueError("SR worker extension escaped Ref profile")
                    results.append({"profile": name, "mode": mode, "role": role, "argv": argv, "status": "PASS"})
        with patch.object(sys, "stderr", io.StringIO()):
            try:
                parser.parse_args(argv[1:] + ["--unknown-h3sr-production-option"])
            except SystemExit as error:
                if error.code != 2:
                    raise ValueError("Unexpected unknown-option rejection") from error
            else:
                raise ValueError("Native parser accepted an unknown option")
        dingo = dingo_parser_profiles()
    return {"status": "PASS_ACTUAL_INSTALLED_PARSER_ONLY", "profiles": results, "dingo_profiles": dingo,
            "unknown_options_rejected": True, "dispatch_executed": False,
            "engine_started": False, "model_loaded": False, "gpu_status": "NOT_RUN"}

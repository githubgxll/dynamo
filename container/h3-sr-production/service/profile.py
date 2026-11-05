"""Validated production profile translation; no model or monitoring imports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

EXTENSION = "service.worker_extension.IdleCacheWorkerExtension"
PROFILES = {"fl2va-sr", "ref2va-sr"}
IMAGE_IDENTITY_ENV = {"H3_SR_PRODUCTION_SOURCE_COMMIT", "H3_SR_PRODUCTION_PARENT_IMAGE"}


def validate(profile):
    if not isinstance(profile, dict) or profile.get("schema") != "h3sr-production-target/1" or profile.get("profile") not in PROFILES:
        raise ValueError("Unknown H3/SR production profile")
    ref = profile["profile"] == "ref2va-sr"
    if profile.get("topology", {}).get("gpu_business_concurrency") != 1 or profile["topology"].get("gpu_count") != 8:
        raise ValueError("Production contract requires serial GPU business on eight devices")
    monitoring = profile.get("monitoring", {})
    if monitoring.get("mode") not in ("off", "stages") or monitoring.get("native_output_peak_memory") is not False:
        raise ValueError("Explicit off/stages mode and disabled business peak sampling required")
    for name in ("memory_sampling", "nvml_sampler", "per_pid_sampler", "block_hooks", "offload_timing", "cache_dit_summary"):
        if monitoring.get(name) is not False:
            raise ValueError("Unsupported production monitoring enabled: " + name)
    if profile.get("handoff", {}).get("enabled") is not ref or profile["h3"].get("cpu_offload_enabled") is not ref:
        raise ValueError("FL/Ref offload and handoff scope differ")
    for role in ("h3", "sr"):
        argv = profile[role].get("native_argv")
        if (not isinstance(argv, list) or len(argv) < 3 or argv[:2] != ["vllm-omni", "serve"]
                or any(not isinstance(v, str) for v in argv)):
            raise ValueError("Native backend argv must be explicit strings")
        if "--host" not in argv or argv[argv.index("--host") + 1] != "127.0.0.1":
            raise ValueError("Native backends must remain loopback-only")
    h3 = profile["h3"]["native_argv"]
    if ("--diffusion-offload-config" in h3) is not ref:
        raise ValueError("FL must omit offload; Ref must explicitly select components")
    if not ref and any("offload" in arg for arg in h3):
        raise ValueError("FL must not contain legacy offload switches")
    return profile


def load(path):
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))


def gateway_argv(profile, profile_path, extra):
    """Accept Dingo arguments only; the launcher fixes the Python module."""
    validate(profile)
    if (not isinstance(extra, list) or not extra or extra[0] != "--model"
            or any(not isinstance(value, str) or not value or "\0" in value for value in extra)):
        raise ValueError("Gateway command file must be a Dingo argument list starting with --model")
    required = {"--h3-sr-profile": str(Path(profile_path).resolve()), "--request-adapter": "minimax_h3",
                "--request-adapter-workflow": profile["profile"].removesuffix("-sr"), "--max-num-seqs": "1"}
    for flag in ("--model", *required):
        if extra.count(flag) != 1 or any(value.startswith(flag + "=") for value in extra):
            raise ValueError("Gateway requires exactly one explicit " + flag)
        pos = extra.index(flag) + 1
        if pos >= len(extra) or extra[pos].startswith("-"):
            raise ValueError("Gateway flag is missing its value: " + flag)
        actual = str(Path(extra[pos]).resolve()) if flag == "--h3-sr-profile" else extra[pos]
        if flag in required and actual != required[flag]:
            raise ValueError("Gateway profile/workflow/serial contract differs: " + flag)
    native_only = {"--omni", "--num-gpus", "--tensor-parallel-size", "--ulysses-degree", "--ring-degree",
                   "--stage-overrides", "--diffusion-offload-config", "--enable-cpu-offload"}
    if any(value.split("=", 1)[0] in native_only for value in extra):
        raise ValueError("Gateway command cannot contain native H3/SR GPU argv")
    return list(extra)


def gateway_environment(profile, *, profile_path, epoch, root, inherited=None):
    """Dingo keeps its DYN/connection configuration, without model-stage env."""
    validate(profile)
    env = dict(os.environ if inherited is None else inherited)
    identity = {name: env[name] for name in IMAGE_IDENTITY_ENV if name in env}
    deployed_image = env.get("H3SR_IMAGE_DIGEST")
    if deployed_image is not None and not re.fullmatch(r"[^@\s]+/[^@\s]+@sha256:[0-9a-f]{64}", deployed_image):
        raise ValueError("Gateway H3SR_IMAGE_DIGEST must be the full deployed repo@sha256 digest")
    if profile["monitoring"]["mode"] == "stages" and deployed_image is None:
        raise ValueError("Stages Gateway requires deployed H3SR_IMAGE_DIGEST; no parent-image fallback")
    sanitize = profile["environment_sanitization"]
    prefixes = tuple(sanitize["clear_prefixes"]) + ("H3SR_", "H3_OFFLOAD_")
    names = set(sanitize["clear_names"]) | {"GROUP_RANK", "ROLE_RANK", "LOCAL_WORLD_SIZE", "SEEDVR2_METRICS_DIR"}
    for name in list(env):
        if name in names or name.startswith(prefixes):
            env.pop(name, None)
    env.update(identity)
    if deployed_image is not None:
        env["H3SR_IMAGE_DIGEST"] = deployed_image
    for name in ("PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env[name] = profile["common_environment"].get(name, "1")
    env.update(H3SR_PROFILE_PATH=str(Path(profile_path).resolve()), H3SR_SERVICE_EPOCH=epoch,
               H3SR_CONTROL_ROOT=str(root), PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    return env


def argv(profile, role):
    validate(profile)
    result = list(profile[role]["native_argv"])
    for forbidden in ("--enable-diffusion-pipeline-profiler", "--enable-cache-dit-summary"):
        result = [item for item in result if item != forbidden]
    for flag in ("--collect-stage-metrics", "--no-collect-stage-metrics", "--log-stats", "--no-log-stats", "--disable-log-stats"):
        result = [item for item in result if item != flag]
    result += ["--disable-log-stats", "--collect-stage-metrics" if profile["monitoring"]["mode"] == "stages"
               else "--no-collect-stage-metrics"]
    if role == "h3" and profile["monitoring"]["mode"] == "stages":
        result.append("--enable-diffusion-pipeline-profiler")
    if "--record-output-peak-memory" in result:
        raise ValueError("Production profile cannot enable native business peak sampling")
    if "--no-record-output-peak-memory" not in result:
        result.append("--no-record-output-peak-memory")
    if role == "sr":
        pos = result.index("--stage-overrides") + 1
        value = json.loads(result[pos])
        if set(value) != {"0"}:
            raise ValueError("Single SR stage0 is required")
        if profile["profile"] == "ref2va-sr":
            current = value["0"].get("worker_extension_cls")
            if current not in (None, "h3sr_idle_cache.IdleCacheWorkerExtension", EXTENSION):
                raise ValueError("Another SR worker extension is already selected")
            value["0"]["worker_extension_cls"] = EXTENSION
        elif "worker_extension_cls" in value["0"]:
            raise ValueError("FL cannot register the Ref worker extension")
        result[pos] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return result


def environment(profile, role, *, profile_path, epoch, root, inherited=None):
    validate(profile)
    env = dict(os.environ if inherited is None else inherited)
    image_identity = {name: env[name] for name in IMAGE_IDENTITY_ENV if name in env}
    sanitize = profile["environment_sanitization"]
    prefixes = tuple(sanitize["clear_prefixes"]) + ("H3SR_",)
    names = set(sanitize["clear_names"]) | {"GROUP_RANK", "ROLE_RANK", "LOCAL_WORLD_SIZE", "SEEDVR2_METRICS_DIR"}
    for name in list(env):
        if name in names or name.startswith(prefixes):
            env.pop(name, None)
    env.update(profile["common_environment"])
    env.update(profile[role]["stage_environment"])
    # These two image-baked identities are used by runtime.verify. Broad H3_
    # experiment cleanup must not remove or replace them with profile values.
    env.update(image_identity)
    if role == "sr" or profile["profile"] == "fl2va-sr":
        for name in list(env):
            if name.startswith("H3_OFFLOAD_"):
                env.pop(name)
    env["H3SR_PROFILE_PATH"] = str(Path(profile_path).resolve())
    env["H3SR_SERVICE_EPOCH"] = epoch
    env["H3SR_CONTROL_ROOT"] = str(root)
    # Intentional, narrow module path for spawn workers. No sitecustomize or
    # installed vllm_omni package is provided by this directory.
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    if role == "sr" and profile["monitoring"]["mode"] == "stages":
        env["SEEDVR2_METRICS_DIR"] = profile["monitoring"]["stage_mode_target"]["sr_metrics_dir"]
    for name in ("TMPDIR", "TMP", "TEMP", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "SPEAKER_SAMPLES_DIR"):
        if name in env:
            Path(env[name]).mkdir(parents=True, exist_ok=True)
    return env

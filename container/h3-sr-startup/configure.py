"""Prepare and check local H3/SR startup files. No network or cluster calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


ROOT = Path(__file__).resolve().parent
BASELINE_IMAGE = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:609cfa150ec8ccc81ccf33bc591daf20692e20f94d5ef163c30ca21302e6a4b3"
BASELINE_COMMIT = "2d58b79df39a9335e2f850d3a83bf938610cc611"
IMAGE_PATTERN = r"[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}"
EXTENSION = "service.worker_extension.IdleCacheWorkerExtension"
OFFLOAD = {"mode": "module", "components": ["dit", "text_encoder"], "pin_memory": True}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def flag(argv, name):
    require(isinstance(argv, list) and all(isinstance(x, str) for x in argv), "Explicit argv strings required")
    require(argv.count(name) == 1 and not any(x.startswith(name + "=") for x in argv), "Require exactly one explicit " + name)
    position = argv.index(name) + 1
    require(position < len(argv) and not argv[position].startswith("--"), "Missing value for " + name)
    return argv[position]


def validate_profile(profile, variant):
    ref = variant == "ref2va"
    require(profile.get("schema") == "h3sr-production-target/1" and profile.get("profile") == variant + "-sr", "Profile variant differs")
    require(profile["topology"]["gpu_count"] == 8 and profile["topology"]["gpu_business_concurrency"] == 1, "Eight GPUs and serial GPU business required")
    h3, sr = profile["h3"], profile["sr"]
    argv = h3["native_argv"]
    expected_model = "/models/h3/Ref2VA" if ref else "/models/h3/FL2VA"
    require(argv[:3] == ["vllm-omni", "serve", expected_model], "Native H3 model path differs")
    require(flag(argv, "--dtype") == "bfloat16" and h3.get("quantization_config") is None, "H3 must use BF16 with null quantization")
    require(not any(x.startswith("--") and ("quant" in x.lower() or "fp8" in x.lower()) for x in argv), "H3 quantization flags must be absent")
    indirect = {"--config", "--config-file", "--stage-overrides", "--stage-config", "--stage-configs-path", "--hf-overrides", "--hf-config-path", "--model-loader-extra-config"}
    require(not any(x.split("=", 1)[0] in indirect for x in argv), "Unreviewed indirect H3 model configuration")
    require(flag(argv, "--task-type") == variant, "Native H3 task differs")
    require(h3.get("cpu_offload_enabled") is ref and profile["handoff"]["enabled"] is ref, "FL/Ref offload or handoff differs")
    if ref:
        require(h3.get("offload_config") == OFFLOAD and json.loads(flag(argv, "--diffusion-offload-config")) == OFFLOAD, "Ref requires module offload for dit/text_encoder with pinned memory")
        require(h3["stage_environment"].get("H3_OFFLOAD_FASTPATH") == "1", "Ref requires H3_OFFLOAD_FASTPATH=1")
    else:
        require(h3.get("offload_config") is None and not any("offload" in x.lower() for x in argv), "FL must omit offload")
        require(not any(x.startswith("H3_OFFLOAD_") for x in h3["stage_environment"]), "FL must omit offload environment")
    for environment in (profile["common_environment"], h3["stage_environment"]):
        for name, value in environment.items():
            if "QUANT" in name.upper() or "FP8" in name.upper():
                require(str(value).lower() in {"", "0", "false", "none"}, "Enabled H3 quantization environment: " + name)
    require(flag(sr["native_argv"], "--dtype") == "float16", "SR must retain FP16")
    stage = json.loads(flag(sr["native_argv"], "--stage-overrides"))["0"]
    require(stage.get("worker_extension_cls") == EXTENSION if ref else "worker_extension_cls" not in stage, "FL/Ref SR worker extension differs")
    require(profile["sr_weight_preparation"]["init_required"] is True, "SR weight preparation required")
    require(profile["monitoring"]["mode"] == "off", "This startup bundle defaults to monitoring off")
    for key in ("memory_sampling", "nvml_sampler", "per_pid_sampler", "block_hooks", "offload_timing", "cache_dit_summary"):
        require(profile["monitoring"].get(key) is False, "Optional monitoring enabled: " + key)


def validate_release(profile):
    require(re.fullmatch(IMAGE_PATTERN, profile.get("release_image", "")), "Published repository@sha256 required")
    require(re.fullmatch(IMAGE_PATTERN, profile.get("parent_image", "")), "Actual build parent repository@sha256 required")
    require(re.fullmatch(r"[0-9a-f]{40}", profile.get("source_commit", "")), "Published 40-character source commit required")
    require(profile["release_image"] != BASELINE_IMAGE and profile["source_commit"] != BASELINE_COMMIT, "Baseline 609 image does not contain the new production SR-only branch")


def validate_gateway(argv, profile, variant, mounted_dir):
    require(flag(argv, "--model") == profile["h3"]["native_argv"][2], "Gateway model path differs")
    require(flag(argv, "--served-model-name") == flag(profile["h3"]["native_argv"], "--served-model-name"), "Gateway served model differs")
    require(flag(argv, "--h3-sr-profile") == str(PurePosixPath(mounted_dir) / "profile.json"), "Gateway mounted profile path differs")
    for name, expected in (("--request-adapter", "minimax_h3"), ("--request-adapter-workflow", variant), ("--max-num-seqs", "1"), ("--output-modalities", "video")):
        require(flag(argv, name) == expected, "Gateway contract differs: " + name)
    require(re.fullmatch(r"[a-z][a-z0-9-]{0,62}", flag(argv, "--namespace")), "Set the real Gateway model-pool Dynamo namespace")
    require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", flag(argv, "--endpoint")), "Use a leaf endpoint such as generate")
    require("--component" not in argv, "Dingo component is fixed to backend")


def payload(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def prepare(args):
    snapshots = read_json(ROOT / "BASELINE-SNAPSHOTS.json")
    source = ROOT / "profiles" / (args.variant + "-sr.json")
    expected = next(x["sha256"] for x in snapshots["snapshots"] if x["variant"] == args.variant)
    require(hashlib.sha256(source.read_bytes()).hexdigest() == expected, "Baseline profile snapshot changed")
    profile = read_json(source)
    validate_profile(profile, args.variant)
    profile.update(release_image=args.image, parent_image=args.parent_image, source_commit=args.source_commit,
                   status="CONFIGURED_NEW_RELEASE_RUNTIME_ACCEPTANCE_PENDING")
    profile.pop("release_date", None)
    validate_release(profile)
    argv = read_json(ROOT / ("gateway-command." + args.variant + ".json"))
    argv[argv.index("--namespace") + 1] = args.namespace
    argv[argv.index("--endpoint") + 1] = args.endpoint
    argv[argv.index("--h3-sr-profile") + 1] = str(PurePosixPath(args.mounted_dir) / "profile.json")
    validate_gateway(argv, profile, args.variant, args.mounted_dir)
    files = {"profile.json": payload(profile), "gateway-command.json": payload(argv),
             "launcher-env.json": payload([{"name": "H3SR_IMAGE_DIGEST", "value": args.image}])}
    lock = {"schema": "h3sr-video-gateway-startup-config/1", "variant": args.variant,
            "image": args.image, "parent_image": args.parent_image, "source_commit": args.source_commit,
            "dynamo_namespace": args.namespace, "endpoint_leaf": args.endpoint, "mounted_dir": args.mounted_dir,
            "publication_verification": "RELEASE_OWNER_PROVIDED_VALUES_LOCAL_FORMAT_CHECK_ONLY",
            "baseline_profile_sha256": expected,
            "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    files["configuration-lock.json"] = payload(lock)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = destination / name
        require(not path.is_symlink(), "Output cannot be a symlink: " + str(path))
        if path.exists():
            require(path.read_bytes() == data, "Configuration is frozen; use a new output directory: " + str(path))
    for name, data in files.items():
        path = destination / name
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
    print("STARTUP_CONFIG_READY=" + str(destination.resolve()))


def check(args):
    profile, gateway = read_json(args.profile), read_json(args.gateway_command)
    validate_profile(profile, args.variant)
    validate_release(profile)
    validate_gateway(gateway, profile, args.variant, args.mounted_dir)
    if args.runtime_identity:
        require(os.environ.get("H3SR_IMAGE_DIGEST") == profile["release_image"], "Actual deployed H3SR_IMAGE_DIGEST differs from profile")
        require(os.environ.get("H3_SR_PRODUCTION_SOURCE_COMMIT") == profile["source_commit"], "Image-baked source commit differs from profile")
        require(os.environ.get("H3_SR_PRODUCTION_PARENT_IMAGE") == profile["parent_image"], "Image-baked build parent differs from profile")
    print("STARTUP_CONFIG_OK=" + args.variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    configure = subparsers.add_parser("configure")
    verify = subparsers.add_parser("check")
    for child in (configure, verify):
        child.add_argument("--variant", choices=("fl2va", "ref2va"), required=True)
        child.add_argument("--mounted-dir", default="/etc/h3-sr")
    configure.add_argument("--image", required=True)
    configure.add_argument("--parent-image", required=True)
    configure.add_argument("--source-commit", required=True)
    configure.add_argument("--namespace", required=True)
    configure.add_argument("--endpoint", default="generate")
    configure.add_argument("--output-dir", required=True)
    verify.add_argument("--profile", required=True)
    verify.add_argument("--gateway-command", required=True)
    verify.add_argument("--runtime-identity", action="store_true")
    args = parser.parse_args()
    require(PurePosixPath(args.mounted_dir).is_absolute() and ".." not in PurePosixPath(args.mounted_dir).parts, "Use an absolute Linux configuration mount directory")
    prepare(args) if args.command == "configure" else check(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print("STOP: " + type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)

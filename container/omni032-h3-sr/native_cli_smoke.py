#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parse the frozen real H3/SR argv with the installed native 0.32 parser.

No dispatch/validation method, engine, model configuration or model loader is
called. The official GPUless parser-platform fallback is used only to construct
vLLM's argument parser. This is not a GPU or model-support acceptance gate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
from enum import Enum
import hashlib
import importlib
import importlib.metadata as md
import io
import json
import os
from pathlib import Path
import re
import socket
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "native-cli-argv.json"
PROFILE_NAMES = ("h3-functional", "sr-functional", "h3-core-profiler-parser-probe")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def contract():
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (value.get("schema") != "omni032-native-cli-contract/1"
            or value.get("upstream_version") != "0.32.0"
            or tuple(profile.get("name") for profile in value.get("profiles", [])) != PROFILE_NAMES):
        raise ValueError("Unexpected frozen native CLI contract")
    for profile in value["profiles"]:
        argv = profile.get("argv")
        if (not isinstance(argv, list) or len(argv) < 4 or argv[:2] != ["vllm-omni", "serve"]
                or any(not isinstance(item, str) for item in argv)
                or profile.get("argv_sha256") != canonical_hash(argv)):
            raise ValueError("Frozen native argv changed: " + str(profile.get("name")))
    origin = value["k8s_contract"]
    if origin["snapshot_file"] != "native-cli-k8s-source.json":
        raise ValueError("Unexpected local K8S argv snapshot path")
    snapshot = (HERE / origin["snapshot_file"]).read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != origin["shared_document_sha256"]:
        raise ValueError("Frozen K8S source snapshot changed")
    shared = json.loads(snapshot.decode("utf-8-sig"))
    if (shared["h3_argv"] != value["profiles"][0]["argv"]
            or shared["sr_argv"] != value["profiles"][1]["argv"]):
        raise ValueError("Frozen parser argv does not match the shared K8S source")
    fingerprint = hashlib.sha256(json.dumps({"h3": shared["h3_argv"], "sr": shared["sr_argv"]},
                                           sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if fingerprint != origin["argv_fingerprint_sha256"]:
        raise ValueError("Frozen K8S argv fingerprint changed")
    return value


def normalize(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return normalize(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if dataclasses.is_dataclass(value):
        return normalize(dataclasses.asdict(value))
    if callable(getattr(value, "model_dump", None)):
        return normalize(value.model_dump(mode="json"))
    raise TypeError("Unreviewed CLI value type: " + type(value).__name__)


@contextmanager
def no_network():
    def denied(*args, **kwargs):
        raise RuntimeError("Native CLI parser gate must not connect to networks")
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), \
         patch.object(socket, "create_connection", denied):
        yield


def parsed_values(namespace, fields):
    # Core defaults may contain opaque engine configuration objects unrelated
    # to the frozen argv. Serialize only the fields whose values we certify.
    values = {name: normalize(getattr(namespace, name)) for name in fields if hasattr(namespace, name)}
    if isinstance(values.get("cache_config"), str):
        values["cache_config"] = json.loads(values["cache_config"])
    return values


def verify_profile(profile, namespace, *, stage_parser, offload_module, cache_class):
    fields = set(profile["expected"]) | {"enable_sleep_mode", "quantization"}
    if profile["name"] == "h3-core-profiler-parser-probe":
        fields.add("profiler_config")
    values = parsed_values(namespace, fields)
    failures = {key: {"expected": expected, "actual": values.get(key)}
                for key, expected in profile["expected"].items() if values.get(key) != expected}
    if failures:
        raise ValueError("Native parser changed frozen values: " + json.dumps(failures, sort_keys=True))
    if values.get("enable_sleep_mode") or values.get("quantization"):
        raise ValueError("Parser unexpectedly enabled an unrequested sleep/quantization mode")
    explicit = namespace.get_explicit_kwargs_dict()
    offload = offload_module.parse_diffusion_offload_config(values.get("diffusion_offload_config"))
    if profile["name"].startswith("h3-"):
        if (not isinstance(offload, offload_module.ParsedDiffusionOffloadConfig)
                or offload.mode is not offload_module.OffloadMode.MODULE
                or offload.components != frozenset({"dit", "text_encoder"}) or offload.pin_memory is not True):
            raise ValueError("Native H3 module-offload type/fields changed")
        cache = cache_class.from_dict(values["cache_config"])
        for key, expected in profile["expected"]["cache_config"].items():
            if getattr(cache, key) != expected:
                raise ValueError("Native Cache-DiT configuration changed: " + key)
        typed = {"offload_type": type(offload).__name__, "mode": offload.mode.value,
                 "components": sorted(offload.components), "pin_memory": offload.pin_memory,
                 "cache_type": type(cache).__name__, "cache": profile["expected"]["cache_config"]}
    else:
        stages = stage_parser(values["stage_overrides"])
        if stages != profile["expected"]["stage_overrides"] or offload is not None:
            raise ValueError("Native SeedVR2 stage0/SP8/offload configuration changed")
        typed = {"stage_overrides": stages, "offload": None}
    if profile["name"] == "h3-core-profiler-parser-probe":
        if "profiler_config" not in explicit or "profiler_config" not in values:
            raise ValueError("The core profiler option was not explicitly parsed")
        profiler = values["profiler_config"]
        if not isinstance(profiler, dict) or profiler.get("profiler") is not None:
            raise ValueError("The parser-only core profiler probe must remain disabled")
        typed["core_profiler_probe"] = {"parsed": profiler, "used_in_actual_K8S_argv": False}
    selected = {key: values[key] for key in profile["expected"]}
    return {"name": profile["name"], "role": profile["role"], "status": "PASS",
            "argv": profile["argv"], "argv_sha256": profile["argv_sha256"],
            "parsed_frozen_values": selected, "explicit_keys": sorted(explicit), "typed_config": typed,
            "scope": "Complete parse_args and pure configuration validation; no command dispatch/model/GPU"}


def run(output):
    if os.geteuid() != 1000:
        raise ValueError("Run the native parser gate as final UID1000")
    value = contract()
    source_commit = os.environ.get("OMNI032_H3_SR_SOURCE_COMMIT", "")
    if not re.fullmatch(r"[a-f0-9]{40}", source_commit):
        raise ValueError("Native parser gate requires the full new image source commit")
    if md.version("vllm") != "0.32.0" or md.version("vllm-omni") != "0.32.0":
        raise ValueError("Native parser gate requires the installed formal 0.32 framework pair")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1")
    with no_network():
        serve = importlib.import_module("vllm_omni.entrypoints.cli.serve")
        tracking = importlib.import_module("vllm_omni.utils.tracking_parser")
        offload = importlib.import_module("vllm_omni.diffusion.offloader.config")
        utils = importlib.import_module("vllm_omni.entrypoints.utils")
        data = importlib.import_module("vllm_omni.diffusion.data")
        from vllm import platforms
        before = {"class": type(platforms.current_platform).__name__, "device_type": platforms.current_platform.device_type}
        serve._ensure_vllm_platform()
        after = {"class": type(platforms.current_platform).__name__, "device_type": platforms.current_platform.device_type}
        if not after["device_type"]:
            raise ValueError("Official parser-platform fallback did not resolve a device type")
        root = Path(md.distribution("vllm-omni").locate_file("")).resolve()
        imported = {}
        for module in (serve, tracking, offload, utils, data):
            path = Path(module.__file__).resolve()
            relative = path.relative_to(root).as_posix()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if value["parser_sources_sha256"].get(relative) != actual:
                raise ValueError("Native parser/config import differs from the official wheel: " + relative)
            imported[relative] = actual
        command = next(item for item in serve.cmd_init() if isinstance(item, serve.OmniServeCommand))
        parser = tracking.TrackingArgumentParser(prog="vllm-omni")
        command.subparser_init(parser.add_subparsers(required=True, dest="subparser"))
        reports = []
        with patch.object(serve.OmniServeCommand, "cmd", side_effect=AssertionError("Parser gate dispatched a server")), \
             patch.object(serve, "omni_run_server", side_effect=AssertionError("Parser gate started a model server")), \
             patch.object(serve, "run_omni_api_server_worker_proc", side_effect=AssertionError("Parser gate spawned API workers")):
            for profile in value["profiles"]:
                namespace = parser.parse_args(profile["argv"][1:])
                if not isinstance(namespace, tracking.TrackingNamespace):
                    raise ValueError("Native TrackingArgumentParser did not return a TrackingNamespace")
                reports.append(verify_profile(profile, namespace, stage_parser=utils.parse_stage_overrides,
                    offload_module=offload, cache_class=data.DiffusionCacheConfig))
            # A complete parser must reject unknown flags, not silently retain
            # them for a later engine invocation.
            with patch.object(sys, "stderr", io.StringIO()):
                try:
                    parser.parse_args(value["profiles"][0]["argv"][1:] + ["--unknown-omni032-parser-probe"])
                except SystemExit as error:
                    if error.code != 2:
                        raise ValueError("Unexpected unknown-option rejection") from error
                else:
                    raise ValueError("Native parser accepted an unknown option")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "omni032-native-cli-validation/1", "status": "PASS", "euid": os.geteuid(),
        "source_commit": source_commit, "dispatch_executed": False, "engine_started": False, "model_loaded": False,
        "profiles": reports, "unknown_options_rejected": True, "parser_sources_sha256": imported,
        "argv_contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "k8s_source_sha256": value["k8s_contract"]["shared_document_sha256"],
        "k8s_argv_fingerprint_sha256": value["k8s_contract"]["argv_fingerprint_sha256"],
        "optional_core_profiler_probe": reports[2],
        "platform_before": before, "platform_after": after,
        "platform_scope": "Official _ensure_vllm_platform only for CPU parser construction; not hardware acceptance",
        "not_run": ["command dispatch", "model config resolution", "model loading", "server startup", "network", "CUDA/H100", "media inference"],
        "scope": "Actual native installed parse_args for frozen complete K8S argv; pure offload/cache/stage types only"}
    (output / "native-cli.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()

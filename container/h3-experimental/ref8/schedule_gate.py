"""Offline, non-root CPU check of the installed Ref8 sigma calculation.

Independent helper called by check_contract.py --installed; Linux non-root only.
Run before publication with the image, not on the Windows host:
  docker run --rm -i --network none --entrypoint python3 IMAGE -B - < schedule_gate.py
The script uses real CPU torch and hash-pinned function AST, without importing
the GPU pipeline or loading model weights. It does not establish GPU quality.
"""
from __future__ import annotations

import ast
from collections.abc import Sequence
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


EXPECTED = {
    "time_request.py": "c20569799afd74c984e6a323a12c0a587dc6c70d84d375770c81cdccb503de9e",
    "pipeline_minimax_h3.py": "e8d37d9e086b72930d33b419efadbf25ff695bcef295cba8f3c0c2b59edc85e0",
    "lora.py": "4628f4b90b47aa824bfae141ea5c3c17713ea13f73c88135215afac556d774cf",
}
ARTIFACT = "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def function(tree, name):
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(matches) == 1, "Native function absent or ambiguous: " + name)
    return matches[0]


def same_expr(node, expression):
    return ast.dump(node, include_attributes=False) == ast.dump(
        ast.parse(expression, mode="eval").body, include_attributes=False)


def check_static_path(pipeline_bytes):
    """Check the actual request-to-sigma expressions; do not synthesize a path."""
    tree = ast.parse(pipeline_bytes)
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.level == 1 and node.module == "time_request"
               and any(alias.name == "minimax_h3_time_shift_sigmas" and alias.asname is None
                       for alias in node.names)]
    require(len(imports) == 1, "Pipeline sigma helper no longer comes from time_request")
    request = function(tree, "_prepare_encoder_conditioning_inputs")
    returned = [node.value for node in ast.walk(request)
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)]
    require(len(returned) == 1, "Expected a single request-conditioning dictionary")
    fields = {key.value: value for key, value in zip(returned[0].keys, returned[0].values)
              if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    for key, expression in {
        "video_shift": 'float(extra.get("flow_shift", self.default_video_shift))',
        "audio_shift": 'float(extra.get("audio_flow_shift", self.default_audio_shift))',
        "num_steps": "num_steps",
        "base_schedule": "base_schedule",
    }.items():
        require(key in fields and same_expr(fields[key], expression),
                "Request field no longer forwards the expected expression: " + key)
    diffuse = function(tree, "diffuse")
    calls = [node for node in ast.walk(diffuse) if isinstance(node, ast.Call)
             and same_expr(node.func, "self._build_denoise_inputs")]
    require(len(calls) == 1, "Expected one diffuse-to-denoise-input call")
    arguments = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    for name in ("video_shift", "audio_shift", "num_steps", "base_schedule"):
        require(name in arguments and same_expr(arguments[name], name),
                "Denoise input no longer forwards: " + name)
    denoise_inputs = function(tree, "_build_denoise_inputs")
    for destination, shift in (("video_sigmas", "video_shift"), ("audio_sigmas", "audio_shift")):
        # 0.31 has a nested optional refine branch that truncates an already
        # built ladder. This gate checks the first-pass (no-refine) assignment.
        # The entire pipeline hash is fixed independently by the source lock.
        assigns = [node for node in denoise_inputs.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == destination
                           for target in node.targets)]
        require(len(assigns) == 1 and isinstance(assigns[0].value, ast.Call),
                "Expected one native sigma assignment: " + destination)
        call = assigns[0].value
        require(same_expr(call.func, "minimax_h3_time_shift_sigmas"), "Sigma helper changed")
        arguments = {keyword.arg: keyword.value for keyword in call.keywords}
        for key, value in (("num_steps", "num_steps"), ("shift_scale", shift),
                           ("base_schedule", "base_schedule")):
            require(key in arguments and same_expr(arguments[key], value),
                    "Sigma call no longer forwards " + destination + ":" + key)
    return {"request_fields": ["flow_shift", "audio_flow_shift"],
            "request_function": request.name, "denoise_function": denoise_inputs.name,
            "sigma_helper": "minimax_h3_time_shift_sigmas", "status": "PASS_STATIC_PATH_ONLY"}


def native_sigma_function(time_bytes):
    tree = ast.parse(time_bytes)
    for name, callee in (("minimax_h3_time_shift_sigmas", "MINIMAX_H3_SHAPE_PLANNER.time_shift_sigmas"),
                         ("time_shift_sigmas", "_time_shift_sigmas")):
        wrapper = function(tree, name)
        require(len(wrapper.body) == 1 and isinstance(wrapper.body[0], ast.Return)
                and isinstance(wrapper.body[0].value, ast.Call), "Unexpected sigma wrapper: " + name)
        call = wrapper.body[0].value
        require(same_expr(call.func, callee), "Sigma wrapper target changed: " + name)
        arguments = {keyword.arg: keyword.value for keyword in call.keywords}
        require(set(arguments) == {"num_steps", "shift_scale", "base_schedule"},
                "Sigma wrapper argument set changed: " + name)
        require(all(same_expr(value, key) for key, value in arguments.items()),
                "Sigma wrapper no longer forwards its arguments: " + name)
    node = function(tree, "_time_shift_sigmas")
    # The tested uniform path has base_schedule=None. The untouched optional
    # DMD2 branch is not invoked, and no stand-in for torch is installed.
    namespace = {"Sequence": Sequence}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "installed-time_request.py", "exec"), namespace)
    return namespace[node.name]


def reference_sigmas(steps, shift):
    return [shift * q / (1 + (shift - 1) * q)
            for q in ((steps - index) / steps for index in range(steps + 1))]


def check_numbers(sigma):
    schedules = {}
    for label, steps, shift in (("ref8_video12", 8, 12), ("legacy_video6", 8, 6),
                                ("ref8_audio3", 8, 3), ("fl4_video6", 4, 6)):
        actual = sigma(num_steps=steps, shift_scale=shift, base_schedule=None)
        require(len(actual) == steps + 1, label + ": wrong boundary count")
        require(actual[0] == 1.0 and actual[-1] == 0.0, label + ": wrong endpoints")
        require(all(math.isfinite(value) for value in actual), label + ": nonfinite sigma")
        require(all(a > b for a, b in zip(actual, actual[1:])), label + ": not strictly descending")
        expected = reference_sigmas(steps, shift)
        require(all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7) for a, b in zip(actual, expected)),
                label + ": disagrees with independent author-grid calculation")
        schedules[label] = actual
    require(all(a > b for a, b in zip(schedules["ref8_video12"][1:-1],
                                     schedules["legacy_video6"][1:-1])),
            "Video shift 12 did not change every interior sigma relative to 6")
    repeated_audio = sigma(num_steps=8, shift_scale=3, base_schedule=None)
    require(repeated_audio == schedules["ref8_audio3"], "Audio schedule changed across video-shift calculations")
    return schedules


def run():
    require(sys.platform == "linux" and os.geteuid() != 0,
            "Run this gate in the Linux derived image as its default non-root user")
    require(importlib.metadata.version("vllm-omni") == "0.31.0rc1", "Unexpected vllm-omni distribution")
    spec = importlib.util.find_spec("vllm_omni")
    require(spec is not None and spec.submodule_search_locations is not None,
            "Cannot locate installed vllm_omni package")
    locations = list(spec.submodule_search_locations)
    require(len(locations) == 1, "Ambiguous installed vllm_omni root")
    root = Path(locations[0]).resolve() / "diffusion/models/minimax_h3"
    sources, hashes = {}, {}
    for name, expected in EXPECTED.items():
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
                "Unsafe source path: " + name)
        raw = path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        require(observed == expected, "Installed source hash mismatch: " + name)
        sources[name], hashes[name] = raw, observed
    path_report = check_static_path(sources["pipeline_minimax_h3.py"])
    import torch  # Real installed torch; all native sigma operations use device='cpu'.
    require(not torch.cuda.is_initialized(), "CUDA was initialized before the CPU-only check")
    schedules = check_numbers(native_sigma_function(sources["time_request.py"]))
    require(not torch.cuda.is_initialized(), "CPU schedule gate unexpectedly initialized CUDA")
    return {"schema": "h3-ref8-schedule-check/1", "status": "PASS_SCHEDULE_CPU_ONLY", "errors": [],
            "artifact": ARTIFACT, "uid": os.geteuid(), "torch_version": str(torch.__version__),
            "source_root": str(root), "source_sha256": hashes, "request_path": path_report,
            "native_cpu_torch_executed": True, "schedule_boundary_count": 9,
            "denoiser_intervals": 8, "schedules": schedules, "gpu_validated": False,
            "limits": ["Uniform sigma helper and static request path only; no full pipeline or weights",
                       "No HTTP, actual adapter activation, HSDP, media generation or quality acceptance"]}


def main():
    try:
        report = run()
    except Exception as exc:
        report = {"schema": "h3-ref8-schedule-check/1", "status": "FAIL", "gpu_validated": False,
                  "errors": [type(exc).__name__ + ": " + str(exc)]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS_SCHEDULE_CPU_ONLY" else 1


if __name__ == "__main__":
    sys.exit(main())

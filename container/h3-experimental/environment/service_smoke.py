"""Import the packaged Gateway/Omni integration and display help; never start services."""
from __future__ import annotations
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys

IMPORTS = (
    "dynamo._core", "dynamo.runtime", "dingo.video_gateway.app",
    "dingo.video_gateway.telemetry", "dingo.video_gateway.adapters.minimax_h3",
    "dingo.vllm.omni.args", "dingo.vllm.omni.engine_monitor",
    "dingo.vllm.omni.output_formatter", "dingo.vllm.omni.request_adapters.minimax_h3",
    "dingo.vllm.omni.omni_handler", "dingo.vllm.omni.stage_worker",
    "dingo.vllm.omni.stage_router",
    "opentelemetry.sdk.trace",
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
    "opentelemetry.exporter.prometheus",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(os.environ.get("H3_BUILD_INFO", "/opt/dynamo/build-info")))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "DO_NOT_TRACK": "1", "VLLM_NO_USAGE_STATS": "1"}
    # Outbound sockets are forbidden even in the framework build layer whose
    # dependency installer legitimately had network access earlier.
    guard = """
import socket
def deny(*args, **kwargs):
    raise RuntimeError('Service import/help smoke must not connect to networks')
socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.create_connection = deny
"""
    checks = [
        ("service-imports", [sys.executable, "-B", "-c", guard +
            "\nimport importlib,json\nmodules=" + repr(IMPORTS) +
            "\nfor name in modules: importlib.import_module(name)\nprint(json.dumps({'imports':modules}))"]),
        ("gateway-cli-help", [sys.executable, "-B", "-c", guard +
            "\nimport runpy,sys\nsys.argv=['dingo.video_gateway','--help']\nrunpy.run_module('dingo.video_gateway',run_name='__main__')"]),
    ]
    results = []
    for name, command in checks:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env, timeout=180, check=False)
        (args.output_dir / f"{name}.txt").write_text(result.stdout, encoding="utf-8")
        results.append({"name": name, "exit_code": result.returncode})
        if result.returncode:
            print(result.stdout, flush=True)
            raise RuntimeError(f"{name} failed before starting any service")
        if name == "gateway-cli-help" and "--config" not in result.stdout:
            raise RuntimeError("Gateway --help did not expose the expected configuration option")
    report = {"status": "PASS_SERVICE_IMPORTS_AND_HELP_ONLY", "checks": results,
              "packages": {name: md.version(name) for name in (
                  "ai-dingo", "ai-dingo-runtime", "vllm", "vllm-omni",
                  "opentelemetry-api", "opentelemetry-sdk", "opentelemetry-exporter-otlp",
                  "opentelemetry-exporter-prometheus")},
              "scope": "Real installed imports and Gateway help with outbound socket calls denied; no NATS/ETCD/Gateway server/Omni engine/GPU/model started"}
    (args.output_dir / "service-smoke.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

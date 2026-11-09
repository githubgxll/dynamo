#!/usr/bin/env python3
"""Native Omni backend entry with only the Ref SR internal control extension."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "service"

from . import profile as profiles


def verify_service_package():
    import service
    if Path(service.__file__).resolve() != Path(__file__).resolve().with_name("__init__.py"):
        raise RuntimeError("Production service package import is shadowed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--role", choices=("h3", "sr"), required=True)
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--import-report", type=Path)
    args = parser.parse_args()
    if args.check_imports != (args.import_report is not None):
        parser.error("--check-imports and --import-report must be supplied together")
    verify_service_package()
    profile = profiles.load(args.profile)
    argv = profiles.argv(profile, args.role)
    if not args.check_imports and args.role == "sr" and profile["handoff"]["enabled"]:
        from .control import ControlGate, install
        from .worker_extension import source_sha256
        from vllm_omni.diffusion.models.seedvr2 import long_video
        from vllm_omni.entrypoints.openai import api_server
        root = Path(os.environ["H3SR_CONTROL_ROOT"])
        token = (root / "control-token").read_text(encoding="ascii").strip()
        original = api_server.build_openai_app

        def build(*a, **kw):
            app = original(*a, **kw)
            gate = ControlGate(root=root, epoch=os.environ["H3SR_SERVICE_EPOCH"], token=token,
                long_video=long_video, engine_provider=lambda: getattr(app.state, "diffusion_engine", None),
                extension_sha256=source_sha256())
            return install(app, gate)

        api_server.build_openai_app = build
    from vllm_omni.entrypoints.cli.main import main as native_main
    if args.check_imports:
        import cProfile
        import profile as stdlib_profile
        import sysconfig
        import torch._dynamo

        expected = Path(sysconfig.get_path("stdlib")) / "profile.py"
        service_dir = Path(__file__).resolve().parent
        if (Path(stdlib_profile.__file__).resolve() != expected.resolve()
                or not hasattr(stdlib_profile, "run")
                or any(Path(entry).resolve() == service_dir for entry in sys.path)):
            raise RuntimeError("Backend entrypoint shadows the standard-library profile module")
        args.import_report.write_text(json.dumps({
            "status": "PASS_ACTUAL_BACKEND_IMPORTS_ONLY", "role": args.role,
            "profile": profile["profile"], "stdlib_profile": str(expected.resolve()),
            "service_profile": str(Path(profiles.__file__).resolve()),
            "safe_path": sys.flags.safe_path,
            "native_cli": str(Path(importlib.import_module(native_main.__module__).__file__).resolve()),
            "torch_dynamo": str(Path(torch._dynamo.__file__).resolve()),
            "sys_path": list(sys.path), "model_loaded": False, "gpu_status": "NOT_RUN",
        }, indent=2) + "\n", encoding="utf-8")
        return
    sys.argv = argv
    native_main()


if __name__ == "__main__":
    main()

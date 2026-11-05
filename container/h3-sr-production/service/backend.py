#!/usr/bin/env python3
"""Native Omni backend entry with only the Ref SR internal control extension."""
from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    verify_service_package()
    profile = profiles.load(args.profile)
    argv = profiles.argv(profile, args.role)
    if args.role == "sr" and profile["handoff"]["enabled"]:
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
    sys.argv = argv
    native_main()


if __name__ == "__main__":
    main()

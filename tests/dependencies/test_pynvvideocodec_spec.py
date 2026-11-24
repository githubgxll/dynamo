# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare every runtime codec guard with the version its requirements install."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check_guards(framework: str, dockerfile: Path) -> None:
    requirements = ROOT / f"container/deps/requirements.{framework}.txt"
    pins = re.findall(
        r"(?im)^PyNvVideoCodec==([\w.]+)[ \t]*(?:#.*)?$", requirements.read_text()
    )
    if len(pins) != 1:
        raise ValueError(f"{requirements}: expected one exact PyNvVideoCodec pin")

    guards = [
        (number, line.split())
        for number, line in enumerate(dockerfile.read_text().splitlines(), 1)
        if "compliance.check_pynvvideocodec" in line
        and not line.lstrip().startswith("#")
    ]
    if not guards:
        raise ValueError(f"{dockerfile}: missing PyNvVideoCodec guard")
    for number, args in guards:
        if args[-2:] != ["--pinned", pins[0]]:
            raise ValueError(
                f"{dockerfile}:{number}: expected --pinned {pins[0]}, got {args[-2:]}"
            )


def test_runtime_codec_guards() -> None:
    if not (ROOT / "container/templates/vllm_runtime.Dockerfile").exists():
        import pytest

        pytest.skip("runtime templates are excluded from component images")
    for framework in ("vllm", "sglang"):
        template = ROOT / f"container/templates/{framework}_runtime.Dockerfile"
        check_guards(framework, template)


if __name__ == "__main__":
    framework, dockerfile = sys.argv[1:]
    check_guards(framework, Path(dockerfile))

# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github/scripts/prepare_dingo_image_matrix.py"
SPEC = importlib.util.spec_from_file_location(
    "prepare_dingo_image_matrix", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
prepare_dingo_image_matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_dingo_image_matrix)


def load_repository_inputs() -> tuple[dict, dict]:
    config = prepare_dingo_image_matrix.load_config(
        REPO_ROOT / ".github/dingo-images.json"
    )
    with (REPO_ROOT / "container/context.yaml").open(encoding="utf-8") as context_file:
        context = yaml.safe_load(context_file)
    return config, context


def test_framework_tags_use_context_runtime_image_tags() -> None:
    config, context = load_repository_inputs()
    matrix = prepare_dingo_image_matrix.build_matrix(
        config, "a" * 40, "all"
    )
    images = {entry["framework"]: entry["image"] for entry in matrix}

    for framework in ("vllm", "sglang"):
        runtime_tag = context[framework]["cuda13.0"]["runtime_image_tag"]
        assert images[framework].endswith(
            f":{runtime_tag.split('@', 1)[0]}-aaaaaaaaaaaa"
        )

    dynamo_targets = {
        entry["docker_target"] for entry in matrix if entry["framework"] == "dynamo"
    }
    assert dynamo_targets == {"router", "runtime"}


def test_runtime_tag_digest_is_not_part_of_output_tag() -> None:
    value = "v0.32.0-ubuntu2404@sha256:" + "a" * 64
    assert prepare_dingo_image_matrix.runtime_tag_prefix(value) == "v0.32.0-ubuntu2404"
    assert prepare_dingo_image_matrix.runtime_tag_prefix("v0.32.0") == "v0.32.0"


@pytest.mark.parametrize("suffix", ["sha256:abc", "", "sha512:" + "a" * 64, "sha256:" + "a" * 64 + "@extra"])
def test_invalid_runtime_digest_pin_is_rejected(suffix: str) -> None:
    with pytest.raises(ValueError, match="invalid digest pin"):
        prepare_dingo_image_matrix.runtime_tag_prefix("v0.32.0@" + suffix)


if __name__ == "__main__":
    test_framework_tags_use_context_runtime_image_tags()

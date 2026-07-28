#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Dingo image configuration and emit a GitHub Actions matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SUPPORTED_FRAMEWORKS = {"dynamo", "vllm", "sglang"}
SUPPORTED_SELECTIONS = {"configured", "dynamo", "vllm", "sglang", "all"}
SUPPORTED_PLATFORMS = {"linux/amd64"}
SUPPORTED_CUDA_VERSIONS = {"13.0"}
SUPPORTED_DOCKER_TARGETS = {
    "dynamo": {"runtime"},
    "vllm": {"runtime", "pre_runtime"},
    "sglang": {"runtime", "pre_runtime"},
}

REGISTRY_PATTERN = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)(?::[0-9]{1,5})?$"
)
REPOSITORY_PATH_PATTERN = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*$"
)
TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--github-sha", required=True)
    parser.add_argument(
        "--selection", choices=sorted(SUPPORTED_SELECTIONS), default="configured"
    )
    parser.add_argument("--github-output", type=Path, required=True)
    return parser.parse_args()


def require_string(config: dict, key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("top-level configuration must be an object")
    return config


def select_image(enabled: bool, framework: str, selection: str) -> bool:
    if selection == "configured":
        return enabled
    if selection == "all":
        return True
    return framework == selection


def build_matrix(config: dict, github_sha: str, selection: str) -> list[dict[str, str]]:
    if not GIT_SHA_PATTERN.fullmatch(github_sha):
        raise ValueError("--github-sha must be a full 40-character Git commit SHA")

    registry = require_string(config, "registry").rstrip("/")
    if "://" in registry or not REGISTRY_PATTERN.fullmatch(registry):
        raise ValueError("registry must be a host[:port] without a URL scheme or path")

    namespace = require_string(config, "namespace").strip("/")
    if not REPOSITORY_PATH_PATTERN.fullmatch(namespace):
        raise ValueError("namespace is not a valid lowercase container repository path")

    platform = require_string(config, "platform")
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"platform must be one of {sorted(SUPPORTED_PLATFORMS)}")

    cuda_version = require_string(config, "cuda_version")
    if cuda_version not in SUPPORTED_CUDA_VERSIONS:
        raise ValueError(
            f"cuda_version must be one of {sorted(SUPPORTED_CUDA_VERSIONS)}"
        )

    sha_length = config.get("commit_sha_length", 12)
    if not isinstance(sha_length, int) or not 7 <= sha_length <= 40:
        raise ValueError("commit_sha_length must be an integer between 7 and 40")
    short_sha = github_sha[:sha_length].lower()

    keep_buildkit_state = config.get("keep_buildkit_state", False)
    if not isinstance(keep_buildkit_state, bool):
        raise ValueError("keep_buildkit_state must be true or false")

    images = config.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("images must be a non-empty array")

    matrix: list[dict[str, str]] = []
    seen_frameworks: set[str] = set()
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError(f"images[{index}] must be an object")

        framework = require_string(image, "framework")
        if framework not in SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"images[{index}].framework must be one of "
                f"{sorted(SUPPORTED_FRAMEWORKS)}"
            )
        if framework in seen_frameworks:
            raise ValueError(f"duplicate framework configuration: {framework}")
        seen_frameworks.add(framework)

        enabled = image.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError(f"images[{index}].enabled must be true or false")
        if not select_image(enabled, framework, selection):
            continue

        repository = require_string(image, "repository")
        if not REPOSITORY_PATH_PATTERN.fullmatch(repository):
            raise ValueError(
                f"images[{index}].repository is not a valid lowercase repository name"
            )

        tag_prefix = require_string(image, "tag_prefix")
        tag = f"{tag_prefix}-{short_sha}"
        if not TAG_PATTERN.fullmatch(tag):
            raise ValueError(
                f"generated tag for {framework} is invalid or longer than 128 characters"
            )

        docker_target = image.get("docker_target", "runtime")
        if (
            not isinstance(docker_target, str)
            or docker_target not in SUPPORTED_DOCKER_TARGETS[framework]
        ):
            raise ValueError(
                f"images[{index}].docker_target must be one of "
                f"{sorted(SUPPORTED_DOCKER_TARGETS[framework])} for {framework}"
            )

        architecture = platform.rsplit("/", 1)[-1]
        image_repository = f"{registry}/{namespace}/{repository}"
        matrix.append(
            {
                "framework": framework,
                "registry": registry,
                "platform": platform,
                "cuda_version": cuda_version,
                "keep_buildkit_state": (
                    "true" if keep_buildkit_state else "false"
                ),
                "dockerfile": (
                    f"container/{framework}-runtime-cuda{cuda_version}-"
                    f"{architecture}-rendered.Dockerfile"
                ),
                "docker_target": docker_target,
                "image": f"{image_repository}:{tag}",
            }
        )

    return matrix


def write_github_output(path: Path, matrix: list[dict[str, str]]) -> None:
    compact_matrix = json.dumps({"include": matrix}, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as output:
        output.write(f"has_images={'true' if matrix else 'false'}\n")
        output.write(f"matrix={compact_matrix}\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    matrix = build_matrix(config, args.github_sha, args.selection)
    write_github_output(args.github_output, matrix)

    if matrix:
        print("Images selected for build:")
        for entry in matrix:
            print(f"  - {entry['image']}")
    else:
        print("No images are enabled for this workflow run.")


if __name__ == "__main__":
    main()

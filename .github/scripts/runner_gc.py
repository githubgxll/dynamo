#!/usr/bin/env python3
"""Retain the newest five local Dingo runtime images; dry-run by default."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(*args: str) -> str:
    print(f"$ {shlex.join(args)}", flush=True)
    result = subprocess.run(
        args, check=False, text=True, capture_output=True, timeout=120
    )
    if result.stderr:
        print(result.stderr, end="", flush=True)
    result.check_returncode()
    return result.stdout


def managed_tag(tag: str, repositories: set[str]) -> bool:
    repository, separator, name = tag.rpartition(":")
    return bool(
        separator
        and repository in repositories
        and not name.startswith(("builder-", "buildcache-"))
        and re.fullmatch(r".+-[0-9a-f]{7,40}", name)
    )


def creation_key(image: dict[str, Any]) -> tuple[datetime, int, str]:
    # Docker emits RFC3339Nano; Python 3.10 fromisoformat cannot parse nine
    # fractional digits. Preserve those digits separately for exact ordering.
    match = re.fullmatch(
        r"(.+T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})",
        image["Created"],
    )
    if match is None:
        raise ValueError(f"invalid image creation time: {image['Created']}")
    seconds = datetime.fromisoformat(match[1] + match[3].replace("Z", "+00:00"))
    nanoseconds = int((match[2] or "").ljust(9, "0"))
    return seconds, nanoseconds, image["Id"]


def plan_images(
    images: list[dict[str, Any]], repositories: set[str], keep: int
) -> dict[str, list[dict[str, Any]]]:
    if keep < 1:
        raise ValueError("keep must be positive")
    managed, protected = [], []
    # One image with multiple tags counts once.
    for image in {item["Id"]: item for item in images}.values():
        tags = image.get("RepoTags") or []
        if tags and all(managed_tag(tag, repositories) for tag in tags):
            managed.append(image)
        else:
            protected.append(image)
    managed.sort(key=creation_key, reverse=True)
    return {
        "keep": managed[:keep],
        "candidates": managed[keep:],
        "protected": protected,
    }


def recovery_reference(tag: str, image: dict[str, Any]) -> str:
    """Verify the remote manifest config matches this exact local image ID."""
    descriptor = json.loads(
        run(
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            tag,
            "--format",
            "{{json .Manifest}}",
        )
    )
    digest = descriptor["digest"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("invalid remote manifest digest")
    repository = tag.rsplit(":", 1)[0]
    reference = f"{repository}@{digest}"
    manifest = json.loads(
        run("docker", "buildx", "imagetools", "inspect", reference, "--raw")
    )
    if "manifests" in manifest:
        matches = [
            item
            for item in manifest["manifests"]
            if item.get("platform", {}).get("os") == image["Os"]
            and item.get("platform", {}).get("architecture") == image["Architecture"]
            and item.get("platform", {}).get("variant", "") == image.get("Variant", "")
        ]
        if len(matches) != 1:
            raise ValueError("remote platform is ambiguous or missing")
        child = f"{repository}@{matches[0]['digest']}"
        manifest = json.loads(
            run("docker", "buildx", "imagetools", "inspect", child, "--raw")
        )
    if manifest.get("config", {}).get("digest") != image["Id"]:
        raise ValueError("remote tag does not contain the local image")
    return reference


def verify_image(image: dict[str, Any]) -> list[dict[str, str]]:
    # Validate every tag before removing any. Docker refuses in-use image removal;
    # never force removal. The runner must be exclusive during this operation.
    references = [
        {"tag": tag, "digest_reference": recovery_reference(tag, image)}
        for tag in image["RepoTags"]
    ]
    return references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = run("docker", "info", "--format", "{{.DockerRootDir}}").strip()
    if root != "/runner-data/docker":
        raise RuntimeError(f"unexpected Docker data root: {root}")
    config = json.loads(args.config.read_text())
    repositories = {
        f"{config['registry']}/{config['namespace']}/{item['repository']}"
        for item in config["images"]
    }
    ids = sorted(set(run("docker", "image", "ls", "-aq", "--no-trunc").split()))
    images = json.loads(run("docker", "image", "inspect", *ids)) if ids else []
    plan = plan_images(images, repositories, args.keep)
    (args.output / "inventory.json").write_text(json.dumps(images, indent=2) + "\n")
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    failures = []
    deletion_failed = False
    with (args.output / "recovery.jsonl").open("a") as recovery:
        for image in plan["candidates"]:
            verified = False
            print(
                f"CANDIDATE {image['Created']} {image['Id']} {image['RepoTags']}",
                flush=True,
            )
            try:
                references = verify_image(image)
                recovery.write(
                    json.dumps({"id": image["Id"], "references": references}) + "\n"
                )
                recovery.flush()
                verified = True
                if args.apply:
                    for reference in references:
                        tag = reference["tag"]
                        current = json.loads(run("docker", "image", "inspect", tag))[0]
                        if current["Id"] != image["Id"]:
                            raise ValueError("local tag changed during GC")
                        print(
                            run("docker", "image", "rm", "--no-prune", tag), flush=True
                        )
                else:
                    print("DRY RUN: verified; no deletion", flush=True)
            except (subprocess.SubprocessError, ValueError, KeyError) as error:
                if args.apply and verified:
                    deletion_failed = True
                failures.append(image["Id"])
                print(f"SKIP/FAILED {image['Id']}: {error}", flush=True)
    summary = (
        f"# Runner GC — {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Mode: {'apply' if args.apply else 'dry-run'}\n\n"
        f"Keep newest: {len(plan['keep'])}; candidates: {len(plan['candidates'])}; "
        f"protected: {len(plan['protected'])}; skipped/failed: {len(failures)}.\n\n"
        "Order: image Created descending, globally across configured Dingo repositories.\n"
        "Builder/base, unknown tags, and untagged images are excluded.\n"
    )
    (args.output / "summary.md").write_text(summary)
    print(summary)
    if failures:
        print("Some images could not be safely removed; inspect GC artifacts")
        raise SystemExit(1 if deletion_failed else 2)


if __name__ == "__main__":
    main()

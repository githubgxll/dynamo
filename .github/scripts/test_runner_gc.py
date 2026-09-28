from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "runner_gc", Path(__file__).with_name("runner_gc.py")
)
assert SPEC is not None and SPEC.loader is not None
gc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gc)
REPO = "registry.example:8443/team/runtime"


def image(number: int, tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "Id": f"sha256:{number:064x}",
        "Created": f"2026-09-{number:02d}T00:00:00Z",
        "RepoTags": tags if tags is not None else [f"{REPO}:runtime-{number:012x}"],
        "Os": "linux",
        "Architecture": "amd64",
    }


def test_keep_newest_five_unique_ids_across_repositories() -> None:
    other = "registry.example:8443/team/other"
    images = [image(n) for n in [2, 7, 1, 4, 6, 3, 5]]
    images[1]["RepoTags"] = [f"{other}:runtime-abcdef123456"]
    images[4]["RepoTags"].append(f"{REPO}:runtime-aaaaaaaaaaaa")
    plan = gc.plan_images(images + [images[1]], {REPO, other}, 5)
    assert [item["Id"] for item in plan["keep"]] == [
        image(n)["Id"] for n in [7, 6, 5, 4, 3]
    ]
    assert [item["Id"] for item in plan["candidates"]] == [
        image(2)["Id"],
        image(1)["Id"],
    ]


def test_protect_builder_unknown_alias_and_untagged() -> None:
    images = [
        image(1, [f"{REPO}:builder-cu130-amd64-aaaaaaaaaaaa"]),
        image(2, [f"{REPO}:runtime-aaaaaaaaaaaa", f"{REPO}:latest"]),
        image(3, []),
        image(4, ["ubuntu:22.04"]),
    ]
    plan = gc.plan_images(images, {REPO}, 5)
    assert plan["protected"] == images
    assert plan["candidates"] == []


def test_keep_fewer_than_five_and_reject_zero() -> None:
    assert gc.plan_images([image(1)], {REPO}, 5)["candidates"] == []
    with pytest.raises(ValueError):
        gc.plan_images([image(1)], {REPO}, 0)


def test_nanosecond_creation_order_and_timezones() -> None:
    images = [image(n) for n in range(1, 4)]
    images[0]["Created"] = "2026-09-28T08:00:00.000000002Z"
    images[1]["Created"] = "2026-09-28T08:00:00.000000001Z"
    images[2]["Created"] = "2026-09-28T15:59:59.999999999+08:00"
    plan = gc.plan_images(images, {REPO}, 1)
    assert plan["keep"] == images[:1]
    assert plan["candidates"] == images[1:]


@pytest.mark.parametrize("matches", [True, False])
def test_verify_remote_config_digest(
    monkeypatch: pytest.MonkeyPatch, matches: bool
) -> None:
    local = image(1)
    digest = "sha256:" + "a" * 64
    outputs = iter(
        [
            json.dumps({"digest": digest}),
            json.dumps(
                {"config": {"digest": local["Id"] if matches else image(2)["Id"]}}
            ),
        ]
    )
    monkeypatch.setattr(gc, "run", lambda *args: next(outputs))
    if matches:
        assert gc.recovery_reference(local["RepoTags"][0], local) == f"{REPO}@{digest}"
    else:
        with pytest.raises(ValueError, match="does not contain"):
            gc.recovery_reference(local["RepoTags"][0], local)


def test_verify_platform_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    local = image(1)
    digest = "sha256:" + "a" * 64
    outputs = iter(
        [
            json.dumps({"digest": digest}),
            json.dumps(
                {
                    "manifests": [
                        {
                            "digest": "sha256:" + "b" * 64,
                            "platform": {"os": "linux", "architecture": "amd64"},
                        },
                        {
                            "digest": "sha256:" + "c" * 64,
                            "platform": {"os": "linux", "architecture": "arm64"},
                        },
                    ]
                }
            ),
            json.dumps({"config": {"digest": local["Id"]}}),
        ]
    )
    monkeypatch.setattr(gc, "run", lambda *args: next(outputs))
    assert gc.recovery_reference(local["RepoTags"][0], local) == f"{REPO}@{digest}"


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("verified", [False, True])
def test_main_dry_run_and_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apply: bool, verified: bool
) -> None:
    images = [image(n) for n in range(1, 7)]
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "registry": "registry.example:8443",
                "namespace": "team",
                "images": [{"repository": "runtime"}],
            }
        )
    )
    commands: list[tuple[str, ...]] = []

    def fake_run(*args: str) -> str:
        commands.append(args)
        if args[1] == "info":
            return "/runner-data/docker\n"
        if args[1:3] == ("image", "ls"):
            return "\n".join(item["Id"] for item in images)
        if args[1:3] == ("image", "inspect"):
            return json.dumps([images[0]] if len(args) == 4 else images)
        if args[1:3] == ("image", "rm"):
            return "removed"
        raise AssertionError(args)

    def fake_reference(tag: str, item: dict[str, Any]) -> str:
        if not verified:
            raise ValueError("registry content does not match")
        return f"{REPO}@{item['Id']}"

    monkeypatch.setattr(gc, "run", fake_run)
    monkeypatch.setattr(gc, "recovery_reference", fake_reference)
    monkeypatch.setattr(
        "sys.argv",
        ["gc", "--config", str(config), "--output", str(tmp_path / "out")]
        + (["--apply"] if apply else []),
    )
    if verified:
        gc.main()
    else:
        with pytest.raises(SystemExit, match="could not be safely removed"):
            gc.main()
    deletions = [cmd for cmd in commands if cmd[1:3] == ("image", "rm")]
    assert deletions == (
        [("docker", "image", "rm", "--no-prune", images[0]["RepoTags"][0])]
        if apply and verified
        else []
    )

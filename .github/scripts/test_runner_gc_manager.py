from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "deploy/ci/github-runner/gc/manager.py"
SPEC = importlib.util.spec_from_file_location("gc_manager", PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("RUNNER_GC_RUNTIME", str(tmp_path / "runtime"))
    monkeypatch.setenv("RUNNER_GC_DATA", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    instance = module.Manager()
    monkeypatch.setattr(instance, "check_root", lambda: None)
    monkeypatch.setattr(instance, "free", lambda: 200 * module.GIB)
    return instance


def test_daily_boundary_uses_beijing_time() -> None:
    assert (
        module.schedule_date(datetime(2026, 9, 28, 20, 59, tzinfo=timezone.utc))
        == "2026-09-28"
    )
    assert (
        module.schedule_date(datetime(2026, 9, 28, 21, 0, tzinfo=timezone.utc))
        == "2026-09-29"
    )


@pytest.mark.parametrize(
    "initial,after,expected",
    [
        (200, [], []),
        (90, [160], [("until=168h",)]),
        (90, [120, 160], [("until=168h",), ("30GB",)]),
        (90, [110, 130, 180], [("until=168h",), ("30GB",), ()]),
    ],
)
def test_watermarks_stop_as_soon_as_target_is_met(
    manager: Any,
    monkeypatch: pytest.MonkeyPatch,
    initial: int,
    after: list[int],
    expected: list[tuple[str, ...]],
) -> None:
    free = [initial]
    remaining = iter(after)
    calls = []
    monkeypatch.setattr(manager, "free", lambda: free[0] * module.GIB)

    def cache(*args: str) -> None:
        calls.append(tuple(args[1:]))
        free[0] = next(remaining)

    monkeypatch.setattr(manager, "cache", cache)
    manager.recover_space(daily=False)
    assert calls == expected


def test_pre_job_refuses_insufficient_space_without_leaving_active_marker(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "worker", lambda: {"pid": 123, "start": "1"})
    monkeypatch.setattr(manager, "free", lambda: 90 * module.GIB)
    monkeypatch.setattr(manager, "cache", lambda *args: None)
    with pytest.raises(RuntimeError, match="refusing build"):
        manager.started()
    assert not list(manager.runtime.glob("job-*.json"))
    assert not manager.blocked.exists()  # Completed GC, simply insufficient space.
    assert manager.status()["last_error"]


def test_hook_markers_exclude_daily_gc_until_job_completion(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = {"pid": 123, "start": "1"}
    monkeypatch.setattr(module, "worker", lambda: owner)
    monkeypatch.setattr(module, "process", lambda pid: owner)
    calls = []
    monkeypatch.setattr(
        manager, "collect", lambda reason, **kwargs: calls.append(reason)
    )
    manager.started()
    manager.tick(datetime.now(timezone.utc))
    assert calls == []
    manager.completed()
    manager.tick(datetime.now(timezone.utc))
    assert calls == ["daily"]


@pytest.mark.parametrize("current", [None, {"pid": 123, "start": "NEW"}])
def test_dead_worker_or_pid_reuse_does_not_block_gc(
    manager: Any, monkeypatch: pytest.MonkeyPatch, current: dict[str, Any] | None
) -> None:
    path = manager.runtime / "job-123.json"
    path.write_text(json.dumps({"pid": 123, "start": "OLD"}))
    monkeypatch.setattr(module, "process", lambda pid: current)
    assert manager.jobs() == []
    assert not path.exists()


def test_cross_process_lock_blocks_another_gc(manager: Any) -> None:
    code = """import importlib.util, sys
spec = importlib.util.spec_from_file_location('m', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
try:
    with m.Manager().lock():
        pass
except m.Busy:
    sys.exit(42)
"""
    with manager.lock():
        result = subprocess.run([sys.executable, "-c", code, str(PATH)], timeout=10)
    assert result.returncode == 42
    assert (
        subprocess.run([sys.executable, "-c", code, str(PATH)], timeout=10).returncode
        == 0
    )


@pytest.mark.parametrize("image_status", [1, 2])
def test_image_failure_does_not_block_cache_recovery(
    manager: Any, monkeypatch: pytest.MonkeyPatch, image_status: int
) -> None:
    calls = []
    monkeypatch.setattr(
        manager,
        "command",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, image_status),
    )
    monkeypatch.setattr(manager, "cache", lambda *args: calls.append(args))
    state = manager.collect("daily")
    assert calls == [("--filter", "until=168h")]
    assert bool(state["last_error"]) == (image_status == 1)
    assert not manager.blocked.exists()
    if image_status == 2:
        assert state["last_success"] > 0
        assert state["last_daily_date"]
        assert (
            'image_verification_warning{runner="dingo-gxl-runner"} 1'
            in manager.metrics()
        )


def test_prune_timeout_fails_closed_for_following_jobs(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cache(*args: str) -> None:
        raise subprocess.TimeoutExpired("docker", 300)

    monkeypatch.setattr(manager, "cache", cache)
    with pytest.raises(subprocess.TimeoutExpired):
        manager.collect("daily", images=False)
    assert manager.blocked.exists()
    with pytest.raises(RuntimeError, match="interrupted"):
        manager.allowed()


def test_killed_gc_process_leaves_fail_closed_marker(manager: Any) -> None:
    code = """import importlib.util, os, sys
spec = importlib.util.spec_from_file_location('m', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
manager = m.Manager()
manager.check_root = lambda: None
manager.free = lambda: 200 * m.GIB
manager.cache = lambda *args: os._exit(9)
with manager.lock():
    manager.collect('daily', images=False)
"""
    assert (
        subprocess.run([sys.executable, "-c", code, str(PATH)], timeout=10).returncode
        == 9
    )
    with manager.lock():
        with pytest.raises(RuntimeError, match="interrupted"):
            manager.allowed()


def test_daily_runs_once_and_catches_up_when_previously_busy(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(timezone.utc)
    calls = []

    def collect(reason: str) -> None:
        calls.append(reason)
        manager.save({"last_daily_date": module.schedule_date(now)})

    monkeypatch.setattr(manager, "collect", collect)
    with manager.lock():
        manager.tick(now)
    assert calls == []
    manager.tick(now)
    manager.tick(now)
    assert calls == ["daily"]
    # A new process reads the same daily completion marker.
    another = module.Manager()
    monkeypatch.setattr(another, "collect", collect)
    another.tick(now)
    assert calls == ["daily"]


def test_manual_apply_argument_is_not_swallowed(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "Manager", lambda: manager)
    monkeypatch.setattr(module, "worker", lambda: None)
    calls = []
    monkeypatch.setattr(manager, "collect", lambda reason: calls.append(reason) or {})
    monkeypatch.setattr(sys, "argv", ["manager.py", "manual", "--apply"])
    module.main()
    assert calls == ["manual"]


def test_rendered_bundle_contains_hooks_gc_and_no_github_schedule(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runner.yaml"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy/ci/github-runner/render_gc.py"),
            "--output",
            str(output),
        ],
        check=True,
    )
    documents = list(yaml.safe_load_all(output.read_text()))
    cm = next(
        d
        for d in documents
        if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "dingo-gxl-runner-gc"
    )
    assert set(cm["data"]) == {
        "manager.py",
        "runner_gc.py",
        "job-started.sh",
        "job-completed.sh",
        "dingo-images.json",
    }
    assert json.loads(cm["data"]["dingo-images.json"]) == json.loads(
        (ROOT / ".github/dingo-images.json").read_text()
    )
    base = next(
        d
        for d in documents
        if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "dingo-gxl-runner"
    )
    assert json.loads(base["data"]["daemon.json"])["builder"]["gc"]["enabled"]
    entry = base["data"]["entrypoint.sh"]
    assert entry.index("manager.py bootstrap") < entry.index("docker_config=")
    assert entry.index("manager.py serve") < entry.index("./run.sh &")
    assert "ACTIONS_RUNNER_HOOK_JOB_STARTED=/etc/dingo-gc/job-started.sh" in entry
    for script in [entry, cm["data"]["job-started.sh"], cm["data"]["job-completed.sh"]]:
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    deployment = next(d for d in documents if d["kind"] == "Deployment")
    assert deployment["spec"]["template"]["metadata"]["annotations"][
        "dingo-ci/gc-checksum"
    ]
    assert any(
        v["name"] == "gc-config"
        for v in deployment["spec"]["template"]["spec"]["volumes"]
    )
    workflow = yaml.load(
        (ROOT / ".github/workflows/runner-gc.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert "schedule" not in workflow["on"]
    assert "workflow_dispatch" in workflow["on"]


def test_registry_secret_isolated_from_job_credentials(tmp_path: Path) -> None:
    output = tmp_path / "runner.yaml"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy/ci/github-runner/render_gc.py"),
            "--output",
            str(output),
            "--registry-secret",
            "gc-readonly",
        ],
        check=True,
    )
    docs = list(yaml.safe_load_all(output.read_text()))
    spec = next(d for d in docs if d["kind"] == "Deployment")["spec"]["template"][
        "spec"
    ]
    volume = next(v for v in spec["volumes"] if v["name"] == "gc-registry")
    assert volume["secret"]["secretName"] == "gc-readonly"
    assert volume["secret"]["items"] == [
        {"key": ".dockerconfigjson", "path": "config.json"}
    ]
    assert {
        "name": "RUNNER_GC_DOCKER_CONFIG",
        "value": "/etc/dingo-gc-registry",
    } in spec["containers"][0]["env"]


def test_image_command_uses_dedicated_auth(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNNER_GC_DOCKER_CONFIG", "/readonly-auth")
    calls = []

    def command(*args: str, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return subprocess.CompletedProcess(args, 2)

    monkeypatch.setattr(manager, "command", command)
    monkeypatch.setattr(manager, "cache", lambda *args: None)
    manager.collect("daily")
    assert calls[0]["env"]["DOCKER_CONFIG"] == "/readonly-auth"
    manager.collect("pre-job", images=False)
    assert manager.status()["last_image_status"] == 2

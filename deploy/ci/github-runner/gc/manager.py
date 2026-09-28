#!/usr/bin/env python3
"""Runner-local scheduling, job admission and GC coordination (Linux only)."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Iterator

CST = timezone(timedelta(hours=8))
GIB = 1024**3


def process(pid: int) -> dict[str, Any] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        name = text[text.index("(") + 1 : text.rindex(")")]
        fields = text[text.rindex(")") + 2 :].split()
        return {"pid": pid, "start": fields[19], "name": name, "ppid": int(fields[1])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def worker() -> dict[str, Any] | None:
    pid = os.getppid()
    while pid > 1:
        item = process(pid)
        if item is None:
            return None
        if item["name"] == "Runner.Worker":
            return item
        pid = item["ppid"]
    return None


def schedule_date(now: datetime) -> str:
    local = now.astimezone(CST)
    if local.hour < 5:
        local -= timedelta(days=1)
    return local.date().isoformat()


class Busy(RuntimeError):
    pass


class Manager:
    def __init__(self) -> None:
        self.runtime = Path(os.environ.get("RUNNER_GC_RUNTIME", "/run/dingo-runner-gc"))
        self.data = Path(os.environ.get("RUNNER_GC_DATA", "/runner-data"))
        self.history = self.data / "gc"
        self.config = Path(__file__).with_name("dingo-images.json")
        self.target = int(os.environ.get("RUNNER_GC_TARGET_GIB", "150")) * GIB
        self.minimum = int(os.environ.get("RUNNER_GC_MIN_GIB", "100")) * GIB
        self.timeout = int(os.environ.get("RUNNER_GC_COMMAND_TIMEOUT", "300"))
        if not 0 < self.minimum <= self.target or self.timeout <= 0:
            raise ValueError("invalid GC thresholds or timeout")
        self.runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.blocked = self.runtime / "maintenance.json"

    def free(self) -> int:
        return shutil.disk_usage(self.data).free

    @contextlib.contextmanager
    def lock(self, wait: float = 0) -> Iterator[None]:
        with (self.runtime / "lock").open("a") as handle:
            deadline = time.monotonic() + wait
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise Busy("GC/build lock is busy")
                    time.sleep(1)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def jobs(self, exclude: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        active = []
        for path in self.runtime.glob("job-*.json"):
            item = json.loads(path.read_text())
            current = process(item["pid"])
            if current is None or current["start"] != item["start"]:
                path.unlink()  # Dead Worker or reused PID; no permanent stale lock.
            elif exclude is None or (item["pid"], item["start"]) != (
                exclude["pid"],
                exclude["start"],
            ):
                active.append(item)
        return active

    def allowed(self, exclude: dict[str, Any] | None = None) -> None:
        if self.blocked.exists():
            raise RuntimeError(
                "previous GC was interrupted; inspect maintenance.json before restarting the Pod"
            )
        if self.jobs(exclude):
            raise Busy("a build job is active")

    def status(self) -> dict[str, Any]:
        # Runtime copy takes precedence if the PVC was full when saving status.
        for path in (self.runtime / "status.json", self.history / "status.json"):
            if path.exists():
                return json.loads(path.read_text())
        return {}

    def save(self, state: dict[str, Any]) -> None:
        for path in (self.runtime / "status.json", self.history / "status.json"):
            try:
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(state, indent=2) + "\n")
                temporary.replace(path)
            except OSError:
                if path.parent == self.runtime:
                    raise
                print(
                    "WARNING: persistent GC status unavailable; runtime copy retained",
                    flush=True,
                )

    def command(
        self, *args: str, timeout: int | None = None, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        print("COMMAND " + json.dumps(args), flush=True)
        result = subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout or self.timeout,
            env=env,
        )
        print(result.stdout, end="", flush=True)
        return result

    def check_root(self) -> None:
        actual = subprocess.check_output(
            ["docker", "info", "--format", "{{.DockerRootDir}}"], text=True, timeout=30
        ).strip()
        if actual != "/runner-data/docker" or str(self.data) != "/runner-data":
            raise RuntimeError(f"unexpected Docker root/data: {actual}, {self.data}")

    def cache(self, *options: str) -> None:
        self.command(
            "docker", "builder", "prune", "--all", "--force", *options
        ).check_returncode()

    def recover_space(self, *, daily: bool) -> None:
        # Measure the filesystem after every stage; cache accounting is not a quota.
        if daily or self.free() < self.target:
            self.cache("--filter", "until=168h")
        if self.free() < self.target:
            self.cache("--keep-storage", "30GB")
        if self.free() < self.target:
            self.cache()

    def collect(self, reason: str, *, images: bool = True) -> dict[str, Any]:
        self.check_root()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        # Reports must be writable even when the data PVC is completely full.
        output = self.runtime / f"run-{stamp}"
        output.mkdir(mode=0o700)
        state = self.status()
        state.update(last_attempt=time.time(), last_reason=reason, last_error="")
        self.save(state)
        self.blocked.write_text(
            json.dumps(
                {"reason": reason, "started": time.time(), "report": str(output)}
            )
        )
        try:
            with (
                (output / "gc.log").open("w") as log,
                contextlib.redirect_stdout(log),
                contextlib.redirect_stderr(log),
            ):
                print(
                    datetime.now(timezone.utc).isoformat(),
                    reason,
                    "before_bytes",
                    self.free(),
                    flush=True,
                )
                # First make room for image metadata updates. Registry failures
                # must never prevent independent build-cache reclamation.
                if self.free() < self.minimum:
                    self.recover_space(daily=False)
                image_code = 0
                if images:
                    image_env = os.environ.copy()
                    if image_env.get("RUNNER_GC_DOCKER_CONFIG"):
                        image_env["DOCKER_CONFIG"] = image_env[
                            "RUNNER_GC_DOCKER_CONFIG"
                        ]
                    result = self.command(
                        sys.executable,
                        str(Path(__file__).with_name("runner_gc.py")),
                        "--config",
                        str(self.config),
                        "--keep",
                        "5",
                        "--output",
                        str(output),
                        "--apply",
                        timeout=900,
                        env=image_env,
                    )
                    image_code = result.returncode
                    # 2 means protected/unverifiable candidates, not a failed GC.
                    if image_code not in (0, 2):
                        print(
                            "WARNING: image GC failed; continuing cache reclamation",
                            flush=True,
                        )
                self.recover_space(daily=reason in ("daily", "manual"))
                free = self.free()
                if images:
                    state["last_image_status"] = image_code
                state.update(
                    free_bytes=free,
                    last_error=""
                    if free >= self.minimum and image_code in (0, 2)
                    else "space or image GC failure",
                )
                if not state["last_error"]:
                    state["last_success"] = time.time()
                    if reason == "daily":
                        state["last_daily_success"] = time.time()
                        state["last_daily_date"] = schedule_date(
                            datetime.now(timezone.utc)
                        )
                print("after_bytes", free, json.dumps(state), flush=True)
            self.save(state)
            self.blocked.unlink()  # Only a completed run clears the fail-closed marker.
        except BaseException as error:
            state["last_error"] = str(error)
            self.save(state)
            raise
        finally:
            try:
                self.history.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copytree(output, self.history / output.name)
                shutil.rmtree(output)
            except OSError:
                pass  # Emergency report remains on the container's root disk.
            self.rotate_reports()
        return state

    def rotate_reports(self) -> None:
        cutoff = time.time() - 14 * 86400
        for root in (self.history, self.runtime):
            for path in root.glob("run-*"):
                if (
                    path.is_dir()
                    and not path.is_symlink()
                    and path.stat().st_mtime < cutoff
                ):
                    shutil.rmtree(path)

    def admit(self) -> None:
        if self.free() < self.target:
            self.collect("pre-job", images=False)
        if self.free() < self.minimum:
            raise RuntimeError("less than 100GiB available after GC; refusing build")

    def started(self) -> None:
        owner = worker()
        if owner is None:
            raise RuntimeError("job hook must be a descendant of Runner.Worker")
        with self.lock(wait=1200):
            self.allowed(owner)
            self.check_root()
            self.admit()
            (self.runtime / f"job-{owner['pid']}.json").write_text(json.dumps(owner))

    def completed(self) -> None:
        owner = worker()
        if owner is None:
            raise RuntimeError("job hook must be a descendant of Runner.Worker")
        with self.lock(wait=1200):
            path = self.runtime / f"job-{owner['pid']}.json"
            if path.exists():
                path.unlink()
            # Cleanup is safe here: the completed hook runs after all job steps.
            self.allowed(owner)
            if self.free() < self.target:
                self.collect("post-job", images=False)

    def tick(self, now: datetime) -> None:
        state = self.status()
        if state.get("last_daily_date") == schedule_date(now):
            return
        if time.time() - state.get("last_attempt", 0) < 1800 and state.get(
            "last_error"
        ):
            return
        try:
            with self.lock():
                self.allowed()
                self.collect("daily")
        except Busy:
            pass  # Retry after one minute; the daily request is not lost.

    def metrics(self) -> str:
        state = self.status()
        values = {
            "free_bytes": self.free(),
            "last_success_timestamp_seconds": state.get("last_success", 0),
            "last_daily_success_timestamp_seconds": state.get("last_daily_success", 0),
            "last_attempt_timestamp_seconds": state.get("last_attempt", 0),
            "maintenance_blocked": int(self.blocked.exists()),
            "last_run_failed": int(bool(state.get("last_error"))),
            "image_verification_warning": int(state.get("last_image_status", 0) == 2),
        }
        return "".join(
            f'dingo_runner_gc_{key}{{runner="dingo-gxl-runner"}} {value}\n'
            for key, value in values.items()
        )

    def serve(self) -> None:
        manager = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/metrics":
                    self.send_error(404)
                    return
                payload = manager.metrics().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        server = ThreadingHTTPServer(("0.0.0.0", 9105), Handler)
        Thread(target=server.serve_forever, daemon=True).start()
        while True:
            try:
                self.tick(datetime.now(timezone.utc))
            except Exception as error:
                print(f"GC ERROR: {error}", flush=True)
            time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["bootstrap", "serve", "started", "completed", "manual", "run"]
    )
    parser.add_argument("--apply", action="store_true")
    args, command = parser.parse_known_args()
    if command and args.mode != "run":
        parser.error("unexpected arguments")
    manager = Manager()
    if args.mode == "serve":
        manager.serve()
    elif args.mode == "started":
        manager.started()
    elif args.mode == "completed":
        manager.completed()
    else:
        with manager.lock(wait=1200):
            manager.allowed(worker())
            manager.check_root()
            if args.mode == "bootstrap":
                manager.admit()
            elif args.mode == "run":
                manager.admit()
                command = command[1:] if command[:1] == ["--"] else command
                if not command:
                    raise ValueError("run requires a command")
                manager.blocked.write_text(
                    json.dumps({"reason": "manual-build", "started": time.time()})
                )
                code = subprocess.call(command)
                manager.blocked.unlink()
                raise SystemExit(code)
            elif args.apply:
                state = manager.collect("manual")
                if state.get("last_error"):
                    raise SystemExit(1)
            else:
                output = manager.runtime / f"preview-{time.time_ns()}"
                preview_env = os.environ.copy()
                if preview_env.get("RUNNER_GC_DOCKER_CONFIG"):
                    preview_env["DOCKER_CONFIG"] = preview_env[
                        "RUNNER_GC_DOCKER_CONFIG"
                    ]
                raise SystemExit(
                    subprocess.call(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("runner_gc.py")),
                            "--config",
                            str(manager.config),
                            "--keep",
                            "5",
                            "--output",
                            str(output),
                        ],
                        env=preview_env,
                    )
                )


if __name__ == "__main__":
    main()

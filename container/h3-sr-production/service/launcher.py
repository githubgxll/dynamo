#!/usr/bin/env python3
"""Start H3 then SR and expose operational readiness, without GPU monitoring."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from uuid import uuid4
from datetime import datetime, timedelta, timezone

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "service"

from . import profile as profiles
from .control import HTTPControlClient
from .state import FailureLatch, atomic_json, now, process_birth, read_json

PREFLIGHT_TIMEOUT = 600
IDENTITY_TIMEOUT = 150
GATEWAY_TIMEOUT = 600


def backend_command(profile_path, role):
    return [sys.executable, "-B", "-P", "-m", "service.backend",
            "--profile", str(profile_path), "--role", role]


def process_group_identity(pid):
    """Read a stable PID birth together with its Linux group/session."""
    before = process_birth(pid)
    pgid, sid = os.getpgid(pid), os.getsid(pid)
    if process_birth(pid) != before:
        raise RuntimeError("PID birth changed while checking process-group ownership")
    return {"birth": before, "pgid": pgid, "sid": sid}


def process_group_members(pgid):
    """Snapshot this group, ignoring only processes which disappeared."""
    members = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            if int(fields[2]) != pgid:
                continue
            identity = process_group_identity(int(entry.name))
            if identity["pgid"] == pgid:
                members[entry.name] = identity
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


class Supervisor:
    def __init__(self, profile_path, *, gateway_command_file, root=None, readiness_port=8125, startup_timeout=10800):
        if type(startup_timeout) is not int or not 0 < startup_timeout <= 10800:
            raise ValueError("Model startup timeout must be within the 10800-second maximum")
        self.profile_path = Path(profile_path).resolve()
        self.profile = profiles.load(self.profile_path)
        self.gateway_command_file = Path(gateway_command_file).resolve()
        self.gateway_argv = profiles.gateway_argv(self.profile, self.profile_path,
            json.loads(self.gateway_command_file.read_text(encoding="utf-8")))
        self.root = Path(root or "/work/h3-sr/control")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise RuntimeError("Supervisor control root cannot be a symlink")
        self.epoch = "svc-" + uuid4().hex
        self.latch = FailureLatch(self.root)
        self.children = {}
        self.logs = []
        self.startup_timeout = startup_timeout
        self.readiness_port = readiness_port
        self.state = {"schema": "h3sr-production-state/1", "profile": self.profile["profile"],
            "service_epoch": self.epoch, "service_ready": False, "gateway_ready": False, "phase": "preparing", "children": {},
            "monitoring_mode": self.profile["monitoring"]["mode"], "started_at": now(),
            "startup_deadline": (datetime.now(timezone.utc) + timedelta(
                seconds=2 * startup_timeout + PREFLIGHT_TIMEOUT + IDENTITY_TIMEOUT + GATEWAY_TIMEOUT)).isoformat(),
            "profile_sha256": hashlib.sha256(self.profile_path.read_bytes()).hexdigest()}

    def write_state(self, **values):
        self.state.update(values)
        atomic_json(self.root / "state.json", self.state)

    def assert_fresh(self):
        for name in ("state.json", "active.json", "blocked.json", "control-token", "gateway-ready.json"):
            if (self.root / name).exists():
                raise RuntimeError("Existing lifecycle/lease state preserved; use explicit recovery, never auto-replay")

    def _server(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path not in ("/health", "/ready", "/state"):
                    self.send_error(404)
                    return
                state = dict(owner.state)
                ready = owner.is_ready()
                state["service_ready"] = ready
                code = 200 if self.path == "/state" or (self.path == "/health" and state["phase"] != "failed") or ready else 503
                body = json.dumps(state, allow_nan=False).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", self.readiness_port), Handler)
        threading.Thread(target=server.serve_forever, name="h3sr-readiness", daemon=True).start()
        return server

    def _wait_health(self, role, port):
        deadline = time.monotonic() + self.startup_timeout
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            self._assert_children_alive()
            try:
                with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError(role + " model startup health budget exceeded")

    def _assert_children_alive(self):
        for role, process in self.children.items():
            if process.poll() is not None:
                raise RuntimeError(role + " owned process exited; inspect its log and preserve lifecycle state")

    def is_ready(self):
        return (self.state["service_ready"] is True and self.state["gateway_ready"] is True
                and self.state["phase"] == "ready" and self.latch.blocked is None
                and not (self.root / "blocked.json").exists()
                and set(self.children) == {"h3", "sr", "gateway"}
                and all(process.poll() is None for process in self.children.values()))

    def _spawn_owned(self, role, command, env):
        logs = self.root.parent / "logs"
        logs.mkdir(exist_ok=True)
        stream = (logs / (role + ".log")).open("xb")
        self.logs.append(stream)
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.children[role] = child
        self.state["children"][role] = {"pid": child.pid, "birth": process_birth(child.pid),
            "pgid": child.pid, "sid": child.pid, "members": {}, "command": command}
        self._verify_owned_group(role, child)
        self.write_state(phase="loading-" + role)

    def _start(self, role):
        env = profiles.environment(self.profile, role, profile_path=self.profile_path, epoch=self.epoch, root=self.root)
        command = backend_command(self.profile_path, role)
        self._spawn_owned(role, command, env)
        port = 8123 if role == "h3" else 8124
        self._wait_health(role, port)
        self._verify_owned_group(role, self.children[role])
        self.state["children"][role]["ready"] = True
        self.write_state(phase=role + "-ready")

    def _wait_gateway_registration(self):
        deadline = time.monotonic() + GATEWAY_TIMEOUT
        path = self.root / "gateway-ready.json"
        while time.monotonic() < deadline:
            self._assert_children_alive()
            if path.exists():
                receipt = read_json(path)
                owned = self.state["children"]["gateway"]
                expected = {"schema": "h3sr-gateway-registration/1", "status": "REGISTERED",
                    "service_epoch": self.epoch, "profile_sha256": self.state["profile_sha256"],
                    "pid": owned["pid"], "birth": owned["birth"]}
                if any(receipt.get(key) != value for key, value in expected.items()):
                    raise RuntimeError("Gateway registration differs from this owned process/profile/epoch")
                if process_birth(owned["pid"]) != owned["birth"]:
                    raise RuntimeError("Gateway registered PID birth changed")
                self._verify_owned_group("gateway", self.children["gateway"])
                return
            time.sleep(1)
        raise TimeoutError("Gateway registration exceeded 600 seconds")

    def _start_gateway(self):
        env = profiles.gateway_environment(self.profile, profile_path=self.profile_path, epoch=self.epoch, root=self.root)
        self._spawn_owned("gateway", [sys.executable, "-B", "-m", "dingo.vllm.omni", *self.gateway_argv], env)
        self._wait_gateway_registration()
        self.state["children"]["gateway"]["ready"] = True

    def start(self):
        self.assert_fresh()
        fd = os.open(self.root / "control-token", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            stream.write(uuid4().hex + uuid4().hex)
        self.write_state()
        self._preflight()
        self._start("h3")
        self._start("sr")
        if self.profile["handoff"]["enabled"]:
            report = asyncio.run(HTTPControlClient(self.root, epoch=self.epoch).identities())
            self.state["sr_worker_identities"] = report["ranks"]
        self._assert_children_alive()
        # The bridge may now validate the two internal model services. Public
        # readiness remains false until Dingo has registered this exact epoch.
        self.write_state(service_ready=True, phase="loading-gateway", gateway_ready=False)
        self._start_gateway()
        self._assert_children_alive()
        self.write_state(gateway_ready=True, phase="ready", ready_at=now())

    def _preflight(self):
        commands = [
            [sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "runtime.py"),
             "verify", "--output-dir", str(self.root.parent / "verification")],
            [sys.executable, "-B", "/opt/h3-weight-init/convert_seedvr2.py", "--source-dir",
             self.profile["sr_weight_preparation"]["source_dir"], "--output-dir",
             self.profile["sr_weight_preparation"]["output_dir"], "--verify-only"],
        ]
        deadline = time.monotonic() + PREFLIGHT_TIMEOUT
        for index, command in enumerate(commands):
            with (self.root / f"preflight-{index}.log").open("xb") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=max(1, deadline - time.monotonic()), env=profiles.environment(
                        self.profile, "h3", profile_path=self.profile_path, epoch=self.epoch, root=self.root))
            if result.returncode != 0:
                raise RuntimeError("Installed runtime or read-only converted weights failed verification")

    def _verify_owned_group(self, role, process):
        """Require a live, previously identified member before group signalling.

        Ready-time and pre-TERM snapshots retain worker births so an exited
        leader does not prevent cleanup of its surviving native workers. An
        unanchored numeric PGID is never sufficient evidence of ownership.
        """
        owned = self.state["children"].get(role, {})
        pgid, sid = owned.get("pgid"), owned.get("sid")
        if pgid != process.pid or sid != process.pid or owned.get("pid") != process.pid or not owned.get("birth"):
            raise RuntimeError(role + " has no verified owned group/session")

        def leader_identity():
            try:
                leader = process_group_identity(process.pid)
            except (FileNotFoundError, ProcessLookupError):
                return None
            if leader["birth"] != owned["birth"] or leader["pgid"] != pgid or leader["sid"] != sid:
                raise RuntimeError(role + " PID birth/group/session changed; no signal sent")
            return leader

        leader = leader_identity()  # poll() may already have reaped this PID.
        members = process_group_members(pgid)
        if leader is not None:
            members[str(process.pid)] = leader
        if any(value["pgid"] != pgid or value["sid"] != sid for value in members.values()):
            raise RuntimeError(role + " group/session ownership differs; no signal sent")
        if not members:
            return False
        known = {**owned.get("members", {}), str(process.pid): owned["birth"]}
        anchors = [(pid, value) for pid, value in members.items() if value["birth"] == known.get(pid)]
        if not anchors:
            raise RuntimeError(role + " group has no surviving owned identity; no signal sent")
        # Recheck both leader (even if previously absent) and an exact surviving
        # anchor immediately before the caller's signal. This is a bounded
        # /proc check, not an atomic Linux process-group handle.
        leader_identity()
        for pid, expected in anchors:
            try:
                current = process_group_identity(int(pid))
            except (FileNotFoundError, ProcessLookupError):
                continue
            if current == expected:
                leader_identity()
                owned["members"] = {pid: value["birth"] for pid, value in members.items()}
                return True
        raise RuntimeError(role + " owned group anchor disappeared or changed; no signal sent")

    def stop_owned(self):
        previous = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGTERM, signal.SIGINT)}
        errors = []
        try:
            for role, process in reversed(list(self.children.items())):
                try:
                    if not self._verify_owned_group(role, process):
                        continue
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
                except (OSError, RuntimeError) as error:
                    errors.append(role + " TERM ownership/cleanup failed: " + str(error))
                    continue
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    if self._verify_owned_group(role, process):
                        os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except (OSError, RuntimeError) as error:
                    errors.append(role + " KILL ownership/cleanup failed: " + str(error))
                    continue
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    errors.append(role + " did not exit after owned-group cleanup")
        finally:
            for stream in self.logs:
                stream.close()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        if errors:
            error = RuntimeError("; ".join(errors))
            self.latch.trip(epoch=self.epoch, error=error)
            self.write_state(service_ready=False, shutdown_errors=errors)
            raise error

    def run(self):
        import fcntl
        ownership = self.root / "supervisor.lock"
        if ownership.is_symlink():
            raise RuntimeError("Supervisor ownership file cannot be a symlink")
        with ownership.open("a+b") as stream:
            # A second launcher cannot overwrite the first service's state or
            # publish a new epoch while the old supervisor still owns it.
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assert_fresh()  # failure here leaves every old receipt intact
            server = self._server()
            try:
                self.start()
                while True:
                    self._assert_children_alive()
                    time.sleep(.5)
            except BaseException as error:
                self.latch.trip(epoch=self.epoch, error=error)
                self.write_state(service_ready=False, gateway_ready=False, phase="failed", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                try:
                    self.stop_owned()
                finally:
                    server.shutdown()
                    server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="/etc/h3-sr/profile.json")
    parser.add_argument("--gateway-command-file", required=True)
    parser.add_argument("--control-root", default="/work/h3-sr/control")
    parser.add_argument("--readiness-port", type=int, default=8125)
    parser.add_argument("--startup-timeout", type=int, default=10800)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Production supervisor requires Linux")

    def stop(signum, _frame):
        raise KeyboardInterrupt("Owned supervisor received signal " + str(signum))

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    Supervisor(args.profile, gateway_command_file=args.gateway_command_file, root=args.control_root, readiness_port=args.readiness_port,
               startup_timeout=args.startup_timeout).run()


if __name__ == "__main__":
    main()

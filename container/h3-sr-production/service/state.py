"""Small durable control state, independent of monitoring and CUDA."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4

ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("Invalid control identity")
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError("Control state cannot be a symlink")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Control state must be an object")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("Control paths cannot be symlinks")
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def process_birth(pid=None):
    pid = os.getpid() if pid is None else pid
    if type(pid) is not int or pid <= 0:
        raise ValueError("Invalid process PID")
    root = Path("/proc") / str(pid)
    fields = (root / "stat").read_text().rsplit(") ", 1)[1].split()
    return {"pid": pid, "process_start_ticks": int(fields[19]),
            "pid_namespace": os.readlink(root / "ns/pid"),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


class FailureLatch:
    """Latch in memory before durable write; failed persistence never reopens it."""
    def __init__(self, root):
        self.root = Path(root)
        self.blocked = None

    def assert_open(self):
        if self.blocked is not None or (self.root / "blocked.json").exists():
            raise RuntimeError("H3/SR control is blocked; reconcile the existing operation")

    def trip(self, *, epoch, action_id=None, error):
        if self.blocked is None:
            self.blocked = {"schema": "h3sr-production-blocked/1", "status": "BLOCKED",
                "service_epoch": epoch, "action_id": action_id, "error_type": type(error).__name__,
                "error": str(error), "at": now()}
            path = self.root / "blocked.json"
            if not path.exists():
                atomic_json(path, self.blocked)
        return self.blocked

"""Fail closed on Python socket access and known download subprocesses.

This is a process-level audit guard, NOT a kernel network namespace or a K8s
NetworkPolicy. Native libraries can bypass Python hooks. Build verification
must additionally run with BuildKit --network=none; Pod isolation needs the
platform's network controls if independently required.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

_installed = False


def install(log_path: Path) -> None:
    global _installed
    if _installed:
        raise RuntimeError("offline guard is already installed")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8", buffering=1)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})

    def deny(event: str, detail: str) -> None:
        # Do not record URLs/credentials or full command lines in diagnostics.
        log.write(json.dumps({"time": time.time(), "event": event, "decision": "blocked", "detail": detail}) + "\n")
        raise RuntimeError("offline guard blocked " + event + ": " + detail)

    def audit(event: str, args: tuple) -> None:
        if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
            deny(event, "DNS/name resolution is disabled")
        if event in {"socket.connect", "socket.sendto", "socket.sendmsg"}:
            if args and getattr(args[0], "family", None) in {socket.AF_INET, socket.AF_INET6}:
                deny(event, "IPv4/IPv6 access is disabled")
        if event in {"os.system", "os.exec", "os.posix_spawn"}:
            deny(event, "uncontrolled process execution is disabled")
        if event == "subprocess.Popen":
            executable = os.fsdecode(args[0]) if args[0] is not None else ""
            name = Path(executable).name.lower()
            command = args[1]
            tokens = [str(x) for x in command] if isinstance(command, (list, tuple)) else [str(command)]
            allowed = name in {"ffmpeg", "ffprobe", "ffmpeg.exe", "ffprobe.exe"}
            # Python/PyTorch may query the local linker cache on import.
            allowed = allowed or (name in {"ldconfig", "ldconfig.real"} and tokens[1:] == ["-p"])
            if not allowed or any("://" in value for value in tokens):
                deny(event, "unapproved child executable: " + name)

    log.write(json.dumps({"event": "guard_installed", "scope": "Python audit hooks; not OS network isolation"}) + "\n")
    sys.addaudithook(audit)
    _installed = True


def assert_no_attempts(log_path: Path) -> None:
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if json.loads(line).get("decision") == "blocked":
            raise RuntimeError("A network/process attempt was caught even if a library swallowed the exception")

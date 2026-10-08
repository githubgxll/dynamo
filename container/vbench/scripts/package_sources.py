#!/usr/bin/env python3
"""Explicit build-only package sources shared by bootstrap and dependency setup."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "package-sources.json"
PYTHON_INDEXES = {"https://pypi.tuna.tsinghua.edu.cn/simple", "https://pypi.org/simple"}
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
BOOTSTRAP_REQUIREMENTS = ("pip==24.3.1", "setuptools==75.8.0", "wheel==0.45.1", "uv==0.8.22")


def load_sources(path: Path | str | None = None) -> dict:
    """Reject ambiguous source overrides, embedded credentials, and TLS bypasses."""
    data = json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    expected = {"schema_version", "python_index", "torch_index"}
    if not isinstance(data, dict) or set(data) != expected or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("package-sources.json requires schema_version 1 and exactly two index URLs")
    for key in ("python_index", "torch_index"):
        value = data[key]
        if not isinstance(value, str):
            raise ValueError(f"{key} must be an approved HTTPS URL")
        parsed = urlsplit(value)
        if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError(f"{key} must use HTTPS without credentials, query, or fragment")
        allowed = PYTHON_INDEXES if key == "python_index" else {TORCH_INDEX}
        if value not in allowed:
            raise ValueError(f"{key} is not an approved package index")
    return data


def bootstrap(sources: dict, evidence: Path) -> None:
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "package-sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    status = {"schema_version": 1, "sources": sources, "requirements": list(BOOTSTRAP_REQUIREMENTS), "status": "STARTED"}
    record = evidence / "package-bootstrap.json"

    def write_status() -> None:
        record.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    write_status()
    print(f"Bootstrap Python index: {sources['python_index']}", flush=True)
    # --isolated ignores user pip configuration; the source is explicit and is
    # not written to pip.conf or an image ENV. Keep TLS verification enabled.
    command = [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check", "install",
               "--no-cache-dir", "--index-url", sources["python_index"], "--retries", "1", "--timeout", "15",
               *BOOTSTRAP_REQUIREMENTS]
    try:
        subprocess.run(command, check=True)
    except (subprocess.CalledProcessError, OSError) as error:
        status.update(status="FAILED", returncode=getattr(error, "returncode", None), error_type=type(error).__name__)
        write_status()
        raise
    status.update(status="PASSED", returncode=0)
    write_status()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("bootstrap", help="Install the four fixed packaging tools")
    command.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    bootstrap(load_sources(), args.evidence)


if __name__ == "__main__":
    main()

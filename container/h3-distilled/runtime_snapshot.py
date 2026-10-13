"""Record installed distributions without importing frameworks or accessing GPUs."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def snapshot():
    packages = [{"name": d.metadata.get("Name", ""), "version": d.version,
                 "metadata_path": str(getattr(d, "_path", ""))}
                for d in importlib.metadata.distributions()]
    return {"python": platform.python_version(), "executable": sys.executable,
            "packages": sorted(packages, key=lambda d: (d["name"].lower(), d["version"], d["metadata_path"])),
            "environment": {k: os.environ.get(k) for k in ("PATH", "VIRTUAL_ENV", "DYNAMO_COMMIT_SHA")}}


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(snapshot(), sort_keys=True, indent=2) + "\n", encoding="utf-8")

#!/usr/bin/env python3
"""Render the DingoRouter-base runner and GC ConfigMaps as one deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def render(registry_secret: str | None = None) -> list[dict[str, Any]]:
    documents = list(yaml.safe_load_all((HERE / "dingo-gxl-runner.yaml").read_text()))
    data = {
        name: (HERE / "gc" / name).read_text()
        for name in ("manager.py", "job-started.sh", "job-completed.sh")
    }
    data["runner_gc.py"] = (ROOT / ".github/scripts/runner_gc.py").read_text()
    data["dingo-images.json"] = (ROOT / ".github/dingo-images.json").read_text()
    fingerprint = hashlib.sha256(
        json.dumps([data, documents], sort_keys=True).encode()
    ).hexdigest()
    documents.insert(
        0,
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "dingo-gxl-runner-gc", "namespace": "elm-test"},
            "data": data,
        },
    )
    for document in documents:
        if (
            document["kind"] == "ConfigMap"
            and document["metadata"]["name"] == "dingo-gxl-runner"
        ):
            daemon = json.loads(document["data"]["daemon.json"])
            daemon["builder"] = {"gc": {"enabled": True, "defaultKeepStorage": "80GB"}}
            document["data"]["daemon.json"] = json.dumps(daemon, indent=2)
        if document["kind"] == "Deployment":
            template = document["spec"]["template"]
            template["metadata"].setdefault("annotations", {}).update(
                {
                    "dingo-ci/gc-checksum": fingerprint,
                    "prometheus.io/scrape": "true",
                    "prometheus.io/port": "9105",
                    "prometheus.io/path": "/metrics",
                }
            )
            spec = template["spec"]
            spec["volumes"].append(
                {
                    "name": "gc-config",
                    "configMap": {"name": "dingo-gxl-runner-gc", "defaultMode": 0o555},
                }
            )
            container = spec["containers"][0]
            container["volumeMounts"].append(
                {"name": "gc-config", "mountPath": "/etc/dingo-gc", "readOnly": True}
            )
            container.setdefault("ports", []).append(
                {"name": "gc-metrics", "containerPort": 9105}
            )
            if registry_secret:
                spec["volumes"].append(
                    {
                        "name": "gc-registry",
                        "secret": {
                            "secretName": registry_secret,
                            "defaultMode": 0o400,
                            "items": [
                                {"key": ".dockerconfigjson", "path": "config.json"}
                            ],
                        },
                    }
                )
                container["volumeMounts"].append(
                    {
                        "name": "gc-registry",
                        "mountPath": "/etc/dingo-gc-registry",
                        "readOnly": True,
                    }
                )
                container.setdefault("env", []).append(
                    {
                        "name": "RUNNER_GC_DOCKER_CONFIG",
                        "value": "/etc/dingo-gc-registry",
                    }
                )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry-secret",
        help="Existing dockerconfigjson Secret with registry read access",
    )
    args = parser.parse_args()
    args.output.write_text(
        yaml.safe_dump_all(render(args.registry_secret), sort_keys=False)
    )


if __name__ == "__main__":
    main()

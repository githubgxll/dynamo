# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render co-located PostgreSQL and HTTP puller; no Kubernetes discovery RBAC."""

import argparse
import json
from pathlib import Path

import http_pull
import render as log_renderer


def render(
    namespace,
    collector_image,
    raw,
    name="video-performance-http",
    postgres_image="postgres:16",
    queue_name=None,
):
    config = http_pull.configuration(raw)
    items = log_renderer.render(
        namespace,
        "configured-per-target",
        collector_image,
        config["storage"],
        name,
        postgres_image,
        queue_name,
    )["items"]
    deployment = next(x for x in items if x["kind"] == "Deployment")
    service = next(x for x in items if x["kind"] == "Service")
    configmap = next(x for x in items if x["kind"] == "ConfigMap")
    configmap["data"] = {"http-pull.json": json.dumps(raw)}
    result = [configmap, service, deployment]
    pod = deployment["spec"]["template"]["spec"]
    pod.pop("serviceAccountName", None)
    pod["automountServiceAccountToken"] = False
    container = next(c for c in pod["containers"] if c["name"] == "collector")
    container["command"] = ["python", "/opt/collector/http_pull.py"]
    container["env"] = [e for e in container["env"] if e["name"] != "CLUSTER_ID"]
    container["env"] += [
        {"name": "PGHOST", "value": "127.0.0.1"},
        {"name": "CONFIG_PATH", "value": "/opt/config/http-pull.json"},
        {"name": "DATA_PATH", "value": "/var/lib/postgresql/data"},
    ]
    container["env"] += [
        {
            "name": env,
            "valueFrom": {"secretKeyRef": {"name": name + "-tokens", "key": env}},
        }
        for env in sorted({t["token_env"] for t in config["targets"]})
    ]
    assert all(x["metadata"]["namespace"] == namespace for x in result)
    assert not any(
        x["kind"]
        in {
            "Namespace",
            "Role",
            "RoleBinding",
            "ClusterRole",
            "ClusterRoleBinding",
            "ServiceAccount",
        }
        for x in result
    )
    return {"apiVersion": "v1", "kind": "List", "items": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--collector-image", required=True)
    parser.add_argument("--postgres-image", default="postgres:16")
    parser.add_argument("--name", default="video-performance-http")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("http-pull.example.json"),
    )
    parser.add_argument("--queue-name")
    args = parser.parse_args()
    print(
        json.dumps(
            render(
                args.namespace,
                args.collector_image,
                json.loads(args.config.read_text()),
                args.name,
                args.postgres_image,
                args.queue_name,
            ),
            indent=2,
        )
    )

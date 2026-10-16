# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render namespaced resources to stdout; never apply or create namespaces."""

import argparse
import json
from pathlib import Path
from string import Template

import yaml
from collector import configuration, short_text

ROOT = Path(__file__).resolve().parent


def validate_name(value):
    configuration({"namespaces": [value]})
    return value


def render(
    namespace,
    cluster_id,
    collector_image,
    config,
    name="dingo-video-performance",
    postgres_image="postgres:16",
    queue_name=None,
):
    validate_name(namespace)
    validate_name(name)
    if len(name) > 53:
        raise ValueError("resource name exceeds suffix budget")
    config = configuration(config)
    values = {
        "name": name,
        "namespace": namespace,
        "cluster_id": json.dumps(short_text(cluster_id)),
        "collector_image": json.dumps(short_text(collector_image)),
        "postgres_image": json.dumps(short_text(postgres_image)),
        "storage_budget": str(config["storage_budget_bytes"]),
        "postgres_storage_limit": str(config["storage_budget_bytes"] + 2147483648),
        "config_json": json.dumps(json.dumps(config)),
    }
    items = []
    for filename in ("deployment.yaml", "configmap.yaml", "service.yaml"):
        template = Template((ROOT / "kubernetes" / filename).read_text())
        items.extend(yaml.safe_load_all(template.substitute(values)))
    for source_namespace in config["namespaces"]:
        template = Template((ROOT / "kubernetes" / "rbac.yaml").read_text())
        items.extend(
            yaml.safe_load_all(
                template.substitute(values, source_namespace=source_namespace)
            )
        )
    for item in items:
        item["metadata"].setdefault("labels", {})[
            "app.kubernetes.io/part-of"
        ] = "dingo-video-performance"
        if item["kind"] == "Deployment":
            pod = item["spec"]["template"]
            if queue_name:
                pod["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] = queue_name
            if namespace == "elm-test":
                pod["spec"].update(
                    tolerations=[{"operator": "Exists"}],
                    affinity={
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [
                                    {
                                        "matchExpressions": [
                                            {
                                                "key": "dc.com/osm.nodepool.type",
                                                "operator": "NotIn",
                                                "values": ["backup"],
                                            },
                                            {
                                                "key": "dc.com/openclaw",
                                                "operator": "DoesNotExist",
                                            },
                                        ]
                                    }
                                ]
                            }
                        }
                    },
                )
    return {"apiVersion": "v1", "kind": "List", "items": items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace", required=True, help="existing namespace hosting the collector"
    )
    parser.add_argument(
        "--cluster-id",
        required=True,
        help="stable regional identity stored with samples",
    )
    parser.add_argument(
        "--collector-image",
        required=True,
        help="image built with this directory's Dockerfile",
    )
    parser.add_argument("--postgres-image", default="postgres:16")
    parser.add_argument("--name", default="dingo-video-performance")
    parser.add_argument("--config", type=Path, default=ROOT / "config.example.json")
    parser.add_argument("--queue-name", help="optional elastic admission queue label")
    args = parser.parse_args()
    result = render(
        args.namespace,
        args.cluster_id,
        args.collector_image,
        json.loads(args.config.read_text()),
        args.name,
        args.postgres_image,
        args.queue_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

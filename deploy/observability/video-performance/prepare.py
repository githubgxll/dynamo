# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create credentials once in the explicitly selected existing namespace."""

import argparse
import base64
import json
import secrets
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--name", default="dingo-video-performance")
    args = parser.parse_args()
    # Reuse the renderer's validation without loading the collector's runtime.
    from render import validate_name

    validate_name(args.namespace)
    validate_name(args.name)
    if len(args.name) > 53:
        raise ValueError("resource name exceeds suffix budget")
    name = args.name + "-auth"
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            args.namespace,
            "get",
            "secret",
            name,
            "--ignore-not-found",
            "-o",
            "name",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        print("Existing credentials retained")
        return
    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "type": "Opaque",
        "metadata": {
            "name": name,
            "namespace": args.namespace,
            "labels": {"app.kubernetes.io/part-of": "dingo-video-performance"},
        },
        "data": {
            key: base64.b64encode(secrets.token_urlsafe(32).encode()).decode()
            for key in ("write-password", "read-password")
        },
    }
    subprocess.run(
        ["kubectl", "-n", args.namespace, "create", "-f", "-"],
        input=json.dumps(manifest),
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()

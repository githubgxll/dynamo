# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("performance_render", ROOT / "render.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def deployment(manifest):
    return next(item for item in manifest["items"] if item["kind"] == "Deployment")


def test_hosting_and_source_namespaces_have_separate_scopes():
    manifest = r.render("monitoring", "region-a", "collector:test", {})
    assert not any(
        item["kind"] in {"Namespace", "ClusterRole", "ClusterRoleBinding"}
        for item in manifest["items"]
    )
    for item in manifest["items"]:
        if item["kind"] in {"Role", "RoleBinding"}:
            assert item["metadata"]["namespace"] == "token-factory"
        else:
            assert item["metadata"]["namespace"] == "monitoring"
        if item["kind"] == "RoleBinding":
            assert item["subjects"][0]["namespace"] == "monitoring"


def test_multinamespace_rbac_is_read_only_and_deduplicated():
    manifest = r.render(
        "monitoring", "region-a", "collector:test", {"namespaces": ["a", "b", "a"]}
    )
    roles = [item for item in manifest["items"] if item["kind"] == "Role"]
    assert len(roles) == 2
    for role in roles:
        assert role["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]},
            {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
        ]


def test_empty_namespace_config_pauses_without_generated_roles():
    manifest = r.render("monitoring", "region-a", "collector:test", {"namespaces": []})
    assert not any(
        item["kind"] in {"Role", "RoleBinding"} for item in manifest["items"]
    )
    cm = next(item for item in manifest["items"] if item["kind"] == "ConfigMap")
    assert json.loads(cm["data"]["config.json"])["namespaces"] == []


def test_disk_budget_controls_volume_and_request_has_headroom():
    manifest = r.render(
        "monitoring", "region-a", "collector:test", {"storage_budget_bytes": 2147483648}
    )
    spec = deployment(manifest)["spec"]["template"]["spec"]
    assert spec["volumes"][0]["emptyDir"]["sizeLimit"] == "2147483648"
    assert (
        spec["containers"][0]["resources"]["limits"]["ephemeral-storage"]
        == "4294967296"
    )
    assert "subPath" not in json.dumps(spec)


def test_elm_test_profile_has_elastic_affinity_and_no_gpu_or_pinned_node():
    manifest = r.render(
        "elm-test",
        "region-a",
        "collector:test",
        {"namespaces": ["elm-test"]},
        name="collector-validation",
        queue_name="elastic-queue",
    )
    assert all(
        item["metadata"]["namespace"] == "elm-test" for item in manifest["items"]
    )
    pod = deployment(manifest)["spec"]["template"]
    assert pod["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "elastic-queue"
    spec = pod["spec"]
    assert spec["tolerations"] == [{"operator": "Exists"}]
    assert "nodeName" not in spec and "nodeSelector" not in spec
    assert "kubernetes.io/hostname" not in json.dumps(spec)
    assert "nvidia.com/gpu" not in json.dumps(spec)
    expressions = spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"]
    assert expressions == [
        {"key": "dc.com/osm.nodepool.type", "operator": "NotIn", "values": ["backup"]},
        {"key": "dc.com/openclaw", "operator": "DoesNotExist"},
    ]


def test_values_cannot_inject_yaml_fields():
    manifest = r.render(
        "monitoring", "region-a\nother: value", "image:test\nother: value", {}
    )
    spec = deployment(manifest)["spec"]["template"]["spec"]
    assert spec["containers"][1]["image"] == "image:test\nother: value"
    assert "other" not in spec["containers"][1]


@pytest.mark.parametrize("name", ["../bad", "-bad", "x" * 54])
def test_invalid_resource_names_fail(name):
    with pytest.raises(ValueError):
        r.render("monitoring", "region-a", "collector:test", {}, name=name)

# SPDX-License-Identifier: Apache-2.0
"""SGLang's runtime NIXL must supply both the Rust C API and UCX plugins."""

from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize(
    "framework, device, target, dynamic",
    [
        ("sglang", "cuda", "runtime", True),
        ("sglang", "cuda", "dev", None),
        ("sglang", "cuda", "local-dev", None),
        ("sglang", "xpu", "runtime", False),
        ("vllm", "cuda", "runtime", False),
        ("dynamo", "cuda", "runtime", False),
    ],
)
def test_runtime_wheel_nixl_linkage(framework, device, target, dynamic):
    root = Path(__file__).resolve().parents[2]
    template = Environment(
        loader=FileSystemLoader(root / "container"),
        undefined=StrictUndefined,
    ).get_template("templates/wheel_builder.Dockerfile")
    text = template.render(
        framework=framework,
        device=device,
        target=target,
        platform="amd64",
        cuda_version="13.0" if device == "cuda" else "",
        builder_image=None,
        make_efa=False,
        device_key="cuda13.0" if device == "cuda" else device,
        context=yaml.safe_load((root / "container/context.yaml").read_text()),
    )
    builds = [
        line
        for line in text.splitlines()
        if "maturin build --release --features" in line
    ]
    if dynamic is None:
        assert not builds  # Dev wheels are built later from the mounted checkout.
        return
    assert len(builds) >= 2
    assert all("--auditwheel skip --compatibility linux" in line for line in builds[:2])
    assert all(
        "--out /opt/dynamo/runtime-wheel-unrepaired" in line for line in builds[:2]
    )
    runtime_stage = text.split("FROM wheel_builder_base AS runtime_wheel_builder", 1)[1]
    runtime_stage = runtime_stage.split("Dynamo KVBM", 1)[0]
    if device == "cuda":
        assert "auditwheel repair --exclude 'lib*.so*'" in runtime_stage
        assert (
            "--wheel-dir /opt/dynamo/dist /opt/dynamo/runtime-wheel-unrepaired/*.whl"
            in runtime_stage
        )
    else:
        assert (
            "cp /opt/dynamo/runtime-wheel-unrepaired/*.whl /opt/dynamo/dist/"
            in runtime_stage
        )
    assert all(("nixl-dynamic" in line) is dynamic for line in builds[:2])

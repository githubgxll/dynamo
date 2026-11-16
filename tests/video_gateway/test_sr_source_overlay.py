# SPDX-License-Identifier: Apache-2.0
"""A Dingo source patch must not relocate the installed native SR runtime."""
import hashlib
from types import SimpleNamespace

from dingo.vllm.omni import seedvr2_process


def test_native_and_worker_sources_are_hashed_from_their_actual_roots(tmp_path):
    native, overlay = tmp_path / "installed", tmp_path / "worker-overlay"
    sources = [
        "vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py",
        "vllm_omni/diffusion/models/seedvr2/sr_telemetry.py",
        "vllm_omni/diffusion/ipc.py",
        "vllm_omni/diffusion/models/seedvr2/long_video.py",
        "vllm_omni/diffusion/models/seedvr2/long_video_support.py",
        "dingo/vllm/omni/seedvr2_handler.py",
        "dingo/video_gateway/adapters/h3_output_policy.py",
        "dingo/common/video_super_resolution.py",
        "dingo/vllm/omni/seedvr2_media.py",
    ]
    expected = {}
    for name in sources:
        actual_root = overlay if name.startswith("dingo/") else native
        path = actual_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("actual " + name)
        expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Stale installed Dingo files must not mask the effective overlay.
        if name.startswith("dingo/"):
            stale = native / name
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_text("stale installed source")
    actual = seedvr2_process.measurement_source_hashes(
        native, dingo_root=overlay, include_geometry=True
    )
    assert actual == expected


def test_runtime_resolver_uses_installed_native_distribution_and_effective_worker(
    monkeypatch, tmp_path
):
    installed, overlay = tmp_path / "installed", tmp_path / "overlay"
    seen = []
    monkeypatch.setattr(
        seedvr2_process.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda path: installed)
        if name == "vllm-omni"
        else None,
    )
    monkeypatch.setattr(
        seedvr2_process, "__file__", str(overlay / "dingo/vllm/omni/seedvr2_process.py")
    )

    def record(native, **kwargs):
        seen.append((native, kwargs))
        return {"proof": "hash"}

    monkeypatch.setattr(seedvr2_process, "measurement_source_hashes", record)
    assert seedvr2_process.runtime_measurement_source_hashes(include_geometry=True) == {
        "proof": "hash"
    }
    assert seen == [(installed, {"dingo_root": overlay, "include_geometry": True})]

"""Read-only Omni 0.30 source check; no model imports, network or GPU allocation.

--wheel FILE checks the reviewed release on Windows.
With no arguments run inside the target container, including through stdin.
Passing does NOT prove image digest, adapter activation, HSDP or GPU execution.
"""
from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

EXPECTED_WHEEL = "141cb0c7b9c07970e5c92a99aab9696e81682dbfa3d2bfcde359c3c1533385f4"
EXPECTED_IMAGE = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:aace31a2806c2c18535cb0bc6a81cc940e0901b4f02f380d2d880c1f34f965c2"
# Filled from the SHA256-verified official wheel by the local review.
EXPECTED_FILES = {
    "vllm_omni/diffusion/data.py": "0eeb59c91a0ab3a26b3980bb58b7306dce31323acc62f732edd78cdc56d6bf2d",
    "vllm_omni/diffusion/distributed/hsdp.py": "bdcab3cfc69831defed1c232968de6f5aae2c2103614b939463292583f2ff9af",
    "vllm_omni/diffusion/lora/__init__.py": "440002d1b29a38c00cfdadf0e0853252bbe761ab33a65ca0f77efe88a1cecacf",
    "vllm_omni/diffusion/lora/layers/__init__.py": "7c4b55ea06371baa7845cbe29d45317e8b5678ebbdc0215e263c70c3def3f366",
    "vllm_omni/diffusion/lora/layers/base_linear.py": "9baa38fca241a0b250a9b6f79ab3467da7385ec941b020f34680c451430d64eb",
    "vllm_omni/diffusion/lora/layers/column_parallel_linear.py": "df0006374d7736d92620605cd09011e49f3cda6d8dbc28fb28ae7ee273edcab7",
    "vllm_omni/diffusion/lora/layers/replicated_linear.py": "2cbf6396bdec27c117dc549a13a128b5d1270cc1908314ca3b5f4244009f7b61",
    "vllm_omni/diffusion/lora/layers/row_parallel_linear.py": "6bc6ddad9a1d897b482f7fc47e1ed3d276e67ac0f61e754d5226e88b2341b2dc",
    "vllm_omni/diffusion/lora/loader.py": "d58ec3768cb3711c6509bed03e7e63d949c19aa2f1d55ac4cd06bd9852cbaeb4",
    "vllm_omni/diffusion/lora/manager.py": "14d41a6e58edd7fbafdcbf835b19e9b90acb087fc533ec506ab8b3fe43192328",
    "vllm_omni/diffusion/lora/utils.py": "3e459ed862b3f77bd889db503b6b9f8d454524623d7432917f2d0f746521b792",
    "vllm_omni/diffusion/model_loader/diffusers_loader.py": "2d3994d56077e2af622287df2b8795fc32fc0f15f746a5869e207b02602bd358",
    "vllm_omni/diffusion/models/minimax_h3/__init__.py": "9a67e11005d020d7e3d7741c2a14c2785d0c77efc389467a4f19eea5cbe84f45",
    "vllm_omni/diffusion/models/minimax_h3/adaln_cache.py": "bae882af5022fbcf2e9d6f64353e4deab31a0572b76e8c10047880b1f81731dc",
    "vllm_omni/diffusion/models/minimax_h3/attention/__init__.py": "b1d617f45f34dbe6f5f58d792d90c84af8d8d0a7e7def77c48170ca93495ca0d",
    "vllm_omni/diffusion/models/minimax_h3/attention/fastvideo_h3.py": "f21648d3ba5b0bad4f5e083f19048dbc54dd2ab29043af6f776e6933d392338d",
    "vllm_omni/diffusion/models/minimax_h3/batched_packing.py": "566dd7dac8b1e1940679d90906bc88794c731bf78d1005136563ada6334010c7",
    "vllm_omni/diffusion/models/minimax_h3/chunked_decode.py": "dcfd5c146dd2814f3605e22ab7ac14046a4cf19c43909c3e2174a13c654f587d",
    "vllm_omni/diffusion/models/minimax_h3/condition_noise.py": "4b4d2d05e55124770fd288201cd1ebe612c9e6c647f0593a4d1381b5e175b108",
    "vllm_omni/diffusion/models/minimax_h3/continuation.py": "66c0b4495e53c53840c6789a154141b43bf738b5e86aab9165f8e9e80835bc0d",
    "vllm_omni/diffusion/models/minimax_h3/denoise_loop.py": "6b84912ccee35f1fd9079e3dabfcceedc122e5b5e9dbf98bf5b5cca37d4c8936",
    "vllm_omni/diffusion/models/minimax_h3/encoder.py": "f9074320d71c66fb8175a3c842d0181f7f6785822a5827a3a09f6ab7cefd8db8",
    "vllm_omni/diffusion/models/minimax_h3/fasth3.py": "fb7b11121f0582d7783f5b48ca7f9bb890c8ceae31f767ebc7f872a38069b80a",
    "vllm_omni/diffusion/models/minimax_h3/fasth3_checkpoint.py": "56d14ccb9d253c0f4b765ff0703555b21e3072dcbdfc00e0fb2133947f475589",
    "vllm_omni/diffusion/models/minimax_h3/latent_mask.py": "e4bd93347140ccb1714b7d61f3a1d99b8d19f4b983fed306066501e622dc9ab1",
    "vllm_omni/diffusion/models/minimax_h3/lora.py": "f811b4d754d295eebb0d6940f60af307e401da54f1adb12ae2dd97b942e24c2f",
    "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py": "53e875307d962be3225ed71f64d554dafece5587b2b0ce4c5e942afa7000e1c5",
    "vllm_omni/diffusion/models/minimax_h3/npu/__init__.py": "2209fb2e181ab6497b513d28c5e37479abc5e4899d229df096af29edc3b53a78",
    "vllm_omni/diffusion/models/minimax_h3/npu/lora.py": "a8673e58060c4d632bbc39fb7df049080e247fb23bf95a97959d8315d3fe2171",
    "vllm_omni/diffusion/models/minimax_h3/ops/__init__.py": "83bd517f4c8d9aa45e548d22572cb8b578ddc343c76de28362b476007a147ed5",
    "vllm_omni/diffusion/models/minimax_h3/ops/vae/__init__.py": "eb6d1bda0a5ed83ff1863ed4f3b75d417c71fb335027dadcf4ebba446380dabb",
    "vllm_omni/diffusion/models/minimax_h3/ops/vae/dispatch.py": "7c7df96a99d6e3bb63fdfe902c142bd78c9dbcb614d4b9b05601ce296a535f83",
    "vllm_omni/diffusion/models/minimax_h3/ops/vae/qk_norm_rope.py": "c1b8eae35dbc8d2acafb7ca2cdb5d0c8561a599df994427c4f88baaea346e7f5",
    "vllm_omni/diffusion/models/minimax_h3/ops/vae/scaled_residual.py": "c6abed0010610ed3ec68967782b3e111d5a2534ae04a296cced718036ed6a551",
    "vllm_omni/diffusion/models/minimax_h3/packed_sequence.py": "a206d2d2fb72a5b485f224e12a71ee867cef650e7b6fe52dabf575a9cd10acb9",
    "vllm_omni/diffusion/models/minimax_h3/packed_tokens.py": "20994e723054071e34acc41398e9d1636c226802e49894ac9162f5a808feb06f",
    "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py": "fda01a7ced807952bf80c3f9370d9dc0b4d77f960485b00de883abd974c952f0",
    "vllm_omni/diffusion/models/minimax_h3/quality_policy.py": "d5adacd0438200af8514bb7ff68280455b6a44bc94caaec59bfc241763461f81",
    "vllm_omni/diffusion/models/minimax_h3/scheduling_minimax_h3_euler_ancestral.py": "e075d34a5415b0e72e91cbf25cf90cddeb037ec9397a2f560387661f2f3b192e",
    "vllm_omni/diffusion/models/minimax_h3/temporal_chunks.py": "533771c28d043b0c78a6750f1bebf1e7e853f402fbb8b474f032d7ab07219e0e",
    "vllm_omni/diffusion/models/minimax_h3/time_request.py": "c20569799afd74c984e6a323a12c0a587dc6c70d84d375770c81cdccb503de9e",
    "vllm_omni/diffusion/models/minimax_h3/vae.py": "15c9f581233b124f32461cac29aaf10b180b5a1045b71d7b378ba69840d306f9",
    "vllm_omni/diffusion/models/minimax_h3/vae_temporal.py": "dcb7efe06564277da83450f106848179dd7a3ce07ba1644282cd7f8f3bbca518",
    "vllm_omni/diffusion/worker/diffusion_model_runner.py": "ada7e5876b814c41fb40e1f5fcba81de170d3427f5c1a2066a81fba4196a3fe0",
    "vllm_omni/diffusion/worker/diffusion_worker.py": "48e36974d51640e05017add752f56c3789c574671e715198a88f1ff784106f6f",
    "vllm_omni/entrypoints/cli/serve.py": "9b9f343467a612b636ec98f821282f3a2152007aa613baf1a6f25b8477f10f34",
    "vllm_omni/entrypoints/openai/api_server.py": "86a6f4f9c677ba3ea4ae7feda1ede072381f31bf0044f4e86baee8931ae7fc04",
    "vllm_omni/entrypoints/openai/protocol/videos.py": "a66c54ea33640f2e864e9b8160902717d3cf5254860eae38304e7a17e4ef6fb0",
    "vllm_omni/entrypoints/openai/serving_video.py": "62cf398d04380f05a03097eee8a1ebfbbb0d3a4654615445447209bcfd4735ac",
    "vllm_omni/entrypoints/openai/video/generation/helpers.py": "d8a16ed63e424d01ae1ed61f5f648588325e5278d0fd1be86bda1281dfc43ba5",
    "vllm_omni/lora/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "vllm_omni/lora/request.py": "7f5a4c31750e2bbd54adf614e09c6d50cc14daaf76f08d2ce5d19347dc8a64da",
    "vllm_omni/lora/utils.py": "747538f89eccc0d83a07f1c4daf7a0ce628b8bed0fcd1db05441febfdc32fe73"
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_files(read_bytes):
    rows, errors = {}, []
    for name, expected in EXPECTED_FILES.items():
        try:
            observed = digest(read_bytes(name))
            rows[name] = {"sha256": observed, "matches": observed == expected}
            if observed != expected:
                errors.append("Source mismatch: " + name)
        except (OSError, KeyError) as exc:
            rows[name] = {"error": type(exc).__name__}
            errors.append("Missing/unreadable source: " + name)
    if not EXPECTED_FILES:
        errors.append("Expected source inventory is empty")
    return rows, errors


def wheel_check(path):
    raw = Path(path).read_bytes()
    if digest(raw) != EXPECTED_WHEEL:
        return {"status": "FAIL", "errors": ["Official wheel SHA256 mismatch"]}
    with zipfile.ZipFile(path) as wheel:
        sources, errors = check_files(wheel.read)
        info = Parser().parsestr(wheel.read("vllm_omni-0.30.0.dist-info/METADATA").decode())
        if info["Version"] != "0.30.0":
            errors.append("Wrong wheel version")
    return {"status": "FAIL" if errors else "PASS_OFFLINE_WHEEL_ONLY",
            "wheel_sha256": EXPECTED_WHEEL, "sources": sources, "errors": errors,
            "limits": ["No installed Linux environment or GPU checked",
                       "No adapter weights checked", "No Harbor access or publication"]}


def runtime_check():
    errors, versions = [], {}
    for name in ("vllm", "vllm-omni", "torch", "diffusers", "transformers", "safetensors"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
            errors.append("Missing package: " + name)
    for name in ("vllm", "vllm-omni"):
        if versions[name] != "0.30.0":
            errors.append("Expected " + name + "==0.30.0")
    spec = importlib.util.find_spec("vllm_omni")
    if spec is None or spec.origin is None:
        return {"status": "FAIL", "versions": versions,
                "errors": errors + ["No effective vllm_omni package"]}
    root = Path(spec.origin).resolve().parent
    sources, source_errors = check_files(
        lambda name: (root / name.removeprefix("vllm_omni/")).read_bytes())
    errors.extend(source_errors)
    direct_requirements = []
    try:
        from packaging.requirements import Requirement
        distribution = metadata.distribution("vllm-omni")
        if Path(distribution.locate_file("vllm_omni")).resolve() != root:
            errors.append("Effective package is shadowing installed distribution")
        for value in distribution.requires or []:
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                version = metadata.version(requirement.name)
                ok = requirement.specifier.contains(version, prereleases=True)
            except metadata.PackageNotFoundError:
                version, ok = None, False
            direct_requirements.append({"requirement": value, "installed": version, "matches": ok})
            if not ok:
                errors.append("Unmet default requirement: " + value)
    except Exception as exc:
        errors.append("Direct dependency check failed: " + type(exc).__name__)
    info_dir = Path("/opt/dynamo/build-info")
    info = {p.name: digest(p.read_bytes()) for p in info_dir.glob("*.json") if p.is_file()}
    return {"status": "FAIL" if errors else "PASS_INSTALLED_CODE_ONLY",
            "expected_deployment_image": EXPECTED_IMAGE,
            "versions": versions, "effective_source_root": str(root),
            "sources": sources, "omni_direct_default_requirements": direct_requirements,
            "build_info_file_sha256": info, "errors": errors,
            "limits": ["Container image identity must be checked separately via Pod and registry metadata",
                       "No package import/ABI, GPU, HSDP, adapter activation or media test",
                       "Direct requirement metadata only; not a full transitive dependency audit",
                       "Adapter SHA256 and Base revision compatibility are separate checks"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel")
    args = parser.parse_args()
    try:
        result = wheel_check(args.wheel) if args.wheel else runtime_check()
    except Exception as exc:
        result = {"status": "FAIL", "errors": [type(exc).__name__ + ": " + str(exc)]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())

"""Regression tests for deliberate metadata exceptions and cv2 provenance.

These fixtures test rejection logic on Windows; installed Linux media checks
remain a separate Docker build gate.
"""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import audit_environment as overlay
import dependency_audit as audit
from test_dependency_audit import Distribution, LINUX, evidence, official_stack, KNOWN_UV_OUTPUT
from runtime_dependencies import runtime_requirements, snapshot


class GatewayOverlayTests(unittest.TestCase):
    def report(self, stack=None, *, env=None, root_extras=None):
        return audit.audit_dependencies(stack or official_stack(), python_version="3.12.3",
            environment=env or LINUX, nccl_evidence=evidence(), root_extras=root_extras)

    def test_only_reviewed_nccl_diagnostic_matches_both_metadata_and_uv(self):
        report = self.report()
        audit.review_uv_check(report, 1, KNOWN_UV_OUTPUT)
        self.assertEqual(report["status"], "PASS_WITH_UPSTREAM_NCCL_OVERRIDE", report["errors"])
        self.assertEqual(len(report["known_metadata_exceptions"]), 1)
        self.assertEqual(report["frameworks"]["vllm-omni"]["default_requirements"]["status"], "PASS")

    def test_pynv_override_from_abandoned_main_recipe_is_rejected(self):
        stack = official_stack()
        stack[0].requires += ["PyNvVideoCodec==2.0.4"]
        stack.append(Distribution("PyNvVideoCodec", "2.2.3"))
        report = self.report(stack)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("PyNvVideoCodec" in error for error in report["errors"]))
        stack[-1].version = "2.0.4"
        self.assertEqual(self.report(stack)["status"], "PASS_WITH_UPSTREAM_NCCL_OVERRIDE")

    def test_unknown_extra_uv_diagnostic_is_not_hidden(self):
        extra = "The package `vllm` requires `pynvvideocodec==2.0.4`, but `2.2.3` is installed\n"
        for output in (KNOWN_UV_OUTPUT + "Unexpected error\n", KNOWN_UV_OUTPUT + extra,
                       KNOWN_UV_OUTPUT.replace("Found 1", "Found 2")):
            report = self.report(); audit.review_uv_check(report, 1, output)
            self.assertEqual(report["status"], "FAIL")

    def test_gateway_reference_has_to_be_exact_and_current(self):
        policy = overlay.read_policy()
        valid = overlay.reference_evidence(policy, gateway_sha=policy["gateway_sha"])
        self.assertTrue(valid["verified"], valid)
        self.assertFalse(overlay.reference_evidence(policy, gateway_sha="wrong")["verified"])
        wrong = json.loads(json.dumps(policy))
        wrong["gateway_reference_sha256"]["reference-gateway-pyproject.toml"] = "0" * 64
        self.assertFalse(overlay.reference_evidence(wrong, gateway_sha=policy["gateway_sha"])["verified"])

    def test_requested_gateway_extra_is_audited_recursively(self):
        root = Distribution("ai-dingo", "1.3.0", [
            'av>=14,<17; extra == "video-gateway"',
            'PyYAML>=6,<7; extra == "video-gateway"',
            'nixl==1.3.1; extra == "vllm"',
        ])
        stack = official_stack() + [root, Distribution("av", "16.1.0"), Distribution("PyYAML", "6.0.3")]
        report = self.report(stack, root_extras={"ai-dingo": ["video-gateway"]})
        self.assertEqual(report["status"], "PASS_WITH_UPSTREAM_NCCL_OVERRIDE", report["errors"])
        self.assertEqual(report["active_extras"]["ai-dingo"], ["video-gateway"])
        self.assertEqual(report["frameworks"]["ai-dingo"]["recursive_dependency_closure"]["status"], "PASS")
        stack[-2].version = "18.0.0"
        self.assertEqual(self.report(stack, root_extras={"ai-dingo": ["video-gateway"]})["status"], "FAIL")
        self.assertEqual(self.report(stack[1:], root_extras={"missing": ["video-gateway"]})["status"], "FAIL")

    def test_missing_reference_is_reported_as_packaging_error_not_version_conflict(self):
        policy = overlay.read_policy()
        with tempfile.TemporaryDirectory() as temporary, patch.object(overlay, "HERE", Path(temporary)):
            result = overlay.reference_evidence(policy, gateway_sha=policy["gateway_sha"])
        self.assertFalse(result["verified"])
        self.assertTrue(all(value is None for value in result["reference_sha256"].values()))
        self.assertTrue(all("is missing" in error and "COPY" in error for error in result["errors"]))

    def test_inherited_override_file_is_not_a_general_override_channel(self):
        overlay.check_override_file("# official\nnvidia-nccl-cu13==2.30.7\n")
        for text in ("", "nvidia-nccl-cu13==2.29.7", "nvidia-nccl-cu13==2.30.7\nfoo==1", "nvidia-nccl-cu13==2.30.7\nnvidia-nccl-cu13==2.30.7"):
            with self.assertRaises(RuntimeError): overlay.check_override_file(text)

    def test_profile_catches_missing_dingo_missing_gms_and_unwanted_plugins(self):
        policy = overlay.read_policy()
        report = {"packages": policy["required_versions"].copy(), "errors": [], "known_metadata_exceptions": []}
        overlay.check_versions(report, policy); self.assertEqual(report["status"], "PASS")
        for change in ("ai-dingo", "gpu-memory-service", "vllm-omni"):
            report = {"packages": policy["required_versions"].copy(), "errors": [], "known_metadata_exceptions": []}
            report["packages"].pop(change)
            overlay.check_versions(report, policy); self.assertEqual(report["status"], "FAIL")
        report = {"packages": {**policy["required_versions"], "kvbm": "1.6.0"}, "errors": [], "known_metadata_exceptions": []}
        overlay.check_versions(report, policy); self.assertEqual(report["status"], "FAIL")

    def test_wheel_origin_checks_both_url_and_checksum(self):
        policy = overlay.read_policy()
        url, digest = policy["omni_wheel"].split("#sha256=")
        good = {"url": url, "archive_info": {"hashes": {"sha256": digest}}}
        with patch.object(overlay.md, "distribution", return_value=SimpleNamespace(read_text=lambda _: json.dumps(good))):
            self.assertEqual(overlay.wheel_origin_evidence(policy)["sha256"], digest)
        for bad in ({**good, "url": "https://different.example/wheel.whl"},
                    {"url": url, "archive_info": {"hashes": {"sha256": "0" * 64}}}):
            with patch.object(overlay.md, "distribution", return_value=SimpleNamespace(read_text=lambda _: json.dumps(bad))):
                with self.assertRaises(RuntimeError): overlay.wheel_origin_evidence(policy)

    def test_gateway_backend_helpers_are_required_without_activating_nixl_extra(self):
        policy = overlay.read_policy()
        packages = {"uvloop": "0.22.1", "blake3": "1.0.8", "soundfile": "0.13.1", "librosa": "0.11.0", "ray": "2.55.0"}
        report = {"packages": packages, "errors": [], "known_metadata_exceptions": []}
        overlay.check_backend_helpers(report, policy)
        self.assertEqual(report["status"], "PASS")
        report = {"packages": {**packages, "ray": "2.54.0"}, "errors": [], "known_metadata_exceptions": []}
        overlay.check_backend_helpers(report, policy)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(any("nixl" in item for item in policy["additional_backend_requirements"]))

    def test_both_opencv_default_requirements_stay_in_graph(self):
        records = [Distribution("vllm", "0.31.0", ["opencv-python-headless>=4.13.0"]),
                   Distribution("vllm-omni", "0.31.0rc1", ["cosmos-guardrail>=0.3.1"]),
                   Distribution("cosmos-guardrail", "0.3.1", ["opencv-python>=4.10.0.84"]),
                   Distribution("opencv-python-headless", "4.13.0.92"),
                   Distribution("opencv-python", "4.13.0.92")]
        self.assertEqual(audit.audit_dependencies(records, python_version="3.12.3", environment=LINUX)["status"], "PASS")
        for index in (3, 4):
            removed = records.copy(); removed.pop(index)
            self.assertEqual(audit.audit_dependencies(removed, python_version="3.12.3", environment=LINUX)["status"], "FAIL")

    def test_same_version_cv2_actual_provider_is_verified_and_tamper_fails(self):
        class Record(str): pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for name in ("cv2/__init__.py", "cv2/cv2.abi3.so"):
                data = ("fixture " + name).encode()
                path = root / name; path.parent.mkdir(exist_ok=True); path.write_bytes(data)
                record = Record(name)
                record.hash = SimpleNamespace(mode="sha256", value=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode())
                paths.append(record)
            dist = SimpleNamespace(files=paths, locate_file=lambda name: root / name)
            with patch.object(overlay.md, "version", return_value="4.13.0.92"), patch.object(overlay.md, "distribution", return_value=dist), patch.object(overlay.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(root / "cv2/__init__.py"))), patch.dict("sys.modules", {"cv2": SimpleNamespace(__version__="4.13.0")}):
                result = overlay.cv2_provider_evidence(overlay.read_policy())
                self.assertEqual(len(result["verified_cv2_file_sha256"]), 2)
                (root / "cv2/cv2.abi3.so").write_bytes(b"overwritten")
                with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                    overlay.cv2_provider_evidence(overlay.read_policy())


if __name__ == "__main__":
    unittest.main()

"""SR-only contract/lifecycle refusals using no model or external services."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import contracts
import sr_only
from media import write_new_json


class SRContractTests(unittest.TestCase):
    def test_c14_full_timeline(self):
        self.assertEqual(contracts.PROFILE["input_frames"], 362)
        self.assertEqual(contracts.PROFILE["sr_size"], [1840, 1104])
        self.assertEqual(contracts.PROFILE["capture_window_starts"], [130, 143, 156, 169, 182])

    def test_launcher_never_loads_h3(self):
        command = sr_only.service_command("/model/seedvr2-3b")
        self.assertEqual(command[:3], ["vllm-omni", "serve", "/model/seedvr2-3b"])
        self.assertEqual(command[command.index("--model-class-name") + 1], "SeedVR2Pipeline")
        self.assertEqual(command[command.index("--num-gpus") + 1], "8")
        self.assertEqual(command[command.index("--cache-backend") + 1], "none")
        self.assertIn("spatial_shard_height", command[command.index("--stage-overrides") + 1])

    def test_environment_ignores_unreviewed_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"VLLM_OMNI_SEEDVR2_MAX_FRAMES": "65", "PYTHONPATH": "/untrusted", "WORLD_SIZE": "4"}):
                env = sr_only.service_environment(Path(temporary))
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("WORLD_SIZE", env)
            self.assertEqual(env["VLLM_OMNI_SEEDVR2_MAX_FRAMES"], "17")
            self.assertEqual(env["VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS"], "64461824")
            self.assertEqual(env["SEEDVR2_C14_PROFILE"], str(Path(temporary) / "capture-profile.json"))

    def test_media_requires362_not360(self):
        proof = {"width": 1280, "height": 768, "frames": 362, "fps": 24, "full_decode": True}
        contracts.validate_media(proof, [1280, 768])
        proof["frames"] = 360
        with self.assertRaises(ValueError):
            contracts.validate_media(proof, [1280, 768])

    def test_native_loopback_bypasses_inherited_proxy(self):
        # The runtime is Linux; Windows os.environ folds upper/lowercase keys.
        with tempfile.TemporaryDirectory() as temp, patch.object(os, "environ", {
                "HTTP_PROXY": "http://proxy.internal:8080", "NO_PROXY": ".svc", "no_proxy": "registry.internal"}):
            env = sr_only.service_environment(Path(temp))
            self.assertEqual(env["HTTP_PROXY"], "http://proxy.internal:8080")
            self.assertEqual(env["NO_PROXY"], env["no_proxy"])
            self.assertEqual(set(env["no_proxy"].split(",")), {".svc", "registry.internal", "localhost", "127.0.0.1", "::1"})

    def test_release_rejects_other_recipe_and_tag_only(self):
        good = {"schema": "seedvr2-c14-publication/1", "status": "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED",
                "source_commit": "a" * 40, "parent_image": contracts.PARENT,
                "deployment_image": contracts.PARENT.split("@")[0] + "@sha256:" + "b" * 64}
        self.assertEqual(contracts.validate_release(good, "a" * 40), good["deployment_image"])
        for change in ({"schema": "omni-gateway-lite-publication/1"}, {"source_commit": "c" * 40},
                       {"deployment_image": "registry.example/image:latest"}, {"parent_image": "other"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                contracts.validate_release({**good, **change}, "a" * 40)

    def test_no_overwrite_existing_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "record.json"
            write_new_json(path, {"original": True})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"original": False})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_wrong_baseline_hash_fails_before_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps({"schema": "seedvr2-c14-frozen-baseline/1",
                "status": "LOCAL_FROZEN_VERIFIED", "image": contracts.PARENT, "parent_base_sha256": "0" * 64}))
            (root / "COPY-STATE.json").write_text('{"status":"LOCAL_FROZEN_VERIFIED"}')
            with self.assertRaisesRegex(ValueError, "Wrong parent"):
                contracts.validate_baseline(root)
            self.assertEqual(len(list(root.iterdir())), 2)

    def test_incomplete_baseline_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text('{"schema":"seedvr2-c14-frozen-baseline/1","status":"LOCAL_FROZEN_VERIFIED"}')
            (root / "COPY-STATE.json").write_text('{"status":"COPYING_DO_NOT_USE"}')
            with self.assertRaisesRegex(ValueError, "incomplete"):
                contracts.validate_baseline(root)

    def test_changed_model_revision_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "conversion-manifest.json").write_text(json.dumps({"schema": "h3-seedvr2-fp16-conversion/1",
                "contract_version": 1, "status": "VERIFIED", "source": {"revision": "other"}}))
            with self.assertRaisesRegex(ValueError, "revision/source"):
                contracts.validate_model(root)

    def test_stop_rejects_reused_supervisor(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "attempt").mkdir()
            (run / "attempt/supervisor.json").write_text('{"pid":42,"pgid":42,"start_ticks":1}')
            with patch.object(sr_only, "check_run"), patch.object(sr_only, "same_process", return_value=False), patch.object(sr_only.os, "kill") as kill:
                with self.assertRaisesRegex(ValueError, "not alive"):
                    sr_only.stop_run(run)
                kill.assert_not_called()

    def test_process_fingerprint_requires_birth_and_group(self):
        expected = {"pid": 44, "pgid": 44, "start_ticks": 123}
        for actual in (None, {**expected, "state": "Z"}, {**expected, "state": "S", "start_ticks": 124},
                       {**expected, "state": "S", "pgid": 45}):
            with patch.object(sr_only, "proc", return_value=actual):
                self.assertFalse(sr_only.same_process(expected))

    def test_missing_birth_never_reports_live_child_stopped(self):
        child = Mock(pid=44)
        child.poll.return_value = None
        with patch.object(sr_only, "members", return_value=[]), self.assertRaisesRegex(ValueError, "UNCONFIRMED"):
            sr_only.cleanup_engine(child, None)
        child.wait.assert_not_called()

    def test_missing_birth_only_accepts_reaped_empty_group(self):
        child = Mock(pid=44)
        child.poll.return_value = 1
        with patch.object(sr_only, "members", return_value=[]):
            sr_only.cleanup_engine(child, None)
        child.wait.assert_called_once()
        with patch.object(sr_only, "members", return_value=[{"pid": 45}]), self.assertRaisesRegex(ValueError, "UNCONFIRMED"):
            sr_only.cleanup_engine(child, None)

    def test_final_encoding_preserves_audio_and_frames(self):
        argv = sr_only.mux_command("raw.mp4", "base.mp4", "final.mp4")
        self.assertEqual(argv[argv.index("-c:a") + 1], "copy")
        self.assertNotIn("-shortest", argv)
        self.assertNotIn("-t", argv)
        self.assertNotIn("-y", argv)
        self.assertIn("1800:1080", argv[argv.index("-vf") + 1])

    def test_help_has_no_framework_requirement(self):
        result = subprocess.run([sys.executable, "-B", str(HERE / "sr_only.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("import-base", result.stdout)
        self.assertNotIn("generate-h3", result.stdout)


if __name__ == "__main__":
    unittest.main()

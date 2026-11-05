"""Profile/lifecycle CPU checks; no native server or CUDA process is started."""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from service import launcher, profile, state, worker_extension as worker


def read_profile(name):
    return json.loads((ROOT / "tests/fixtures/service" / (name + ".json")).read_text(encoding="utf-8"))


class ProfileContract(unittest.TestCase):
    def test_both_roles_disable_native_stats_and_peak_in_off_mode(self):
        for name in ("fl2va-sr", "ref2va-sr"):
            config = read_profile(name)
            for role in ("h3", "sr"):
                argv = profile.argv(config, role)
                self.assertIn("--no-collect-stage-metrics", argv)
                self.assertIn("--disable-log-stats", argv)
                self.assertIn("--no-record-output-peak-memory", argv)
                self.assertNotIn("--enable-diffusion-pipeline-profiler", argv)
                self.assertNotIn("--enable-cache-dit-summary", argv)

    def test_stages_mode_enables_only_required_timing_controls(self):
        config = read_profile("ref2va-sr")
        config["monitoring"]["mode"] = "stages"
        for role in ("h3", "sr"):
            argv = profile.argv(config, role)
            self.assertIn("--collect-stage-metrics", argv)
            self.assertIn("--disable-log-stats", argv)
            self.assertIn("--no-record-output-peak-memory", argv)
            self.assertEqual("--enable-diffusion-pipeline-profiler" in argv, role == "h3")

    def test_fl_has_no_offload_or_ref_worker_extension(self):
        config = read_profile("fl2va-sr")
        argv = profile.argv(config, "h3")
        self.assertFalse(any("offload" in value for value in argv))
        sr = profile.argv(config, "sr")
        override = json.loads(sr[sr.index("--stage-overrides") + 1])
        self.assertNotIn("worker_extension_cls", override["0"])

    def test_ref_maps_only_legacy_extension_alias_to_production_class(self):
        config = read_profile("ref2va-sr")
        argv = profile.argv(config, "sr")
        override = json.loads(argv[argv.index("--stage-overrides") + 1])
        self.assertEqual(override["0"]["worker_extension_cls"], profile.EXTENSION)

    def test_conflicting_extension_and_profile_scope_are_rejected(self):
        config = read_profile("fl2va-sr"); config["handoff"]["enabled"] = True
        with self.assertRaises(ValueError): profile.argv(config, "h3")
        config = read_profile("ref2va-sr")
        pos = config["sr"]["native_argv"].index("--stage-overrides") + 1
        overrides = json.loads(config["sr"]["native_argv"][pos]); overrides["0"]["worker_extension_cls"] = "other.Extension"
        config["sr"]["native_argv"][pos] = json.dumps(overrides)
        with self.assertRaises(ValueError): profile.argv(config, "sr")

    def test_sanitization_preserves_assigned_cuda_and_removes_diagnostic_state(self):
        for name in ("fl2va-sr", "ref2va-sr"):
            with tempfile.TemporaryDirectory() as temporary:
                config = read_profile(name)
                for role in ("h3", "sr"):
                    for key, value in list(config[role]["stage_environment"].items()):
                        if value.startswith("/work/"):
                            config[role]["stage_environment"][key] = str(Path(temporary) / role / key)
                    env = profile.environment(config, role, profile_path=Path(temporary) / "profile.json",
                        epoch="service-epoch", root=Path(temporary), inherited={"CUDA_VISIBLE_DEVICES": "pod-assigned",
                            "OMNI_MEMDIAG_ENABLE": "1", "H3_OFFLOAD_TIMING": "1", "PYTHONPATH": "old-overlay",
                            "RANK": "9", "H3SR_MEMORY_RUN_ID": "old", "SEEDVR2_METRICS_DIR": "old"})
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "pod-assigned")
                    self.assertNotIn("OMNI_MEMDIAG_ENABLE", env)
                    self.assertNotIn("RANK", env)
                    self.assertNotIn("SEEDVR2_METRICS_DIR", env)
                    if role == "sr" or name == "fl2va-sr":
                        self.assertFalse(any(k.startswith("H3_OFFLOAD_") for k in env))
                    self.assertFalse((Path(temporary) / "memdiag").exists())

    def test_sanitization_preserves_only_the_two_baked_image_identities(self):
        identities = {"H3_SR_PRODUCTION_SOURCE_COMMIT": "frozen-image-source",
                      "H3_SR_PRODUCTION_PARENT_IMAGE": "registry/image@sha256:parent"}
        for name in ("fl2va-sr", "ref2va-sr"):
            for role in ("h3", "sr"):
                config = read_profile(name)
                config[role]["stage_environment"] = {}
                config["common_environment"]["H3_SR_PRODUCTION_SOURCE_COMMIT"] = "profile-cannot-replace-image"
                env = profile.environment(config, role, profile_path="profile.json", epoch="epoch", root="control",
                    inherited={**identities, "H3_SR_PRODUCTION_OLD_RUN": "stale", "H3_DEBUG_RUN": "stale"})
                for key, value in identities.items():
                    self.assertEqual(env[key], value)
                self.assertNotIn("H3_SR_PRODUCTION_OLD_RUN", env)
                self.assertNotIn("H3_DEBUG_RUN", env)


class StartupContract(unittest.TestCase):
    def supervisor(self, temporary, name="fl2va-sr"):
        profile_path = Path(temporary) / "profile.json"
        profile_path.write_text(json.dumps(read_profile(name)), encoding="utf-8")
        command_path = Path(temporary) / "gateway-command.json"
        command_path.write_text(json.dumps(["--model", "base", "--h3-sr-profile", str(profile_path),
            "--request-adapter", "minimax_h3", "--request-adapter-workflow", name.removesuffix("-sr"),
            "--max-num-seqs", "1"]), encoding="utf-8")
        return launcher.Supervisor(profile_path, gateway_command_file=command_path, root=Path(temporary) / "control", readiness_port=0)

    def test_auto_start_orders_h3_then_sr_before_ready_without_memdiag(self):
        for name in ("fl2va-sr", "ref2va-sr"):
            with tempfile.TemporaryDirectory() as temporary:
                supervisor = self.supervisor(temporary, name)
                stages = []
                def start(role):
                    self.assertFalse(supervisor.state["service_ready"])
                    stages.append(role)
                def gateway():
                    self.assertTrue(supervisor.state["service_ready"])
                    self.assertFalse(supervisor.state["gateway_ready"])
                    self.assertEqual(supervisor.state["phase"], "loading-gateway")
                    self.assertFalse(supervisor.is_ready())
                    stages.append("gateway")
                identity = {"ranks": [{"rank": r} for r in range(8)]}
                with mock.patch.object(supervisor, "_preflight"), mock.patch.object(supervisor, "_start", side_effect=start), \
                        mock.patch.object(supervisor, "_start_gateway", side_effect=gateway), \
                        mock.patch.object(launcher.HTTPControlClient, "identities", new=mock.AsyncMock(return_value=identity)):
                    supervisor.start()
                self.assertEqual(stages, ["h3", "sr", "gateway"])
                self.assertTrue(supervisor.state["service_ready"])
                self.assertFalse((Path(temporary) / "memdiag").exists())
                self.assertEqual("sr_worker_identities" in supervisor.state, name == "ref2va-sr")

    def test_existing_unknown_lease_prevents_start_without_deleting_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = self.supervisor(temporary)
            path = supervisor.root / "active.json"
            state.atomic_json(path, {"action_id": "old-unknown"})
            with mock.patch.object(supervisor, "_start") as start:
                with self.assertRaises(RuntimeError): supervisor.start()
            start.assert_not_called()
            self.assertTrue(path.exists())

    def test_preflight_subprocesses_receive_both_image_baked_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = self.supervisor(temporary)
            config = supervisor.profile
            config["h3"]["stage_environment"] = {}
            identities = {"H3_SR_PRODUCTION_SOURCE_COMMIT": "frozen-source",
                          "H3_SR_PRODUCTION_PARENT_IMAGE": "registry/image@sha256:parent"}
            with mock.patch.dict(launcher.os.environ, identities, clear=True), \
                    mock.patch.object(launcher.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
                supervisor._preflight()
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                for name, value in identities.items():
                    self.assertEqual(call.kwargs["env"][name], value)

    def test_stop_signals_only_owned_groups_and_checks_birth(self):
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = self.supervisor(temporary)
            process = mock.Mock(pid=1234); process.poll.return_value = None; process.wait.return_value = 0
            supervisor.children = {"h3": process}
            birth = {"pid": 1234, "process_start_ticks": 99}
            identity = {"birth": birth, "pgid": 1234, "sid": 1234}
            supervisor.state["children"] = {"h3": {"pid": 1234, "birth": birth, "pgid": 1234, "sid": 1234}}
            with mock.patch.object(launcher, "process_group_identity", return_value=identity), \
                    mock.patch.object(launcher, "process_group_members", return_value={"1234": identity}), \
                    mock.patch.object(launcher.os, "killpg", create=True) as kill, \
                    mock.patch.object(launcher.signal, "SIGKILL", 9, create=True):
                supervisor.stop_owned()
            self.assertEqual([call.args[0] for call in kill.call_args_list], [1234, 1234])
            changed = {**identity, "birth": {"pid": 1234, "process_start_ticks": 100}}
            with mock.patch.object(launcher, "process_group_identity", return_value=changed), \
                    mock.patch.object(launcher.os, "killpg", create=True) as kill:
                with self.assertRaises(RuntimeError): supervisor.stop_owned()
            kill.assert_not_called()

    def test_operational_http_server_binds_only_loopback(self):
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = self.supervisor(temporary)
            with mock.patch.object(launcher, "ThreadingHTTPServer") as server, \
                    mock.patch.object(launcher.threading, "Thread") as thread:
                result = supervisor._server()
            self.assertEqual(server.call_args.args[0], ("127.0.0.1", 0))
            self.assertIs(result, server.return_value)
            thread.return_value.start.assert_called_once()


if __name__ == "__main__": unittest.main()

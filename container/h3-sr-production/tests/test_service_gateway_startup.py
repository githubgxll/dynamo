"""Three actual CPU child processes; Linux identities/signals remain doubles."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from service import launcher, profile, state


def prepare(root, name="fl2va-sr"):
    root = Path(root)
    config = json.loads((ROOT / "tests/fixtures/service" / (name + ".json")).read_text(encoding="utf-8"))
    for role in ("h3", "sr"):
        for key, value in list(config[role]["stage_environment"].items()):
            if value.startswith("/work/"):
                config[role]["stage_environment"][key] = str(root / role / key)
    path = root / "profile.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    argv = ["--model", config["h3"]["native_argv"][2], "--h3-sr-profile", str(path),
        "--request-adapter", "minimax_h3", "--request-adapter-workflow", name.removesuffix("-sr"),
        "--max-num-seqs", "1", "--namespace", "cpu-test", "--component", "video", "--endpoint", "generate"]
    command = root / "gateway-command.json"
    command.write_text(json.dumps(argv), encoding="utf-8")
    return config, path, command, argv


def birth(pid):
    return {"pid": pid, "process_start_ticks": pid, "pid_namespace": "pid:[cpu-double]", "boot_id": "cpu-test"}


CPU_CHILD = r'''
from pathlib import Path
import hashlib,json,os,sys,time
role,location,bad=sys.argv[1:]
root=Path(location)
if role == 'gateway':
    observed=json.loads(os.environ.get('H3SR_CPU_PARENT_STATE') or (root/'state.json').read_text())
    (root/'gateway-observed.json').write_text(json.dumps(observed))
    receipt={'schema':'h3sr-gateway-registration/1','status':'REGISTERED',
        'service_epoch':os.environ['H3SR_SERVICE_EPOCH'],
        'profile_sha256':hashlib.sha256(Path(os.environ['H3SR_PROFILE_PATH']).read_bytes()).hexdigest(),
        'pid':os.getpid(),'birth':{'pid':os.getpid(),'process_start_ticks':os.getpid(),
            'pid_namespace':'pid:[cpu-double]','boot_id':'cpu-test'}}
    if bad == 'epoch': receipt['service_epoch']='stale-epoch'
    temporary=root/'gateway-ready.tmp'
    temporary.write_text(json.dumps(receipt))
    temporary.replace(root/'gateway-ready.json')
else:
    (root/(role+'-cpu-ready')).write_text('ready')
while True: time.sleep(.1)
'''


class GatewayConfiguration(unittest.TestCase):
    def test_command_is_fixed_module_arguments_and_contract_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, path, _, valid = prepare(temporary)
            self.assertEqual(profile.gateway_argv(config, path, valid), valid)
            bad_commands = ["python -m evil", ["python", "evil.py", *valid],
                [*valid, "--h3-sr-profile", str(path)], [*valid, "--num-gpus", "8"]]
            for flag, value in (("--h3-sr-profile", str(Path(temporary) / "other.json")),
                ("--request-adapter", "other"), ("--request-adapter-workflow", "ref2va"), ("--max-num-seqs", "2")):
                bad = list(valid); bad[bad.index(flag) + 1] = value; bad_commands.append(bad)
            for argv in bad_commands:
                with self.subTest(argv=argv), self.assertRaises(ValueError):
                    profile.gateway_argv(config, path, argv)

    def test_gateway_environment_keeps_dingo_connections_and_exact_deployed_digest_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, path, _, _ = prepare(temporary)
            digest = "registry.example/project/image@sha256:" + "a" * 64
            inherited = {"DYN_SYSTEM_ENABLED": "1", "NATS_SERVER": "nats://service:4222", "ETCD_ENDPOINTS": "http://etcd:2379",
                "CUDA_VISIBLE_DEVICES": "assigned", "H3_OFFLOAD_TIMING": "1", "OMNI_MEMDIAG_ENABLE": "1", "RANK": "7",
                "H3SR_IMAGE_DIGEST": digest, "H3SR_OLD_RUN": "old", "H3SR_SERVICE_EPOCH": "old",
                "H3_SR_PRODUCTION_SOURCE_COMMIT": "source", "H3_SR_PRODUCTION_PARENT_IMAGE": "parent"}
            config["h3"]["stage_environment"]["MODEL_STAGE_ONLY"] = "must-not-leak"
            env = profile.gateway_environment(config, profile_path=path, epoch="new", root=temporary, inherited=inherited)
            for key in ("DYN_SYSTEM_ENABLED", "NATS_SERVER", "ETCD_ENDPOINTS", "CUDA_VISIBLE_DEVICES", "H3SR_IMAGE_DIGEST"):
                self.assertEqual(env[key], inherited[key])
            for key in ("MODEL_STAGE_ONLY", "H3_OFFLOAD_TIMING", "OMNI_MEMDIAG_ENABLE", "RANK", "H3SR_OLD_RUN"):
                self.assertNotIn(key, env)
            self.assertEqual(env["H3SR_SERVICE_EPOCH"], "new")
            model_env = profile.environment(config, "sr", profile_path=path, epoch="new", root=temporary, inherited=inherited)
            self.assertNotIn("H3SR_IMAGE_DIGEST", model_env)

    def test_stages_requires_actual_repo_digest_without_parent_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, path, _, _ = prepare(temporary)
            config["monitoring"]["mode"] = "stages"
            for deployed in (None, "repo:tag", "sha256:" + "a" * 64):
                inherited = {"H3_SR_PRODUCTION_PARENT_IMAGE": "registry/base@sha256:" + "b" * 64}
                if deployed is not None: inherited["H3SR_IMAGE_DIGEST"] = deployed
                with self.subTest(deployed=deployed), self.assertRaises(ValueError):
                    profile.gateway_environment(config, profile_path=path, epoch="new", root=temporary, inherited=inherited)
            full = "registry/project/image@sha256:" + "c" * 64
            env = profile.gateway_environment(config, profile_path=path, epoch="new", root=temporary,
                inherited={"H3SR_IMAGE_DIGEST": full})
            self.assertEqual(env["H3SR_IMAGE_DIGEST"], full)

    def test_cli_requires_gateway_command_file(self):
        with mock.patch.object(sys, "argv", ["launcher.py"]), mock.patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                launcher.main()
        self.assertEqual(error.exception.code, 2)


class ThreeOwnedChildren(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.children, self.commands, self.events, self.streams = {}, [], [], []
        self.bad_receipt = "none"
        self.fail_gateway_spawn = False
        self.exit_h3_before_registration = False
        self.real_popen = subprocess.Popen
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.emergency_cleanup)
        self.stack.enter_context(mock.patch.object(launcher.subprocess, "Popen", side_effect=self.spawn_cpu))
        self.stack.enter_context(mock.patch.object(launcher, "process_birth", side_effect=birth))
        self.stack.enter_context(mock.patch.object(launcher, "process_group_identity", side_effect=self.group_identity))
        self.stack.enter_context(mock.patch.object(launcher, "process_group_members", side_effect=self.group_members))
        self.stack.enter_context(mock.patch.object(launcher.os, "killpg", side_effect=self.signal_group, create=True))
        self.stack.enter_context(mock.patch.object(launcher.signal, "signal"))
        self.stack.enter_context(mock.patch.object(launcher.signal, "SIGKILL", 9, create=True))

    def supervisor(self, name="fl2va-sr"):
        _, path, command, _ = prepare(self.root, name)
        result = launcher.Supervisor(path, gateway_command_file=command, root=self.root / "control", readiness_port=0)
        self.stack.enter_context(mock.patch.object(result, "_preflight"))
        self.stack.enter_context(mock.patch.object(result, "_wait_health", side_effect=lambda role, _port: self.wait_marker(role)))
        return result

    def spawn_cpu(self, command, **kwargs):
        role = "gateway" if command[2:4] == ["-m", "dingo.vllm.omni"] else command[command.index("--role") + 1]
        if role == "gateway" and self.fail_gateway_spawn:
            raise OSError("gateway spawn refused")
        self.commands.append((role, command, kwargs["env"]))
        self.streams.append(kwargs["stdout"])
        self.events.append("spawn-" + role)
        if role == "gateway" and self.exit_h3_before_registration:
            h3 = next(child for child in self.children.values())
            h3.terminate(); h3.wait(timeout=5)
        if role == "gateway" and os.name == "nt":
            # Windows read handles can block atomic replace. This CPU-only
            # child receives the already-published state snapshot at spawn.
            kwargs["env"] = {**kwargs["env"], "H3SR_CPU_PARENT_STATE":
                             (self.root / "control/state.json").read_text(encoding="utf-8")}
        child = self.real_popen([sys.executable, "-B", "-c", CPU_CHILD, role, str(self.root / "control"), self.bad_receipt], **kwargs)
        self.children[child.pid] = child
        return child

    def wait_marker(self, role):
        deadline = time.monotonic() + 5
        while not (self.root / "control" / (role + "-cpu-ready")).exists():
            if time.monotonic() >= deadline: raise TimeoutError("CPU child did not start")
            time.sleep(.01)
        self.events.append("healthy-" + role)

    def group_identity(self, pid):
        if pid not in self.children or self.children[pid].poll() is not None:
            raise ProcessLookupError(pid)
        return {"birth": birth(pid), "pgid": pid, "sid": pid}

    def group_members(self, pgid):
        try: return {str(pgid): self.group_identity(pgid)}
        except ProcessLookupError: return {}

    def signal_group(self, pgid, signum):
        child = self.children[pgid]
        if child.poll() is not None: raise ProcessLookupError(pgid)
        child.terminate() if signum == signal.SIGTERM else child.kill()

    def emergency_cleanup(self):
        for child in self.children.values():
            if child.poll() is None: child.kill()
            child.wait(timeout=5)
        for stream in self.streams:
            stream.close()

    def run_with_failure_cleanup(self, supervisor, expected=RuntimeError):
        self.stack.enter_context(mock.patch.dict(sys.modules, {"fcntl": SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *_args: None)}))
        self.stack.enter_context(mock.patch.object(supervisor, "_server", return_value=mock.Mock()))
        with self.assertRaises(expected):
            supervisor.run()
        self.assertTrue((supervisor.root / "blocked.json").is_file())
        self.assertFalse(supervisor.is_ready())
        self.assertEqual(supervisor.state["phase"], "failed")
        self.assertTrue(all(child.poll() is not None for child in self.children.values()))

    def test_three_cpu_processes_start_in_order_and_public_ready_waits_for_registration(self):
        supervisor = self.supervisor("ref2va-sr")
        async def identities():
            self.events.append("identities")
            return {"ranks": [{"rank": rank} for rank in range(8)]}
        with mock.patch.object(launcher.HTTPControlClient, "identities", side_effect=identities):
            supervisor.start()
        self.assertEqual(self.events, ["spawn-h3", "healthy-h3", "spawn-sr", "healthy-sr", "identities", "spawn-gateway"])
        self.assertEqual(set(supervisor.children), {"h3", "sr", "gateway"})
        self.assertEqual(len({child.pid for child in supervisor.children.values()}), 3)
        observed = state.read_json(supervisor.root / "gateway-observed.json")
        self.assertTrue(observed["service_ready"])
        self.assertFalse(observed["gateway_ready"])
        self.assertEqual(observed["phase"], "loading-gateway")
        self.assertTrue(supervisor.is_ready())
        gateway_command = self.commands[-1][1]
        self.assertEqual(gateway_command[:4], [sys.executable, "-B", "-m", "dingo.vllm.omni"])
        budget = datetime.fromisoformat(supervisor.state["startup_deadline"]) - datetime.fromisoformat(supervisor.state["started_at"])
        self.assertAlmostEqual(budget.total_seconds(), 22950, delta=1)
        supervisor.stop_owned()

    def test_bad_gateway_receipt_latches_and_cleans_all_three_actual_children(self):
        self.bad_receipt = "epoch"
        self.run_with_failure_cleanup(self.supervisor())
        self.assertEqual(len(self.children), 3)

    def test_gateway_spawn_failure_latches_and_cleans_both_loaded_models(self):
        self.fail_gateway_spawn = True
        self.run_with_failure_cleanup(self.supervisor(), expected=OSError)
        self.assertEqual(len(self.children), 2)

    def test_gateway_exit_after_registration_latches_and_cleans_the_model_children(self):
        supervisor = self.supervisor()
        original = supervisor.start
        def start_then_exit():
            original()
            child = supervisor.children["gateway"]
            child.terminate(); child.wait(timeout=5)
        with mock.patch.object(supervisor, "start", side_effect=start_then_exit):
            self.run_with_failure_cleanup(supervisor)
        self.assertEqual(len(self.children), 3)

    def test_h3_exit_before_gateway_registration_latches_and_cleans_other_children(self):
        self.exit_h3_before_registration = True
        self.run_with_failure_cleanup(self.supervisor())
        self.assertEqual(len(self.children), 3)


class RegistrationContract(unittest.TestCase):
    def test_registration_rejects_wrong_hash_pid_birth_or_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, path, command, _ = prepare(temporary)
            supervisor = launcher.Supervisor(path, gateway_command_file=command, root=Path(temporary) / "control")
            child = mock.Mock(pid=1234); child.poll.return_value = None
            supervisor.children = {"gateway": child}
            supervisor.state["children"]["gateway"] = {"pid": 1234, "birth": birth(1234)}
            valid = {"schema": "h3sr-gateway-registration/1", "status": "REGISTERED", "service_epoch": supervisor.epoch,
                "profile_sha256": supervisor.state["profile_sha256"], "pid": 1234, "birth": birth(1234)}
            for key, wrong in (("service_epoch", "stale"), ("profile_sha256", "0" * 64), ("pid", 5678), ("birth", birth(5678))):
                state.atomic_json(supervisor.root / "gateway-ready.json", {**valid, key: wrong})
                with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "Gateway registration differs"):
                    supervisor._wait_gateway_registration()

    def test_registration_timeout_is_bounded_and_stale_receipt_prevents_new_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, path, command, _ = prepare(temporary)
            supervisor = launcher.Supervisor(path, gateway_command_file=command, root=Path(temporary) / "control")
            with mock.patch.object(launcher.time, "monotonic", side_effect=[0, 601]):
                with self.assertRaisesRegex(TimeoutError, "600"):
                    supervisor._wait_gateway_registration()
            state.atomic_json(supervisor.root / "gateway-ready.json", {"service_epoch": "old"})
            with self.assertRaisesRegex(RuntimeError, "Existing lifecycle"):
                supervisor.assert_fresh()

    def test_external_ready_requires_gateway_phase_and_three_live_owned_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, path, command, _ = prepare(temporary)
            supervisor = launcher.Supervisor(path, gateway_command_file=command, root=Path(temporary) / "control")
            supervisor.children = {role: mock.Mock() for role in ("h3", "sr", "gateway")}
            for child in supervisor.children.values(): child.poll.return_value = None
            supervisor.state.update(service_ready=True, gateway_ready=False, phase="loading-gateway")
            self.assertFalse(supervisor.is_ready())
            supervisor.state["gateway_ready"] = True
            self.assertFalse(supervisor.is_ready())  # Still loading-gateway.
            supervisor.state.update(gateway_ready=True, phase="ready")
            self.assertTrue(supervisor.is_ready())
            supervisor.children["h3"].poll.return_value = 1
            self.assertFalse(supervisor.is_ready())
            supervisor.children["h3"].poll.return_value = None
            state.atomic_json(supervisor.root / "blocked.json", {"reason": "unknown"})
            self.assertFalse(supervisor.is_ready())


if __name__ == "__main__":
    unittest.main()

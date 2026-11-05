"""CPU lifecycle regressions; Linux process identity/syscalls are doubles."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from service import launcher


def identity(pid, ticks, *, pgid=1234, sid=1234):
    return {"birth": {"pid": pid, "process_start_ticks": ticks,
                      "pid_namespace": "pid:[owned]", "boot_id": "same-boot"},
            "pgid": pgid, "sid": sid}


class GroupCleanup(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        profile = ROOT / "tests/fixtures/service/fl2va-sr.json"
        command_path = self.root / "gateway-command.json"
        command_path.write_text(json.dumps(["--model", "base", "--h3-sr-profile", str(profile),
            "--request-adapter", "minimax_h3", "--request-adapter-workflow", "fl2va", "--max-num-seqs", "1"]), encoding="utf-8")
        self.supervisor = launcher.Supervisor(profile, gateway_command_file=command_path, root=self.root / "control", readiness_port=0)
        self.leader = identity(1234, 99)
        self.worker = identity(1235, 100)
        self.process = mock.Mock(pid=1234)
        self.process.poll.return_value = None
        self.process.wait.return_value = 0
        self.supervisor.children = {"h3": self.process}
        self.supervisor.state["children"] = {"h3": {
            "pid": 1234, "birth": self.leader["birth"], "pgid": 1234, "sid": 1234,
            "members": {"1234": self.leader["birth"], "1235": self.worker["birth"]}}}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(launcher.signal, "signal"))
        self.stack.enter_context(mock.patch.object(launcher.signal, "SIGKILL", 9, create=True))
        self.kill = self.stack.enter_context(mock.patch.object(launcher.os, "killpg", create=True))

    def probes(self, identities, members):
        def read(pid):
            value = identities.get(pid)
            if value is None:
                raise ProcessLookupError(pid)
            return value
        self.stack.enter_context(mock.patch.object(launcher, "process_group_identity", side_effect=read))
        self.stack.enter_context(mock.patch.object(launcher, "process_group_members", side_effect=lambda _pgid: dict(members)))

    def test_exited_leader_with_reused_pid_is_never_signalled(self):
        self.process.poll.return_value = 0
        self.probes({1234: identity(1234, 999)}, {"1234": identity(1234, 999)})
        with self.assertRaisesRegex(RuntimeError, "birth/group/session changed"):
            self.supervisor.stop_owned()
        self.kill.assert_not_called()
        self.assertTrue((self.supervisor.root / "blocked.json").is_file())

    def test_pid_reused_between_term_and_kill_is_not_killed(self):
        live = {1234: self.leader, 1235: self.worker}
        members = {"1234": self.leader, "1235": self.worker}
        self.probes(live, members)
        def wait(*_args, **_kwargs):
            live[1234] = identity(1234, 999)
            return 0
        self.process.wait.side_effect = wait
        with self.assertRaisesRegex(RuntimeError, "birth/group/session changed"):
            self.supervisor.stop_owned()
        self.assertEqual(self.kill.call_args_list, [mock.call(1234, signal.SIGTERM)])

    def test_live_leader_with_changed_group_or_session_is_not_signalled(self):
        for changed in (identity(1234, 99, pgid=5678), identity(1234, 99, sid=5678)):
            with self.subTest(changed=changed), mock.patch.object(launcher, "process_group_identity", return_value=changed):
                with self.assertRaisesRegex(RuntimeError, "birth/group/session changed"):
                    self.supervisor.stop_owned()
        self.kill.assert_not_called()

    def test_term_then_kill_cleans_surviving_original_workers_after_leader_reaped(self):
        live = {1234: self.leader, 1235: self.worker}
        members = {"1234": self.leader, "1235": self.worker}
        self.probes(live, members)
        def wait(*_args, **_kwargs):
            live.pop(1234, None)
            members.pop("1234", None)
            return 0
        self.process.wait.side_effect = wait
        self.supervisor.stop_owned()
        self.assertEqual(self.kill.call_args_list,
                         [mock.call(1234, signal.SIGTERM), mock.call(1234, signal.SIGKILL)])

    def test_already_reaped_leader_still_cleans_previously_identified_worker_group(self):
        self.process.poll.return_value = 0
        self.probes({1235: self.worker}, {"1235": self.worker})
        self.supervisor.stop_owned()
        self.assertEqual([call.args[1] for call in self.kill.call_args_list], [signal.SIGTERM, signal.SIGKILL])

    def test_absent_leader_and_absent_group_need_no_signal(self):
        self.probes({}, {})
        self.supervisor.stop_owned()
        self.kill.assert_not_called()

    def test_numeric_group_without_any_original_identity_is_refused(self):
        stranger = identity(5678, 999)
        self.probes({5678: stranger}, {"5678": stranger})
        with self.assertRaisesRegex(RuntimeError, "no surviving owned identity"):
            self.supervisor.stop_owned()
        self.kill.assert_not_called()

    def test_group_reused_after_term_without_leader_pid_is_refused(self):
        live = {1234: self.leader}
        members = {"1234": self.leader}
        self.probes(live, members)
        def wait(*_args, **_kwargs):
            live.clear()
            members.clear()
            members["5678"] = identity(5678, 999)
            live[5678] = members["5678"]
            return 0
        self.process.wait.side_effect = wait
        with self.assertRaisesRegex(RuntimeError, "no surviving owned identity"):
            self.supervisor.stop_owned()
        self.assertEqual(self.kill.call_args_list, [mock.call(1234, signal.SIGTERM)])

    def test_identity_unreadable_fails_closed(self):
        with mock.patch.object(launcher, "process_group_identity", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(RuntimeError, "denied"):
                self.supervisor.stop_owned()
        self.kill.assert_not_called()

    def test_ready_records_real_group_session_and_worker_births(self):
        self.supervisor.children.clear()
        self.supervisor.state["children"].clear()
        self.probes({1234: self.leader, 1235: self.worker}, {"1234": self.leader, "1235": self.worker})
        with mock.patch.object(launcher.profiles, "environment", return_value={}), \
                mock.patch.object(launcher.subprocess, "Popen", return_value=self.process) as spawn, \
                mock.patch.object(launcher, "process_birth", return_value=self.leader["birth"]), \
                mock.patch.object(self.supervisor, "_wait_health"):
            self.supervisor._start("h3")
        for stream in self.supervisor.logs:
            self.addCleanup(stream.close)
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        saved = self.supervisor.state["children"]["h3"]
        self.assertEqual((saved["pgid"], saved["sid"]), (1234, 1234))
        self.assertEqual(saved["members"]["1235"], self.worker["birth"])
        self.assertTrue(saved["ready"])


class StableIdentity(unittest.TestCase):
    def test_group_identity_double_checks_birth_around_group_and_session_reads(self):
        original = identity(1234, 99)["birth"]
        replacement = identity(1234, 100)["birth"]
        with mock.patch.object(launcher, "process_birth", side_effect=[original, replacement]), \
                mock.patch.object(launcher.os, "getpgid", return_value=1234, create=True), \
                mock.patch.object(launcher.os, "getsid", return_value=1234, create=True):
            with self.assertRaisesRegex(RuntimeError, "birth changed"):
                launcher.process_group_identity(1234)

    def test_proc_snapshot_filters_numeric_group_and_handles_disappeared_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for pid, group in ((1234, 1234), (7777, 7777)):
                entry = root / str(pid)
                entry.mkdir()
                (entry / "stat").write_text(f"{pid} (worker (with) spaces) S 1 {group} {group} 0", encoding="ascii")
            (root / "1235").mkdir()  # Disappears between directory and stat reads.
            (root / "sys").mkdir()
            expected = identity(1234, 99)
            with mock.patch.object(launcher, "Path", return_value=root), \
                    mock.patch.object(launcher, "process_group_identity", return_value=expected) as read:
                result = launcher.process_group_members(1234)
            self.assertEqual(result, {"1234": expected})
            read.assert_called_once_with(1234)

    def test_proc_snapshot_does_not_keep_process_moved_out_of_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = root / "1234"
            entry.mkdir()
            (entry / "stat").write_text("1234 (worker) S 1 1234 1234 0", encoding="ascii")
            with mock.patch.object(launcher, "Path", return_value=root), \
                    mock.patch.object(launcher, "process_group_identity", return_value=identity(1234, 99, pgid=7777)):
                self.assertEqual(launcher.process_group_members(1234), {})


if __name__ == "__main__":
    unittest.main()

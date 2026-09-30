"""CPU-only regression tests for preparation boundaries, not GPU qualification."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_environment as prep
import score


class WheelPatchTests(unittest.TestCase):
    def test_only_metadata_identity_and_record_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / prep.ORIGINAL_FILENAME
            payload = {
                "facexlib/__init__.py": b"VERSION = '0.3.0'\n",
                prep.OLD_INFO + "/LICENSE": b"original license terms\n",
                prep.OLD_INFO + "/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
                prep.OLD_INFO + "/METADATA": b"Name: facexlib\nVersion: 0.3.0\nRequires-Dist: opencv-python\nRequires-Dist: torch\n\nDescription\n",
                prep.OLD_INFO + "/RECORD": b"old record\n",
            }
            with zipfile.ZipFile(original, "w") as wheel:
                for name, value in payload.items():
                    wheel.writestr(name, value)
            with patch.object(prep, "ORIGINAL_SHA256", prep.sha256(original)):
                rebuilt = prep.patch_facexlib(original, root / "out", root / "evidence")
                first_hash = prep.sha256(rebuilt)
                rebuilt = prep.patch_facexlib(original, root / "out", root / "evidence")
                self.assertEqual(prep.sha256(rebuilt), first_hash)
            with zipfile.ZipFile(rebuilt) as wheel:
                self.assertEqual(wheel.read("facexlib/__init__.py"), payload["facexlib/__init__.py"])
                self.assertEqual(wheel.read(prep.NEW_INFO + "/LICENSE"), payload[prep.OLD_INFO + "/LICENSE"])
                self.assertEqual(wheel.read(prep.NEW_INFO + "/WHEEL"), payload[prep.OLD_INFO + "/WHEEL"])
                metadata = wheel.read(prep.NEW_INFO + "/METADATA").decode()
                self.assertIn("Requires-Dist: opencv-python-headless\n", metadata)
                self.assertIn("Requires-Dist: torch\n", metadata)
                rows = list(csv.reader(io.StringIO(wheel.read(prep.NEW_INFO + "/RECORD").decode())))
                self.assertEqual({row[0] for row in rows}, set(wheel.namelist()))
                for name, digest, size in rows:
                    if name.endswith("/RECORD"):
                        self.assertEqual((digest, size), ("", ""))
                    else:
                        data = wheel.read(name)
                        self.assertEqual(size, str(len(data)))
                        self.assertEqual(digest, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode())
            evidence = json.loads((root / "evidence/facexlib-patch.json").read_text())
            self.assertFalse(evidence["code_changed"])
            self.assertTrue(evidence["licenses_preserved"])

    def test_unreviewed_wheel_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "bad.whl"
            wheel.write_bytes(b"not the reviewed upstream wheel")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                prep.patch_facexlib(wheel, Path(directory), Path(directory))


class ScoreTests(unittest.TestCase):
    def test_worker_arguments_no_torchrun_or_transcoding(self):
        args = score.worker_command(Path("/input"), Path("/output/subject"), "subject_consistency")
        self.assertEqual(args[-2:], ["--worker-dimension", "subject_consistency"])
        self.assertEqual(args[2:6], ["--videos", str(Path("/input")), "--output", str(Path("/output/subject"))])
        self.assertNotIn("torchrun", args)
        with self.assertRaises(ValueError):
            score.worker_command(Path("/a"), Path("/b"), "scene")

    def test_musiq_only_per_video_divided_once(self):
        path = str(Path("clip.mp4").resolve())
        raw = {dim: [0.75, [{"video_path": path, "video_results": 75 if dim == "imaging_quality" else 0.75}]]
               for dim in score.DIMENSIONS}
        rows = score.normalized_results(raw, {path})
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["score"] == 0.75 for row in rows))
        self.assertEqual(raw["imaging_quality"][0], 0.75)
        self.assertEqual(raw["imaging_quality"][1][0]["video_results"], 75)

    def test_incomplete_and_duplicate_results_rejected(self):
        path = str(Path("clip.mp4").resolve())
        raw = {dim: [0.5, [{"video_path": path, "video_results": 0.5}]] for dim in score.DIMENSIONS}
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            score.normalized_results(raw, {path, str(Path("other.mp4").resolve())})
        raw["subject_consistency"][1] *= 2
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            score.normalized_results(raw, {path})

    def test_inventory_hashes_original_and_rejects_gif(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "case.mp4"
            video.write_bytes(b"unchanged original MP4 payload")
            inventory = score.video_inventory(root)
            self.assertEqual(inventory[0]["sha256"], hashlib.sha256(video.read_bytes()).hexdigest())
            (root / "other.gif").write_bytes(b"gif")
            with self.assertRaisesRegex(ValueError, "GIF"):
                score.video_inventory(root)

    def test_uppercase_extension_rejected_without_transcoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.MP4"
            path.write_bytes(b"original bytes")
            with self.assertRaisesRegex(ValueError, "rename.*without transcoding"):
                score.video_inventory(path)


class OfflineTests(unittest.TestCase):
    def run_isolated(self, body: str) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            program = ("import sys,socket,subprocess,os\nfrom pathlib import Path\n"
                       "sys.path.insert(0, " + repr(str(SCRIPTS)) + ")\n"
                       "from offline_guard import install,assert_no_attempts\n"
                       "p=Path(" + repr(str(path)) + ")\ninstall(p)\n" + body)
            result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
            events = [json.loads(line) for line in path.read_text().splitlines()]
            return {"returncode": result.returncode, "stderr": result.stderr, "events": events}

    def test_socket_attempt_is_blocked_and_evidenced(self):
        result = self.run_isolated("socket.socket().connect(('127.0.0.1', 9))\n")
        self.assertNotEqual(result["returncode"], 0)
        self.assertIn("socket.connect", result["stderr"])
        self.assertEqual(result["events"][-1]["decision"], "blocked")

    def test_download_subprocess_and_shell_are_blocked(self):
        for body in ("subprocess.run(['wget','https://example.invalid/model'])\n", "os.system('curl https://example.invalid/model')\n"):
            result = self.run_isolated(body)
            self.assertNotEqual(result["returncode"], 0)
            self.assertEqual(result["events"][-1]["decision"], "blocked")

    def test_caught_attempt_still_fails_final_gate(self):
        result = self.run_isolated("try:\n socket.getaddrinfo('example.invalid',443)\nexcept RuntimeError:\n pass\nassert_no_attempts(p)\n")
        self.assertNotEqual(result["returncode"], 0)
        self.assertIn("swallowed", result["stderr"])

    def test_local_media_command_allowed_but_url_rejected(self):
        result = self.run_isolated("sys.audit('subprocess.Popen','ffprobe',['ffprobe','-version'],None,None)\nassert_no_attempts(p)\n")
        self.assertEqual(result["returncode"], 0, result["stderr"])
        result = self.run_isolated("sys.audit('subprocess.Popen','ffprobe',['ffprobe','https://example.invalid/a.mp4'],None,None)\n")
        self.assertNotEqual(result["returncode"], 0)


if __name__ == "__main__":
    unittest.main()

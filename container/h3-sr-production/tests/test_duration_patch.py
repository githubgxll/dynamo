"""Actual admission-function AST and CPU decoder checks; no model/GPU claims.

Metadata-only cases are synthetic. Real PyAV cases decode real Matroska files
through the same function bodies with explicit metadata overrides. A tiny
Torch stand-in collects decoded RGB bytes, so NumPy/Torch numerical behavior
is outside this admission-only test. The current PyAV version is reported;
the separate historical PyAV16.1 reproduction is never relabelled local.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = "vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py"
FIXTURE = ROOT / "tests/fixtures/duration/pipeline_seedvr2.py"
spec = importlib.util.spec_from_file_location("_duration_patch_under_test", ROOT / "duration_patch.py")
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


class AdmissionError(ValueError):
    pass


def parent_bytes():
    return FIXTURE.read_bytes()


def actual_functions(data, backend):
    """Execute unmodified AST bodies, without importing the model or CUDA."""
    wanted = {"max_frames", "validate_clip_size", "_read_video", "SourceVideo"}
    tree = ast.parse(data)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise AssertionError("Exact parent admission functions are required")
    module = ModuleType("_duration_actual_admission_" + hashlib.sha256(data).hexdigest()[:12])
    module.__dict__.update(dataclass=dataclass, Fraction=Fraction, av=backend,
        envs=SimpleNamespace(VLLM_OMNI_SEEDVR2_MAX_FRAMES=65), OmniClientError=AdmissionError,
        torch=SimpleNamespace(from_numpy=lambda value: value, stack=lambda values: tuple(values)))
    sys.modules[module.__name__] = module  # dataclass resolves postponed types here.
    body = ast.parse("from __future__ import annotations\n").body + nodes
    exec(compile(ast.Module(body=body, type_ignores=[]), "actual-duration-admission", "exec"), module.__dict__)
    return module


class SyntheticBackend:
    def __init__(self, *, count=65, duration=2709, rate=Fraction(24), frames=0):
        self.decoded = self.converted = self.opened = 0
        self.stream = SimpleNamespace(average_rate=rate, duration=duration, time_base=Fraction(1, 1000),
            frames=frames, codec_context=SimpleNamespace(height=16, width=16))
        self.streams = SimpleNamespace(video=[self.stream], audio=[])
        self.count = count

    def open(self, _path):
        self.opened += 1
        return self

    def __enter__(self): return self
    def __exit__(self, *_args): pass

    def decode(self, stream):
        if stream is not self.stream:
            raise AssertionError("Unexpected decoder stream")
        for index in range(self.count):
            self.decoded += 1
            def rgb(*, format):
                if format != "rgb24":
                    raise AssertionError("Native RGB request changed")
                self.converted += 1
                return b"decoded-rgb"
            yield SimpleNamespace(height=16, width=16, pts=round(Fraction(index * 1000, 24)), to_ndarray=rgb)


def read_with(backend, *, patched=True):
    source = patcher.transform(RELATIVE, parent_bytes()) if patched else parent_bytes()
    module = actual_functions(source, backend)
    return module._read_video("cpu-admission-input.mkv", frame_pixels=16 * 16, clip_pixels=65 * 16 * 16)


class DurationAdmission(unittest.TestCase):
    def test_65_frames_2709ms_pass_and_parent_rejects_same_metadata(self):
        # Exact historical metadata: 2709/1000 * 24 == 65.016 frames.
        backend = SyntheticBackend()
        with self.assertRaisesRegex(AdmissionError, "maximum duration"):
            read_with(backend, patched=False)
        self.assertEqual(backend.decoded, 0)
        frames, source = read_with(backend)
        self.assertEqual(len(frames), 65)
        self.assertEqual((source.fps, source.time_base), (24.0, Fraction(1, 1000)))
        self.assertEqual(backend.decoded, 65)

    def test_66_frames_with_legal_duration_and_forged_frame_count_hit_decode_hard_cap(self):
        backend = SyntheticBackend(count=66, duration=2709, frames=0)
        with self.assertRaisesRegex(AdmissionError, "frame or pixel budget"):
            read_with(backend)
        self.assertEqual((backend.decoded, backend.converted), (66, 65))

    def test_more_than_one_tick_rejected_before_decode(self):
        for duration in (2710, 3000):
            with self.subTest(duration=duration):
                backend = SyntheticBackend(duration=duration)
                with self.assertRaisesRegex(AdmissionError, "maximum duration"):
                    read_with(backend)
                self.assertEqual(backend.decoded, 0)

    def test_duration_none_keeps_decode_based_admission(self):
        for patched in (False, True):
            with self.subTest(patched=patched):
                frames, _ = read_with(SyntheticBackend(duration=None), patched=patched)
                self.assertEqual(len(frames), 65)
                backend = SyntheticBackend(count=66, duration=None)
                with self.assertRaisesRegex(AdmissionError, "frame or pixel budget"):
                    read_with(backend, patched=patched)
                self.assertEqual((backend.decoded, backend.converted), (66, 65))

    def test_invalid_rates_preserve_original_rejection_before_decode(self):
        for rate in (None, Fraction(0), Fraction(-24)):
            for patched in (False, True):
                with self.subTest(rate=rate, patched=patched):
                    backend = SyntheticBackend(rate=rate)
                    with self.assertRaisesRegex(AdmissionError, "positive frame rate"):
                        read_with(backend, patched=patched)
                    self.assertEqual(backend.decoded, 0)


class RealBackend:
    """Real demux/decode with controlled duration/frame-count metadata only."""
    def __init__(self, av, path, *, duration=2709):
        self.av, self.path, self.duration = av, path, duration
        self.decoded = self.converted = 0

    def open(self, _path):
        owner = self
        real = self.av.open(str(self.path))
        video = real.streams.video[0]
        if video.time_base != Fraction(1, 1000):
            real.close()
            raise AssertionError("Real Matroska CPU fixture must use millisecond ticks")
        selected = SimpleNamespace(average_rate=video.average_rate, duration=self.duration,
            time_base=video.time_base, frames=0, codec_context=video.codec_context)

        class Container:
            streams = SimpleNamespace(video=[selected], audio=[])
            def __enter__(self): return self
            def __exit__(self, *_args): real.close()
            def decode(self, stream):
                if stream is not selected:
                    raise AssertionError("Unexpected real decoder stream")
                for frame in real.decode(video):
                    owner.decoded += 1
                    def rgb(*, format, decoded_frame=frame):
                        if format != "rgb24":
                            raise AssertionError("Native RGB request changed")
                        owner.converted += 1
                        return bytes(decoded_frame.reformat(format="rgb24").planes[0])
                    yield SimpleNamespace(height=frame.height, width=frame.width, pts=frame.pts, to_ndarray=rgb)
        return Container()


class RealPyAVAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import av
        cls.av = av
        cls.temporary = tempfile.TemporaryDirectory()
        cls.paths = {}
        print("Duration CPU decode backend: " + json.dumps({"pyav": av.__version__,
            "libraries": av.library_versions, "scope": "Local CPU decoding; historical PyAV16.1 evidence remains separate"}, sort_keys=True))
        try:
            for count in (65, 66):
                path = Path(cls.temporary.name) / (str(count) + ".mkv")
                with av.open(str(path), "w", format="matroska") as container:
                    stream = container.add_stream("ffv1", rate=24)
                    stream.width = stream.height = 16
                    stream.pix_fmt = "yuv420p"
                    for index in range(count):
                        frame = av.VideoFrame(16, 16, "yuv420p")
                        for plane_index, plane in enumerate(frame.planes):
                            plane.update(bytes((index * 17 + plane_index * 31 + offset * 13) % 256
                                for offset in range(plane.buffer_size)))
                        frame.pts, frame.time_base = index, Fraction(1, 24)
                        for packet in stream.encode(frame): container.mux(packet)
                    for packet in stream.encode(): container.mux(packet)
                cls.paths[count] = path
        except BaseException:
            cls.temporary.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_real_65_frame_decode_accepts_exact_historical_duration_metadata(self):
        backend = RealBackend(self.av, self.paths[65])
        frames, source = read_with(backend)
        self.assertEqual((len(frames), len(source.pts), backend.decoded, backend.converted), (65, 65, 65, 65))

    def test_real_66_frame_decode_cannot_bypass_hard_cap_with_forged_metadata(self):
        backend = RealBackend(self.av, self.paths[66])
        with self.assertRaisesRegex(AdmissionError, "frame or pixel budget"):
            read_with(backend)
        self.assertEqual((backend.decoded, backend.converted), (66, 65))


class DurationSourceLock(unittest.TestCase):
    def test_exact_fixture_matches_delivery_parent_and_output_lock(self):
        inputs = ROOT / "inputs" if (ROOT / "inputs").is_dir() else ROOT.parents[1] / "delivery/h3-sr-production"
        parent = json.loads((inputs / "parent-source-manifest.json").read_text(encoding="utf-8"))
        lock = json.loads((ROOT / "duration-patches.lock.json").read_text(encoding="utf-8"))
        before = hashlib.sha256(parent_bytes()).hexdigest()
        self.assertEqual(before, parent["source_sha256"][RELATIVE])
        self.assertEqual(before, lock["changes"][RELATIVE]["before_sha256"])
        output = patcher.transform(RELATIVE, parent_bytes())
        self.assertEqual(hashlib.sha256(output).hexdigest(), lock["changes"][RELATIVE]["after_sha256"])
        self.assertEqual(output, parent_bytes().replace(patcher.BEFORE.encode(), patcher.AFTER.encode(), 1))
        original_functions = {node.name: ast.dump(node) for node in ast.parse(parent_bytes()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        actual = {node.name: ast.dump(node) for node in ast.parse(output).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        self.assertEqual(original_functions.pop("_read_video") == actual.pop("_read_video"), False)
        self.assertEqual(actual, original_functions)  # Includes the hard frame/pixel validator.

    def test_missing_and_duplicate_duration_anchors_rejected(self):
        original = parent_bytes()
        for data in (original.replace(patcher.BEFORE.encode(), b""), original + patcher.BEFORE.encode()):
            with self.subTest(sha=hashlib.sha256(data).hexdigest()):
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    patcher._apply(data)

    def test_unknown_modified_and_already_patched_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown duration source target"):
            patcher.transform("vllm_omni/another.py", parent_bytes())
        for data in (parent_bytes() + b"\n# changed\n", patcher.transform(RELATIVE, parent_bytes())):
            with self.assertRaisesRegex(ValueError, "published c88 input"):
                patcher.transform(RELATIVE, data)

    def test_after_hash_mismatch_rejected(self):
        with mock.patch.object(patcher, "_apply", return_value=parent_bytes()):
            with self.assertRaisesRegex(ValueError, "reviewed one-tick delta"):
                patcher.transform(RELATIVE, parent_bytes())


if __name__ == "__main__":
    unittest.main()

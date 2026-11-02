# SPDX-License-Identifier: Apache-2.0
"""Source/RECORD regressions; these do not simulate Linux package acceptance."""
import base64
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_omni032_recipe_runtime_test", HERE / "runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def row(name, data):
    return [name, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(), str(len(data))]


def record(rows):
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode()


class RecordOwnership(unittest.TestCase):
    def test_rejects_wrong_declared_hash_before_any_write(self):
        with self.assertRaisesRegex(ValueError, "hash differs"):
            runtime.record_update(record([row("a.py", b"wrong")]), {"a.py": (b"before", b"after")}, {})

    def test_rejects_duplicate_owner(self):
        with self.assertRaisesRegex(ValueError, "ownership"):
            runtime.record_update(record([row("a.py", b"x")] * 2), {"a.py": (b"x", b"y")}, {})

    def test_addition_cannot_overwrite_declared_file(self):
        with self.assertRaisesRegex(ValueError, "ownership"):
            runtime.record_update(record([row("new.py", b"x")]), {}, {"new.py": b"y"})

    def test_unmodified_distribution_entries_preserved(self):
        entries = [row("a.py", b"before"), row("native.so", b"native"), ["x.dist-info/RECORD", "", ""]]
        after = list(csv.reader(io.StringIO(runtime.record_update(record(entries), {"a.py": (b"before", b"after")}, {"helper.py": b"new"}).decode())))
        self.assertEqual(after[1:3], entries[1:3])
        self.assertEqual(after[0], row("a.py", b"after"))
        self.assertEqual(after[-1], row("helper.py", b"new"))


class SourceBoundaries(unittest.TestCase):
    def test_parent_escape_and_foreign_paths_rejected(self):
        for name in ("../other.py", "/absolute.py", "C:/other.py", "a\\other.py"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime.safe(Path.cwd(), name)

    def test_every_delta_origins_in_formal_fixture_and_is_prepared_without_writes(self):
        fixture = HERE / "tests/fixtures/upstream"
        self.assertTrue(fixture.is_dir(), "Formal release fixtures are mandatory")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pristine = {}
            for path in fixture.rglob("*.py"):
                name = path.relative_to(fixture).as_posix()
                pristine[name] = path.read_bytes()
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(pristine[name])
            changes, additions, expected = runtime.prepare_changes(root, pristine)
            self.assertEqual(len(changes), 9)
            self.assertEqual(len(additions), 4)
            self.assertTrue(set(changes).issubset(pristine))
            self.assertFalse(set(additions) & set(pristine))
            self.assertEqual(set(expected), set(pristine) | set(additions))
            for name, data in pristine.items():
                self.assertEqual((root / name).read_bytes(), data)
            self.assertFalse(any((root / name).exists() for name in additions))

    def test_source_manifest_does_not_hide_unknown_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "vllm_omni").mkdir()
            (root / "vllm_omni/unknown.py").write_bytes(b"unknown")
            (root / "vllm_omni/__pycache__").mkdir()
            (root / "vllm_omni/__pycache__/unknown.pyc").write_bytes(b"generated")
            self.assertEqual(runtime.sources(root), {"vllm_omni/unknown.py": runtime.digest(b"unknown")})

    def test_manifest_digest_is_order_independent_but_content_sensitive(self):
        self.assertEqual(runtime.map_digest({"a": "1", "b": "2"}), runtime.map_digest({"b": "2", "a": "1"}))
        self.assertNotEqual(runtime.map_digest({"a": "1"}), runtime.map_digest({"a": "2"}))


if __name__ == "__main__":
    unittest.main()

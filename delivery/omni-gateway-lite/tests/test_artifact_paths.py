"""Offline artifact path rejection without needing a registry or model runtime."""
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import import_evidence


class ArtifactPaths(unittest.TestCase):
    def test_expected_nested_recipe_path(self):
        name = "recipe/.github/scripts/omni_gateway_lite_ci.py"
        self.assertEqual(import_evidence.safe_name(name).as_posix(), name)

    def test_directory_marker_is_allowed(self):
        self.assertEqual(import_evidence.safe_name("build/retained/").as_posix(), "build/retained")

    def test_traversal_and_absolute_paths_are_rejected(self):
        for name in ("../escape", "a/../escape", "/etc/passwd", "a/./b", "a//b", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                import_evidence.safe_name(name)

    def test_windows_drive_ads_and_backslash_are_rejected(self):
        for name in ("C:/escape", "file:stream", "a\\b", "\\\\server\\share"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                import_evidence.safe_name(name)


if __name__ == "__main__":
    unittest.main()

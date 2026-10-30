import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from initialization_reference import load_reference_result, validate_initialization_reference


class InitializationReferenceTests(unittest.TestCase):
    def test_reference_is_optional(self):
        self.assertIsNone(load_reference_result(None))
        self.assertFalse(validate_initialization_reference(None, "a" * 64, "b" * 64))

    def test_matching_reference_and_mismatches(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "result.json"
            p.write_text(json.dumps({"classifier_initial_sha256": "a" * 64,
                                     "frozen_backbone_sha256": "b" * 64, "accuracy": 99.}))
            reference = load_reference_result(p)
            self.assertNotIn("accuracy", reference)
            self.assertTrue(validate_initialization_reference(reference, "a" * 64, "b" * 64))
            for head, backbone in [("c" * 64, "b" * 64), ("a" * 64, "c" * 64)]:
                with self.assertRaises(ValueError):
                    validate_initialization_reference(reference, head, backbone)

    def test_missing_and_malformed_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "result.json"
            with self.assertRaises(FileNotFoundError):
                load_reference_result(p)
            for payload in [{}, {"classifier_initial_sha256": "wrong"}, []]:
                p.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    load_reference_result(p)

    def test_cli_help_needs_no_gpu_or_prior_run(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/train_validation.py"), "--help"],
                                    cwd=folder, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--reference-result", result.stdout)

    def test_missing_explicit_reference_fails_before_resource_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "missing.json"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/train_validation.py"),
                                     "--method", "linear", "--output", "runs/not-started",
                                     "--reference-result", str(missing)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Initialization reference not found", result.stderr)
        self.assertNotIn("nvidia-smi", result.stderr)
        self.assertFalse((ROOT / "runs/not-started").exists())


if __name__ == "__main__":
    unittest.main()

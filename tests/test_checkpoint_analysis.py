import sys
from pathlib import Path
import unittest
import tempfile
import json
import hashlib

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_analysis import (AssetExporter, aggregate_predictions, log_probabilities, paired_nll_bootstrap,
                                 per_class_recall, probability_metrics, reliability_bins,
                                 runtime_summary, wilson_interval)


class CheckpointMetricsTests(unittest.TestCase):
    def test_proper_scores_match_hand_calculation(self):
        p = np.array([[.8, .2], [.3, .7]])
        scores = probability_metrics(p, np.array([0, 1]))
        self.assertAlmostEqual(scores["nll"], -(np.log(.8) + np.log(.7)) / 2)
        self.assertAlmostEqual(scores["brier_multiclass_sum"], .13)
        self.assertEqual(scores["correct"], 2)

    def test_extreme_logits_and_common_shift_are_stable(self):
        logits = np.array([[10000., 9999., -10000.], [-9000., -8999., -8998.]])
        lp = log_probabilities(logits)
        self.assertTrue(np.isfinite(lp).all())
        np.testing.assert_allclose(np.exp(lp).sum(1), 1)
        np.testing.assert_allclose(lp, log_probabilities(logits + 30000), atol=1e-10)

    def test_reliability_includes_confidence_one_and_empty_bins(self):
        p = np.array([[1., 0.], [.6, .4]])
        bins = reliability_bins(p, np.array([0, 1]))
        self.assertEqual(int(bins["count"].sum()), 2)
        self.assertEqual(int(bins.iloc[-1]["count"]), 1)
        self.assertTrue(bins.loc[bins["count"] == 0, "accuracy"].isna().all())

    def test_invalid_probabilities_and_labels_fail(self):
        with self.assertRaises(ValueError):
            probability_metrics(np.array([[.8, .8]]), np.array([0]))
        with self.assertRaises(ValueError):
            probability_metrics(np.array([[.8, .2]]), np.array([2]))

    def test_wilson_boundary_remains_inside_probability_range(self):
        self.assertEqual(wilson_interval(200, 200)[1], 1.)
        self.assertEqual(wilson_interval(0, 200)[0], 0.)

    def test_macro_recall_excludes_unobserved_classes(self):
        y = np.array([0, 0, 0, 1])
        predicted = np.zeros(4, dtype=int)
        table = per_class_recall(predicted, y, 3)
        np.testing.assert_array_equal(table["validation_support"], [3, 1, 0])
        self.assertTrue(np.isnan(table.iloc[2]["recall"]))
        scores = probability_metrics(np.tile([.8, .1, .1], (4, 1)), y)
        self.assertEqual(scores["accuracy_percent"], 75.)
        self.assertEqual(scores["macro_recall_observed_percent"], 50.)
        self.assertEqual(scores["classes_observed"], 2)

    def test_runtime_units_distinguish_total_and_per_epoch(self):
        result = runtime_summary(pd.DataFrame({"epoch_seconds": np.full(100, 2.)}))
        self.assertAlmostEqual(result["observed_total_epoch_minutes"], 200 / 60)
        self.assertEqual(result["observed_mean_epoch_seconds"], 2.)
        self.assertNotIn("observed_epoch_minutes", result)
        for values in ([], [-1.], [np.nan]):
            with self.assertRaises(ValueError):
                runtime_summary(pd.DataFrame({"epoch_seconds": values}))

    def test_aggregation_rules_match_direct_calculation(self):
        a = np.array([[4., 0.], [0., 0.]])
        b = np.array([[0., 0.], [0., 2.]])
        def softmax(z):
            ex = np.exp(z)
            return ex / ex.sum(axis=1, keepdims=True)
        probability_mean, _ = aggregate_predictions([a, b], "probability_mean")
        logit_mean, _ = aggregate_predictions([a, b], "logit_mean")
        np.testing.assert_allclose(probability_mean, (softmax(a) + softmax(b)) / 2)
        np.testing.assert_allclose(logit_mean, softmax((a + b) / 2))
        self.assertFalse(np.allclose(probability_mean, logit_mean))
        for rule in ("probability_mean", "logit_mean"):
            expected, _ = aggregate_predictions([a, b], rule)
            reversed_order, _ = aggregate_predictions([b, a], rule)
            shifted, _ = aggregate_predictions([a + 10000, b - 20000], rule)
            np.testing.assert_allclose(expected, reversed_order)
            np.testing.assert_allclose(expected, shifted, atol=1e-10)

    def test_aggregation_handles_extreme_logits_and_rejects_invalid_members(self):
        z = np.array([[10000., -10000.]])
        for rule in ("probability_mean", "logit_mean"):
            p, lp = aggregate_predictions([z, z], rule)
            self.assertTrue(np.isfinite(lp).all())
            self.assertAlmostEqual(probability_metrics(p, np.array([1]), lp)["nll"], 20000.)
        for members, rule in [([], "logit_mean"), ([z], "unknown"), ([z, np.zeros((2, 2))], "logit_mean")]:
            with self.assertRaises(ValueError):
                aggregate_predictions(members, rule)

    def test_bootstrap_is_paired_reproducible_and_signed(self):
        y = np.array([0, 1, 0, 1])
        lp = np.log(np.array([[.8, .2], [.3, .7], [.6, .4], [.1, .9]]))
        same = {"Linear": {"targets": y, "log_probs": lp}, "LoRA": {"targets": y.copy(), "log_probs": lp.copy()}}
        result = paired_nll_bootstrap(same, resamples=100, seed=42).iloc[0]
        self.assertEqual(result["delta_nll_a_minus_b"], 0.)
        self.assertEqual(result["ci95_low"], 0.)
        self.assertEqual(result["ci95_high"], 0.)
        changed = {"Linear": same["Linear"], "LoRA": {"targets": y.copy(), "log_probs": np.log(np.full((4, 2), .5))}}
        first = paired_nll_bootstrap(changed, resamples=100, seed=42)
        second = paired_nll_bootstrap(changed, resamples=100, seed=42)
        pd.testing.assert_frame_equal(first, second)
        expected = -np.log(.5) - (-lp[np.arange(4), y]).mean()
        self.assertAlmostEqual(first.iloc[0]["delta_nll_a_minus_b"], expected)
        changed["LoRA"]["targets"] = y[::-1]
        with self.assertRaises(ValueError):
            paired_nll_bootstrap(changed)

    def test_mermaid_assets_require_matching_sources_and_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assets = root / "assets/checkpoint1"
            (assets / "diagrams").mkdir(parents=True)
            (assets / "figures").mkdir()
            config = assets / "diagrams/mermaid-config.json"
            config.write_text("{}\n")
            source = assets / "diagrams/example.mmd"
            source.write_text("flowchart LR\nA --> B\n")
            png = assets / "figures/example.png"
            Image.new("RGB", (10, 10), "white").save(png)
            svg = assets / "figures/example.svg"
            svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            record = {"source": "diagrams/example.mmd", "source_sha256": digest(source),
                      "png": "figures/example.png", "png_sha256": digest(png),
                      "svg": "figures/example.svg", "svg_sha256": digest(svg)}
            manifest = {"renderer": {"version": "12.0.0"}, "config_sha256": digest(config),
                        "diagrams": {"example": record}}
            (assets / "diagrams/rendered.json").write_text(json.dumps(manifest))
            exporter = AssetExporter(root)
            self.assertEqual(exporter.diagram("example", "Methods", "Example"), png)
            self.assertEqual(len(exporter.records), 3)
            source.write_text("flowchart LR\nA --> C\n")
            with self.assertRaisesRegex(ValueError, "Re-render Mermaid"):
                AssetExporter(root).diagram("example", "Methods", "Example")


if __name__ == "__main__":
    unittest.main()

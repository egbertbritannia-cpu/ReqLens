"""Synthetic-only CPU tests for precomputed-logits export and provenance guards."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.routing.export_prediction_cache import export_cache


class ExportPredictionCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "data.csv"
        self.dataset.write_text("RequirementText,class\nfoo,A\nbar,B\n", encoding="utf-8")
        self.selector = self.root / "selector.weights"
        self.fusion = self.root / "fusion.weights"
        self.selector.write_bytes(b"selector placeholder: synthetic only")
        self.fusion.write_bytes(b"fusion placeholder: synthetic only")
        self.manifest = {
            "status": "SPLIT_RECONSTRUCTED_NOT_PREDICTIONS",
            "dataset_sha256": hashlib.sha256(self.dataset.read_bytes()).hexdigest(),
            "split_seed": 42, "label_order": ["A", "B"],
            "splits": {"validation": [{"source_row_index": 0, "label": "A"},
                                       {"source_row_index": 1, "label": "B"}]},
        }
        self.meta = {
            "variant": "training_separate_fusion", "notebook_path": "model-training/SERF.ipynb",
            "notebook_commit_sha": "e" * 40, "split": "validation", "split_seed": 42,
            "train_seed": 123, "label_order": ["A", "B"], "selector_temperature": 1.0,
            "roberta_temperature": None, "alpha": None, "threshold_selection_split": "validation",
            "tau": .7, "delta": .2, "risky_labels": ["A"],
        }
        self.records = [{"source_row_index": 0, "true_label": "A", "selector_logits": [3., 0.], "fusion_logits": [0., 3.]},
                        {"source_row_index": 1, "true_label": "B", "selector_logits": [0., 3.], "fusion_logits": [3., 0.]}]
        self.dest = self.root / "cache.json"

    def run_export(self):
        return export_cache(self.manifest, self.dataset, self.records, self.meta,
                            self.selector, self.fusion, self.dest)

    def test_separate_fusion_success_and_contract(self):
        payload = self.run_export()
        self.assertEqual(payload["status"], "UNVERIFIED_MODEL_PROVENANCE")
        self.assertEqual([r["fusion_pred"] for r in payload["records"]], ["B", "A"])
        self.assertEqual(len(payload["records"]), 2)
        self.assertAlmostEqual(payload["records"][0]["selector_probs"][0], .9525741268, places=8)
        self.assertEqual(json.loads(self.dest.read_text()), payload)
        self.assertEqual(payload["metadata"]["selector_checkpoint_sha256"], hashlib.sha256(self.selector.read_bytes()).hexdigest())

    def test_development_probability_interpolation(self):
        self.meta.update({"variant": "development_late_fusion", "notebook_path": "code/SERF.ipynb",
                          "selector_temperature": 2.0, "roberta_temperature": 1.0, "alpha": 1.0})
        payload = self.run_export()
        self.assertEqual([r["fusion_pred"] for r in payload["records"]], ["A", "B"])
        self.assertAlmostEqual(payload["records"][0]["selector_probs"][0], .8175744762, places=8)

    def test_wrong_order_rejected_no_output(self):
        self.records.reverse()
        with self.assertRaisesRegex(ValueError, "order/ID"):
            self.run_export()
        self.assertFalse(self.dest.exists())

    def test_wrong_true_label_rejected_no_output(self):
        self.records[0]["true_label"] = "B"
        with self.assertRaisesRegex(ValueError, "ground-truth"):
            self.run_export()
        self.assertFalse(self.dest.exists())

    def test_rejects_modified_dataset(self):
        self.dataset.write_text("modified", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Dataset bytes"):
            self.run_export()

    def test_rejects_missing_checkpoint(self):
        self.selector.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_export()

    def test_rejects_nonfinite_logits(self):
        self.records[0]["selector_logits"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "logits vector"):
            self.run_export()
        self.assertFalse(self.dest.exists())

    def test_exclusive_output_cannot_overwrite(self):
        self.run_export()
        original = self.dest.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_export()
        self.assertEqual(self.dest.read_bytes(), original)

    def test_rejects_test_tuned_thresholds(self):
        self.meta["threshold_selection_split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation-selected"):
            self.run_export()

    def test_rejects_mixed_architecture_metadata(self):
        self.meta["alpha"] = .5
        with self.assertRaisesRegex(ValueError, "Separate fusion"):
            self.run_export()


if __name__ == "__main__":
    unittest.main()

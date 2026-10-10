"""Pure-CPU provenance guard tests; no notebook inference or downloads."""
import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "experiments/routing/prediction_cache_contract.py"
spec = importlib.util.spec_from_file_location("prediction_cache_contract", PATH)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def sample(variant="training_separate_fusion"):
    dev = variant == "development_late_fusion"
    return {
        "metadata": {
            "schema_version": 1, "variant": variant, "dataset_sha256": "a" * 64,
            "notebook_path": ("code/" if dev else "model-training/") + "ELECTRA_RoBERTa_SERF_without_augmentation.ipynb",
            "notebook_commit_sha": "c" * 40, "split": "validation", "split_seed": 42,
            "train_seed": 123, "label_order": ["A", "B", "C"],
            "selector_checkpoint_sha256": "b" * 64, "fusion_checkpoint_sha256": "d" * 64,
            "fusion_method": "probability_interpolation" if dev else "separate_fusion_classifier",
            "selector_temperature": 1.4 if dev else 1.0,
            "roberta_temperature": 1.2 if dev else None, "alpha": 0.25 if dev else None,
            "threshold_selection_split": "validation", "tau": 0.7, "delta": 0.2,
            "risky_labels": ["B"],
        },
        "records": [
            {"source_row_index": 10, "true_label": "A", "selector_probs": [0.8, 0.1, 0.1], "fusion_pred": "B"},
            {"source_row_index": 23, "true_label": "B", "selector_probs": [0.1, 0.8, 0.1], "fusion_pred": "B"},
        ],
    }


class PredictionCacheContractTests(unittest.TestCase):
    def test_both_architectures_with_explicit_source_row(self):
        for variant in ("development_late_fusion", "training_separate_fusion"):
            with self.subTest(variant=variant):
                rows = contract.validate_cache(sample(variant))
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["id"], "a" * 64 + ":10")
                self.assertNotEqual(rows[0]["id"], rows[1]["id"])

    def test_rejects_missing_model_provenance(self):
        x = sample()
        del x["metadata"]["fusion_checkpoint_sha256"]
        with self.assertRaises(ValueError):
            contract.validate_cache(x)

    def test_rejects_cross_architecture_recipe(self):
        x = sample()
        x["metadata"]["fusion_method"] = "probability_interpolation"
        with self.assertRaises(ValueError):
            contract.validate_cache(x)

    def test_rejects_thresholds_selected_on_test(self):
        x = sample()
        x["metadata"]["threshold_selection_split"] = "test"
        with self.assertRaises(ValueError):
            contract.validate_cache(x)

    def test_rejects_duplicate_original_source_index(self):
        x = sample()
        x["records"][1]["source_row_index"] = 10
        with self.assertRaises(ValueError):
            contract.validate_cache(x)

    def test_rejects_inconsistent_class_probabilities(self):
        x = sample()
        x["records"][0]["selector_probs"] = [0.8, 0.8, -0.6]
        with self.assertRaises(ValueError):
            contract.validate_cache(x)

    def test_rejects_unverified_split_and_missing_label(self):
        x = sample()
        x["metadata"]["split"] = "training"
        with self.assertRaises(ValueError):
            contract.validate_cache(x)
        y = sample()
        y["records"][0]["true_label"] = "OUTSIDE"
        with self.assertRaises(ValueError):
            contract.validate_cache(y)

    def test_rejects_dev_cache_without_calibration(self):
        x = sample("development_late_fusion")
        x["metadata"]["roberta_temperature"] = None
        with self.assertRaises(ValueError):
            contract.validate_cache(x)


if __name__ == "__main__":
    unittest.main()

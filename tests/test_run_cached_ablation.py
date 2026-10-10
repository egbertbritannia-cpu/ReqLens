"""CPU-only full-cache ablation checks on SYNTHETIC CSV/logits/checkpoints.

Not a reconstruction of the historical SERF results or trained checkpoints.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.routing.split_provenance import build_split_manifest
from experiments.routing.export_prediction_cache import export_cache
from experiments.routing.run_cached_ablation import run_cached_ablation, REPORT_STATUS


class CachedAblationIntegrationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.dataset = self.dir / "tiny.csv"
        with self.dataset.open("w", encoding="utf-8") as f:
            f.write("ProjectID,RequirementText,class\n")
            for i in range(35):
                for cls in ["A", "B", "C"]:
                    f.write(f"x,text {i} {cls},{cls}\n")
        self.manifest = build_split_manifest(self.dataset)
        self.manifest_file = self.dir / "split.json"
        self.manifest_file.write_text(json.dumps(self.manifest))
        self.selector = self.dir / "dummy_selector.ckpt"
        self.fusion = self.dir / "dummy_fusion.ckpt"
        self.selector.write_bytes(b"test-only-selector-not-a-checkpoint")
        self.fusion.write_bytes(b"test-only-fusion-not-a-checkpoint")
        self.variant = "training_separate_fusion"
        self.split = "validation"
        self.config = {
            "variant": "training_separate_fusion",
            "notebook_path": "model-training/ELECTRA_RoBERTa_SERF_without_augmentation.ipynb",
            "notebook_commit_sha": "b" * 40,
            "split": self.split, "split_seed": 42, "train_seed": 123,
            "label_order": self.manifest["label_order"],
            "selector_temperature": 1.0, "roberta_temperature": None,
            "alpha": None, "threshold_selection_split": "validation",
            "tau": .70, "delta": .20, "risky_labels": ["B"],
        }
        self.cache_file = self.dir / "prediction_cache.json"
        self.output = self.dir / "ablation.json"
        self._write_cache()

    def _write_cache(self):
        self.records = []
        for i, item in enumerate(self.manifest["splits"][self.split]):
            selector = [0.0, 0.0, 0.0]
            fusion = [0.0, 0.0, 0.0]
            selector[i % 3] = 2.0
            fusion[(i + 1) % 3] = 3.0
            self.records.append({"source_row_index": item["source_row_index"],
                                 "true_label": item["label"],
                                 "selector_logits": selector, "fusion_logits": fusion})
        export_cache(self.manifest, self.dataset, self.records, self.config,
                     self.selector, self.fusion, self.cache_file)

    def run_report(self, random_draws=7):
        return run_cached_ablation(self.cache_file, self.manifest_file, self.dataset,
                                   self.selector, self.fusion, self.output, random_draws)

    def test_complete_validation_ablation_all_eight_modes(self):
        result = self.run_report()
        self.assertEqual(result["status"], REPORT_STATUS)
        self.assertTrue(result["scientific_warning"])
        self.assertEqual(len(result["eight_mode_results"]), 8)
        self.assertEqual(result["diagnostics"]["n"], len(self.records))
        self.assertEqual(len(result["sample_diagnostics"]), len(self.records))
        self.assertEqual(result["matched_random_descriptive"]["draws"], 7)
        self.assertTrue(result["matched_random_descriptive"]["not_statistical_significance"])
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertEqual([r["source_row_index"] for r in result["sample_diagnostics"]],
                         [r["source_row_index"] for r in self.records])

    def test_cache_test_split_allowed_only_with_validation_tuning(self):
        self.cache_file.unlink()
        self.split = "test"
        self.config["split"] = "test"
        self._write_cache()
        result = self.run_report()
        self.assertEqual(result["split"], "test")
        self.assertEqual(len(result["sample_diagnostics"]), len(self.manifest["splits"]["test"]))

    def test_rejects_tampered_frozen_manifest(self):
        fake = copy.deepcopy(self.manifest)
        fake["splits"]["validation"].reverse()
        self.manifest_file.write_text(json.dumps(fake))
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.run_report()
        self.assertFalse(self.output.exists())

    def test_rejects_modified_model_weights(self):
        self.selector.write_bytes(b"modified dummy checkpoint")
        with self.assertRaisesRegex(ValueError, "selector checkpoint"):
            self.run_report()
        self.assertFalse(self.output.exists())

    def test_rejects_modified_csv_bytes(self):
        with self.dataset.open("a") as f:
            f.write("extra,forbidden,A\n")
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.run_report()
        self.assertFalse(self.output.exists())

    def test_rejects_cache_reordering(self):
        obj = json.loads(self.cache_file.read_text())
        obj["records"].reverse()
        self.cache_file.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "order/label"):
            self.run_report()

    def test_rejects_incomplete_split_predictions(self):
        obj = json.loads(self.cache_file.read_text())
        obj["records"].pop()
        self.cache_file.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "complete selected split"):
            self.run_report()

    def test_invalid_provenance_status_rejected(self):
        obj = json.loads(self.cache_file.read_text())
        obj["status"] = "VERIFIED"
        self.cache_file.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "UNVERIFIED_MODEL_PROVENANCE"):
            self.run_report()

    def test_exclusive_output_no_overwrite(self):
        self.run_report()
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_report()
        self.assertEqual(self.output.read_bytes(), before)

    def test_deterministic_random_descriptive_not_ci(self):
        report = self.run_report(random_draws=9)
        self.output.unlink()
        again = self.run_report(random_draws=9)
        self.assertEqual(report["matched_random_descriptive"], again["matched_random_descriptive"])
        self.assertFalse("p_value" in json.dumps(report))

    def test_rejects_bad_replicate_count(self):
        for draws in (0, 1001, True, 2.5):
            with self.subTest(draws=draws), self.assertRaises(ValueError):
                self.run_report(draws)
        self.assertFalse(self.output.exists())

    def test_independent_development_variant_supported(self):
        self.cache_file.unlink()
        self.config.update({"variant":"development_late_fusion",
                            "notebook_path":"code/ELECTRA_RoBERTa_SERF_without_augmentation.ipynb",
                            "selector_temperature":1.3, "roberta_temperature":1.1,
                            "alpha":.25})
        self._write_cache()
        result = self.run_report()
        self.assertEqual(result["variant"], "development_late_fusion")
        self.assertEqual(result["diagnostics"]["n"], len(self.records))


if __name__ == "__main__":
    unittest.main()

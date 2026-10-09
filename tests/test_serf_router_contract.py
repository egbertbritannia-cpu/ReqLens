"""CPU-only contract tests for existing SERF notebook routing code.

Loads ONLY named function definitions from notebook source. Never executes notebook
cells, downloads models/data, changes checkpoints, or trains a network.
Run: python -m unittest discover -s tests -p 'test_serf_router_contract.py' -v
"""
import ast
import json
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "code/ELECTRA_RoBERTa_SERF_without_augmentation.ipynb"
TRAINING = ROOT / "model-training/ELECTRA_RoBERTa_SERF_without_augmentation.ipynb"


def load_functions(notebook_path, names):
    """Compile only requested top-level functions, never execute whole cells."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    nodes = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = "".join(source) if isinstance(source, list) else source
        if not any(f"def {name}(" in source for name in names):
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                if node.name in nodes:
                    raise AssertionError(f"Duplicate definition: {node.name}")
                nodes[node.name] = node
    missing = set(names) - set(nodes)
    if missing:
        raise AssertionError(f"Missing routing functions in {notebook_path}: {sorted(missing)}")
    namespace = {"torch": torch, "nn": nn, "np": np,
                 "compute_metrics_from_arrays": lambda labels, preds: {}}
    module = ast.fix_missing_locations(ast.Module(body=[nodes[name] for name in names], type_ignores=[]))
    exec(compile(module, str(notebook_path), "exec"), namespace)
    return namespace


def log_probs(rows):
    return torch.tensor(rows, dtype=torch.float64).log()


class FakeExpert:
    def __init__(self, column):
        self.column = column
        self.class_weights = torch.ones(3, dtype=torch.float64)
        self.batch_sizes = []

    def eval(self):
        return self

    def __call__(self, **batch):
        self.batch_sizes.append(len(batch["labels"]))
        return {"logits": batch[self.column]}


class SERFRouterContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = load_functions(DEVELOPMENT, [
            "predicted_risky_mask", "calibrated_probabilities",
            "probability_statistics", "selector_routing_signals"])
        cls.train = load_functions(TRAINING, [
            "predicted_risky_mask", "evaluate_selective_routing"])

    def route(self, logits, tau=0.70, delta=0.15, risky=()):
        return self.dev["selector_routing_signals"](
            logits, temperature=1.0, tau=tau, delta=delta,
            risky_label_ids=set(risky))

    def test_easy_path_no_trigger(self):
        result = self.route(log_probs([[0.90, 0.05, 0.05]]))
        self.assertEqual(result["route_to_fusion"].tolist(), [False])

    def test_confidence_only_trigger(self):
        result = self.route(log_probs([[0.65, 0.20, 0.15]]))
        self.assertEqual(result["confidence_failure"].tolist(), [True])
        self.assertEqual(result["margin_failure"].tolist(), [False])
        self.assertEqual(result["route_to_fusion"].tolist(), [True])

    def test_margin_only_trigger(self):
        result = self.route(log_probs([[0.52, 0.40, 0.08]]), tau=0.50, delta=0.20)
        self.assertEqual(result["confidence_failure"].tolist(), [False])
        self.assertEqual(result["margin_failure"].tolist(), [True])
        self.assertEqual(result["route_to_fusion"].tolist(), [True])

    def test_risky_class_only_trigger(self):
        result = self.route(log_probs([[0.05, 0.90, 0.05]]), risky={1})
        self.assertEqual(result["confidence_failure"].tolist(), [False])
        self.assertEqual(result["margin_failure"].tolist(), [False])
        self.assertEqual(result["risky_class_trigger"].tolist(), [True])
        self.assertEqual(result["route_to_fusion"].tolist(), [True])

    def test_boundary_equality_is_easy(self):
        result = self.route(log_probs([[0.50, 0.50, 0.0]]), tau=0.50, delta=0.0)
        self.assertEqual(result["route_to_fusion"].tolist(), [False])

    def test_empty_risky_set_and_finite_entropy(self):
        result = self.route(log_probs([[0.05, 0.90, 0.05]]), risky=set())
        self.assertEqual(result["risky_class_trigger"].tolist(), [False])
        self.assertTrue(bool(torch.isfinite(result["entropy"]).all()))

    def test_development_and_training_route_masks_agree(self):
        selector = log_probs([
            [0.90, 0.05, 0.05], [0.52, 0.40, 0.08],
            [0.05, 0.90, 0.05], [0.72, 0.14, 0.14]])
        fusion = log_probs([
            [0.05, 0.90, 0.05], [0.05, 0.05, 0.90],
            [0.90, 0.05, 0.05], [0.05, 0.05, 0.90]])
        batch = {"labels": torch.tensor([0, 2, 0, 0]),
                 "selector_logits": selector, "fusion_logits": fusion}
        selector_model, fusion_model = FakeExpert("selector_logits"), FakeExpert("fusion_logits")
        metrics, labels, predictions = self.train["evaluate_selective_routing"](
            selector_model, fusion_model, [batch], "cpu", 0.70, 0.15, {1})
        dev_mask = self.route(selector, risky={1})["route_to_fusion"]
        self.assertEqual(dev_mask.tolist(), [False, True, True, False])
        self.assertAlmostEqual(metrics["route_to_fusion_rate"], dev_mask.float().mean().item())
        self.assertEqual(selector_model.batch_sizes, [4])
        self.assertEqual(fusion_model.batch_sizes, [2])
        self.assertEqual(labels.tolist(), [0, 2, 0, 0])
        self.assertEqual(predictions.tolist(), [0, 2, 0, 0])

    def test_training_easy_batch_skips_fusion(self):
        selector = log_probs([[0.90, 0.05, 0.05]])
        batch = {"labels": torch.tensor([0]), "selector_logits": selector,
                 "fusion_logits": log_probs([[0.05, 0.90, 0.05]])}
        selector_model, fusion_model = FakeExpert("selector_logits"), FakeExpert("fusion_logits")
        metrics, _, predictions = self.train["evaluate_selective_routing"](
            selector_model, fusion_model, [batch], "cpu", 0.70, 0.15, set())
        self.assertEqual(fusion_model.batch_sizes, [])
        self.assertEqual(predictions.tolist(), [0])
        self.assertAlmostEqual(metrics["route_to_fusion_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

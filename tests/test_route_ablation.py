"""CPU-only tests with valid probability simplexes; no external packages."""
import importlib.util
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "experiments/routing/route_ablation.py"
spec = importlib.util.spec_from_file_location("route_ablation", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(i, probs, label="A", fusion="A"):
    return {"id": str(i), "true_label": label, "selector_probs": probs,
            "fusion_pred": fusion}


class RoutingAblationTests(unittest.TestCase):
    def test_all_eight_patterns_are_possible_outside_legacy_grid(self):
        # tau=.50, delta=.20: all c,m combinations are feasible.
        pairs = [(.90, .05, .05), (.52, .40, .08),
                 (.49, .26, .25), (.40, .35, .25)]
        rows = []
        for p in pairs:
            rows.append(row(len(rows), p))  # top class A, not risky
            rows.append(row(len(rows), (p[1], p[0], p[2])))  # top B, risky
        masks, diag, _ = module.evaluate_cached(rows, ["A", "B", "C"], .50, .20, {"B"})
        self.assertEqual(diag["trigger_patterns"], {f"{i:03b}": 1 for i in range(8)})
        self.assertEqual(diag["exclusive_counts"],
                         {"confidence": 1, "margin": 1, "risky_class": 1})
        self.assertEqual(sum(masks["full"]), 7)
        self.assertEqual(sum(masks["matched_random"]), 7)

    def test_legacy_margin_is_subsumed_by_confidence(self):
        rows = []
        for a in range(0, 101, 2):
            for b in range(0, 101 - a, 2):
                rows.append(row(len(rows), (a / 100, b / 100, (100 - a - b) / 100)))
        for tau in (.70, .80, .85, .90):
            for delta in (.15, .20):
                masks, diag, _ = module.evaluate_cached(rows, ["A", "B", "C"], tau, delta, {"B"})
                self.assertEqual(diag["exclusive_counts"]["margin"], 0)
                self.assertEqual(masks["full"],
                                 [c or r for c, r in zip(masks["confidence_only"],
                                                        masks["risky_class_only"])])

    def test_boundary_and_matched_random_are_deterministic(self):
        rows = [row(i, (.70, .20, .10)) for i in range(6)]
        masks1, _, _ = module.evaluate_cached(rows, ["A", "B", "C"], .70, .20, {"A"}, seed=17)
        masks2, _, _ = module.evaluate_cached(rows, ["A", "B", "C"], .70, .20, {"A"}, seed=17)
        self.assertEqual(masks1["matched_random"], masks2["matched_random"])
        self.assertEqual(sum(masks1["matched_random"]), 6)
        self.assertEqual(sum(masks1["confidence_only"]), 0)
        self.assertEqual(sum(masks1["margin_only"]), 0)

    def test_corrected_and_degraded_counts(self):
        rows = [row(0, (.9, .05, .05), "A", "B"),
                row(1, (.9, .05, .05), "B", "B"),
                row(2, (.9, .05, .05), "A", "B"),
                row(3, (.9, .05, .05), "B", "A")]
        _, _, scores = module.evaluate_cached(rows, ["A", "B", "C"], .70, .20, {"A"})
        self.assertEqual(scores["full"]["corrected_count"], 1)
        self.assertEqual(scores["full"]["degraded_count"], 2)
        self.assertEqual(scores["full"]["net_corrected_count"], -1)
        self.assertAlmostEqual(scores["full"]["accuracy"], .25)
        self.assertAlmostEqual(scores["selector_only"]["accuracy"], .5)

    def test_macro_f1_includes_absent_labels(self):
        rows = [row(0, (1.0, 0.0, 0.0)), row(1, (1.0, 0.0, 0.0))]
        _, _, scores = module.evaluate_cached(rows, ["A", "B", "C"], .70, .20, set())
        self.assertAlmostEqual(scores["selector_only"]["macro_f1"], 1 / 3)

    def test_rejects_invalid_simplex(self):
        for probs in ((.9, .8, -.7), (.8, .8, 0), (.5, float("nan"), .5),
                      (.4, .4), (.3, .3, .3)):
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                module.evaluate_cached([row(0, probs)], ["A", "B", "C"], .7, .2, set())

    def test_rejects_duplicate_ids_and_unknown_labels(self):
        with self.assertRaises(ValueError):
            module.evaluate_cached([row(0, (.8, .1, .1)), row(0, (.8, .1, .1))],
                                   ["A", "B", "C"], .7, .2, set())
        with self.assertRaises(ValueError):
            module.evaluate_cached([row(0, (.8, .1, .1), "UNKNOWN")],
                                   ["A", "B", "C"], .7, .2, set())
        with self.assertRaises(ValueError):
            module.evaluate_cached([row(0, (.8, .1, .1))],
                                   ["A", "B", "C"], .7, .2, {"UNKNOWN"})

    def test_rejects_invalid_thresholds_and_seed(self):
        rows = [row(0, (.8, .1, .1))]
        for tau, delta, seed in ((float("nan"), .2, 42), (.7, -1, 42), (.7, .2, True)):
            with self.subTest(tau=tau, delta=delta, seed=seed), self.assertRaises(ValueError):
                module.evaluate_cached(rows, ["A", "B", "C"], tau, delta, set(), seed=seed)


if __name__ == "__main__":
    unittest.main()

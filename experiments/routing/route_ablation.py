"""CPU-only SERF routing ablation on aligned cached predictions.

IMPLEMENTED_NOT_RUN on research data. All probabilities must come from one
selector variant and all fusion predictions from its *matching* expert run.
Do not tune thresholds on the held-out test set. No model inference/training.

Use evaluate_cached(rows, label_order, tau, delta, risky_labels, seed), where
rows are dicts with id, true_label, selector_probs, fusion_pred. Row-level
bundling and unique IDs guard against accidental array reordering; they do
not independently establish provenance. Verify checkpoint, split, seed,
label mapping, and sample IDs before interpreting real experimental scores.
"""
import math
import random
from collections import Counter


def _validated_rows(rows, label_order):
    if not rows or len(label_order) < 2 or len(set(label_order)) != len(label_order):
        raise ValueError("Require nonempty rows and >=2 unique class labels")
    required = {"id", "true_label", "selector_probs", "fusion_pred"}
    ids = set()
    clean = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"Row {index} lacks required keys")
        sample_id = row["id"]
        if not isinstance(sample_id, (str, int)) or isinstance(sample_id, bool) or sample_id in ids:
            raise ValueError("Sample IDs must be unique strings or integers")
        ids.add(sample_id)
        if row["true_label"] not in label_order or row["fusion_pred"] not in label_order:
            raise ValueError(f"Unknown class label in row {index}")
        probs = row["selector_probs"]
        if not isinstance(probs, (list, tuple)) or len(probs) != len(label_order):
            raise ValueError(f"Wrong probability-vector width in row {index}")
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
               not math.isfinite(p) or p < 0 or p > 1 for p in probs):
            raise ValueError(f"Invalid probability value in row {index}")
        if abs(math.fsum(probs) - 1.0) > 1e-8:
            raise ValueError(f"Probabilities do not sum to one in row {index}")
        top = max(range(len(probs)), key=lambda j: probs[j])
        ordered = sorted(probs, reverse=True)
        clean.append((row["true_label"], label_order[top], row["fusion_pred"],
                      ordered[0], ordered[0] - ordered[1]))
    return clean


def evaluate_cached(rows, label_order, tau, delta, risky_labels, seed=42):
    """Return masks, trigger diagnostics and scores for frozen cached rows.

    Matched-random is ONE deterministic draw, not a statistical test.
    Macro-F1 averages over all explicit labels (including zero-support labels).
    """
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or not 0 <= x <= 1 for x in (tau, delta)):
        raise ValueError("tau and delta must be finite in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not set(risky_labels) <= set(label_order):
        raise ValueError("Risky labels must belong to label_order")
    data = _validated_rows(rows, label_order)
    n = len(data)
    c = [conf < tau for _, _, _, conf, _ in data]
    m = [margin < delta for _, _, _, _, margin in data]
    r = [pred in risky_labels for _, pred, _, _, _ in data]
    full = [a or b or d for a, b, d in zip(c, m, r)]
    chosen = set(random.Random(seed).sample(range(n), sum(full)))
    masks = {
        "selector_only": [False] * n,
        "confidence_only": c,
        "margin_only": m,
        "risky_class_only": r,
        "no_risky_class": [a or b for a, b in zip(c, m)],
        "full": full,
        "matched_random": [i in chosen for i in range(n)],
        "always_fusion": [True] * n,
    }
    patterns = Counter(f"{int(a)}{int(b)}{int(d)}" for a, b, d in zip(c, m, r))
    diagnostics = {
        "n": n,
        "route_count": sum(full),
        "trigger_counts": {"confidence": sum(c), "margin": sum(m), "risky_class": sum(r)},
        "exclusive_counts": {"confidence": patterns["100"], "margin": patterns["010"],
                             "risky_class": patterns["001"]},
        "trigger_patterns": {f"{i:03b}": patterns[f"{i:03b}"] for i in range(8)},
        "random_seed": seed,
    }
    results = {}
    for name, mask in masks.items():
        y_true = [row[0] for row in data]
        selector_pred = [row[1] for row in data]
        fusion_pred = [row[2] for row in data]
        predicted = [f if route else s for s, f, route in zip(selector_pred, fusion_pred, mask)]
        f1_scores = []
        for label in label_order:
            tp = sum(y == label and p == label for y, p in zip(y_true, predicted))
            fp = sum(y != label and p == label for y, p in zip(y_true, predicted))
            fn = sum(y == label and p != label for y, p in zip(y_true, predicted))
            denom = 2 * tp + fp + fn
            f1_scores.append(2 * tp / denom if denom else 0.0)
        corrected = sum(route and s != y and f == y for (y, s, f, _, _), route in zip(data, mask))
        degraded = sum(route and s == y and f != y for (y, s, f, _, _), route in zip(data, mask))
        results[name] = {
            "n": n, "route_count": sum(mask), "route_rate": sum(mask) / n,
            "accuracy": sum(y == p for y, p in zip(y_true, predicted)) / n,
            "macro_f1": sum(f1_scores) / len(label_order),
            "corrected_count": corrected, "degraded_count": degraded,
            "net_corrected_count": corrected - degraded,
        }
    return masks, diagnostics, results

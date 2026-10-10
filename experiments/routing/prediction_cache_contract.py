"""Validate the provenance contract for CPU-only SERF cached-prediction ablations.

No inference/training. A valid manifest is NOT independent proof that recorded
hashes, sample membership, model outputs, or test-set isolation are genuine.
The exporter must preserve source CSV row indices *before* filtering/splitting.
"""
import math
import re

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_VARIANTS = {
    "development_late_fusion": "probability_interpolation",
    "training_separate_fusion": "separate_fusion_classifier",
}
_FIELDS = {
    "schema_version", "variant", "dataset_sha256", "notebook_path",
    "notebook_commit_sha", "split", "split_seed", "train_seed",
    "label_order", "selector_checkpoint_sha256", "fusion_checkpoint_sha256",
    "fusion_method", "selector_temperature", "roberta_temperature",
    "alpha", "threshold_selection_split", "tau", "delta", "risky_labels",
}


def validate_cache(payload):
    """Return rows for route_ablation.evaluate_cached, fail closed otherwise.

    Every record needs source_row_index from the *original* CSV data rows
    (index 0 = first row after the header). IDs are deterministically derived from dataset
    digest and raw row position; ProjectID alone is not a unique sample ID.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("Expected object with metadata and records")
    m = payload["metadata"]
    if not _FIELDS <= set(m):
        raise ValueError(f"Missing provenance fields: {sorted(_FIELDS - set(m))}")
    if m["schema_version"] != 1 or m["variant"] not in _VARIANTS:
        raise ValueError("Unknown provenance schema or SERF variant")
    for field in ("dataset_sha256", "selector_checkpoint_sha256", "fusion_checkpoint_sha256"):
        if not isinstance(m[field], str) or not _SHA256.fullmatch(m[field]):
            raise ValueError(f"{field} must be an actual SHA-256 digest")
    if not isinstance(m["notebook_commit_sha"], str) or not _GIT_SHA.fullmatch(m["notebook_commit_sha"]):
        raise ValueError("notebook_commit_sha must be a 40-char Git commit")
    expected_path = ("code/" if m["variant"] == "development_late_fusion" else "model-training/")
    if not isinstance(m["notebook_path"], str) or not m["notebook_path"].startswith(expected_path) or not m["notebook_path"].endswith(".ipynb"):
        raise ValueError("Notebook path does not match architecture")
    if m["split"] not in ("validation", "test") or m["threshold_selection_split"] != "validation":
        raise ValueError("Only validation-selected parameters and val/test caches are accepted")
    for field in ("split_seed", "train_seed"):
        if type(m[field]) is not int:
            raise ValueError(f"{field} must be an integer")
    labels = m["label_order"]
    if not isinstance(labels, list) or len(labels) < 2 or not all(isinstance(x, str) and x for x in labels) or len(set(labels)) != len(labels):
        raise ValueError("label_order must be a unique, nonempty class vocabulary")
    if m["fusion_method"] != _VARIANTS[m["variant"]]:
        raise ValueError("Fusion recipe does not match notebook architecture")
    for key in ("tau", "delta"):
        x = m[key]
        if type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    risky = m["risky_labels"]
    if not isinstance(risky, list) or len(set(risky)) != len(risky) or not set(risky) <= set(labels):
        raise ValueError("Invalid risky_labels")
    temperature = m["selector_temperature"]
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid selector_temperature")
    if m["variant"] == "development_late_fusion":
        alpha, temp = m["alpha"], m["roberta_temperature"]
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Invalid alpha")
        if type(temp) not in (int, float) or not math.isfinite(temp) or temp <= 0:
            raise ValueError("Invalid roberta_temperature")
    elif temperature != 1.0 or m["roberta_temperature"] is not None or m["alpha"] is not None:
        raise ValueError("Separate-fusion router is uncalibrated and has no interpolation alpha")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("No prediction records")
    seen = set()
    rows = []
    for record in records:
        if not isinstance(record, dict) or not {"source_row_index", "true_label", "selector_probs", "fusion_pred"} <= set(record):
            raise ValueError("Incomplete prediction record")
        index = record["source_row_index"]
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("Original CSV row positions must be unique nonnegative integers")
        seen.add(index)
        if record["true_label"] not in labels or record["fusion_pred"] not in labels:
            raise ValueError("Prediction label outside manifest vocabulary")
        probs = record["selector_probs"]
        if not isinstance(probs, list) or len(probs) != len(labels) or any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs) or abs(math.fsum(probs) - 1) > 1e-8:
            raise ValueError("Selector probabilities must be a normalized class-aligned vector")
        rows.append({"id": f'{m["dataset_sha256"]}:{index}', "true_label": record["true_label"],
                     "selector_probs": probs, "fusion_pred": record["fusion_pred"]})
    return rows

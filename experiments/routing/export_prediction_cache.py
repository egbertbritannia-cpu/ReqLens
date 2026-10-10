"""Export *existing* model logits into a provenance-checked SERF ablation cache.

IMPLEMENTED_NOT_RUN on original checkpoints. Does not load, train, or infer with
Transformers. Call using precomputed logits from the SAME split, seed, class
order, and corresponding selector and fusion/second-expert checkpoints.

Usage after successful inference, from the repository root:
    from experiments.routing.export_prediction_cache import export_cache
    export_cache(split_manifest, csv_path, source_records, run_metadata,
                 selector_checkpoint_path, fusion_checkpoint_path, output_path)

Each source_record: source_row_index (original CSV row, zero-based), true_label,
selector_logits, fusion_logits. In development_late_fusion, fusion_logits are
RoBERTa expert logits; the function interpolates calibrated probabilities.
In training_separate_fusion, fusion_logits are from the trained fusion classifier.
Never conflate those architectures. All source rows MUST match the reconstructed
split sequence exactly; never export only the subset routed by Full SERF.
"""
import hashlib
import json
import math
from pathlib import Path

from experiments.routing.prediction_cache_contract import validate_cache


_VARIANTS = {
    "development_late_fusion": "probability_interpolation",
    "training_separate_fusion": "separate_fusion_classifier",
}


def _sha256(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _softmax(logits, temperature, expected_width):
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    if (not isinstance(logits, (list, tuple)) or len(logits) != expected_width or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in logits)):
        raise ValueError("Each logits vector must be finite, class-aligned and complete")
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    e = [math.exp(x - m) for x in scaled]
    denom = math.fsum(e)
    return [p / denom for p in e]


def export_cache(split_manifest, dataset_path, source_records, run_metadata,
                 selector_checkpoint_path, fusion_checkpoint_path, output_path):
    """Write an *exclusive* JSON cache, returning the validated payload.

    Checkpoint bytes are hashed but never loaded/executed; a SHA proves content
    identity, NOT scientific provenance or that logits came from those weights.
    Outputs keep status=UNVERIFIED_MODEL_PROVENANCE until independently audited.
    """
    if not isinstance(split_manifest, dict) or split_manifest.get("status") != "SPLIT_RECONSTRUCTED_NOT_PREDICTIONS":
        raise ValueError("Requires split_provenance.build_split_manifest output")
    if not isinstance(run_metadata, dict):
        raise ValueError("run_metadata must be a dictionary")
    split = run_metadata.get("split")
    if split not in ("validation", "test"):
        raise ValueError("Only validation/test predictions are exportable")
    if split_manifest.get("split_seed") != run_metadata.get("split_seed"):
        raise ValueError("Split seed does not match manifest")
    labels = split_manifest.get("label_order")
    if run_metadata.get("label_order") != labels:
        raise ValueError("Label order does not match manifest")
    if _sha256(dataset_path) != split_manifest.get("dataset_sha256"):
        raise ValueError("Dataset bytes do not match split manifest")
    if not isinstance(source_records, list):
        raise ValueError("source_records must be a complete ordered list")
    expected = split_manifest.get("splits", {}).get(split)
    if not isinstance(expected, list) or len(expected) != len(source_records) or not expected:
        raise ValueError("Incomplete or missing split predictions")
    variant = run_metadata.get("variant")
    if variant not in _VARIANTS:
        raise ValueError("Unknown SERF architecture")
    alpha = run_metadata.get("alpha")
    if variant == "development_late_fusion":
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Development fusion requires alpha in [0, 1]")
        other_temp = run_metadata.get("roberta_temperature")
        if type(other_temp) not in (int, float) or not math.isfinite(other_temp) or other_temp <= 0:
            raise ValueError("Development fusion requires valid RoBERTa temperature")
    elif run_metadata.get("selector_temperature") != 1.0 or alpha is not None or run_metadata.get("roberta_temperature") is not None:
        raise ValueError("Separate fusion uses uncalibrated selector and no alpha")

    meta = dict(run_metadata)
    meta.update({
        "schema_version": 1,
        "dataset_sha256": split_manifest["dataset_sha256"],
        "selector_checkpoint_sha256": _sha256(selector_checkpoint_path),
        "fusion_checkpoint_sha256": _sha256(fusion_checkpoint_path),
        "fusion_method": _VARIANTS[variant],
        "label_order": labels,
    })
    rows = []
    for actual, requested in zip(source_records, expected):
        if (not isinstance(actual, dict) or type(actual.get("source_row_index")) is not int or
                actual["source_row_index"] != requested.get("source_row_index") or
                actual.get("true_label") != requested.get("label")):
            raise ValueError("Incorrect per-sample order/ID/ground-truth label")
        selector = _softmax(actual.get("selector_logits"), meta.get("selector_temperature"), len(labels))
        other = _softmax(actual.get("fusion_logits"),
                         meta["roberta_temperature"] if variant == "development_late_fusion" else 1.0, len(labels))
        fusion = ([alpha * a + (1 - alpha) * b for a, b in zip(selector, other)]
                  if variant == "development_late_fusion" else other)
        rows.append({
            "source_row_index": actual["source_row_index"],
            "true_label": actual["true_label"],
            "selector_probs": selector,
            "fusion_pred": labels[max(range(len(labels)), key=fusion.__getitem__)],
        })
    payload = {"status": "UNVERIFIED_MODEL_PROVENANCE", "metadata": meta, "records": rows}
    validated = validate_cache(payload)
    if len(validated) != len(expected):
        raise AssertionError("Cache contract lost sample rows")
    # Never overwrite a prior experiment or cache.
    with open(output_path, "x", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    return payload

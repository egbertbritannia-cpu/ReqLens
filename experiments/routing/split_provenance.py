"""Reconstruct the original SERF notebook split with stable CSV row lineage.

CPU-only; no tokenizer, model loading, training or network access.
Preprocessing / two-stage stratified split mirror both no-augmentation notebooks.
The output is a proposed lineage manifest, not verified model predictions.

Usage:
    python experiments/routing/split_provenance.py \
        --csv data/exp/promise_exp.csv --output local_split_manifest.json
Output is written exclusively (never overwritten). Do not commit prediction data.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


SPLIT_SEED = 42


def build_split_manifest(csv_path, split_seed=SPLIT_SEED):
    """Return a traceable split, preserving zero-based indices of raw CSV rows."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if type(split_seed) is not int:
        raise ValueError("split_seed must be an integer")

    dataset_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    original = pd.read_csv(csv_path)
    if not {"RequirementText", "class"} <= set(original.columns):
        raise ValueError("Expected RequirementText and class columns")
    # These are original CSV body row positions before any dropna/filter/reset.
    frame = original.assign(source_row_index=range(len(original)))
    frame = frame[["RequirementText", "class", "source_row_index"]].dropna().reset_index(drop=True)
    frame["RequirementText"] = frame["RequirementText"].astype(str).map(
        lambda text: re.sub(r"\s+", " ", str(text)).strip()
    )
    frame["class"] = frame["class"].astype(str).str.strip().str.upper()
    frame = frame[frame["RequirementText"] != ""].reset_index(drop=True)
    frame = frame[frame["class"].map(frame["class"].value_counts()) > 1].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No eligible rows after notebook preprocessing")
    encoder = LabelEncoder()
    frame["label"] = encoder.fit_transform(frame["class"])

    # Exact same scikit-learn calls and order as BOTH SERF notebooks.
    train, val_test = train_test_split(
        frame, test_size=0.3, stratify=frame["label"], random_state=split_seed
    )
    validation, test = train_test_split(
        val_test, test_size=0.5, stratify=val_test["label"], random_state=split_seed
    )
    splits = {}
    for split_name, subset in (("train", train), ("validation", validation), ("test", test)):
        # .itertuples() may rename the CSV column "class"; use explicit columns.
        splits[split_name] = [
            {"source_row_index": int(source_id), "label": str(label)}
            for source_id, label in subset[["source_row_index", "class"]].itertuples(index=False, name=None)
        ]
    seen = [r["source_row_index"] for items in splits.values() for r in items]
    if len(seen) != len(frame) or len(seen) != len(set(seen)):
        raise AssertionError("Duplicate or missing source rows after splitting")
    return {
        "schema_version": 1,
        "status": "SPLIT_RECONSTRUCTED_NOT_PREDICTIONS",
        "dataset_sha256": dataset_sha256,
        "split_seed": split_seed,
        "source_row_index_semantics": "zero-based row position after CSV header, before filtering",
        "label_order": list(map(str, encoder.classes_)),
        "total_raw_rows": int(len(original)),
        "eligible_rows": int(len(frame)),
        "splits": splits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Local CSV matching the notebook dataset")
    parser.add_argument("--output", required=True, help="New JSON filename; existing output is never overwritten")
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    manifest = build_split_manifest(args.csv, args.split_seed)
    with open(args.output, "x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print("SPLIT_RECONSTRUCTED_NOT_PREDICTIONS", {k: len(v) for k, v in manifest["splits"].items()})


if __name__ == "__main__":
    main()

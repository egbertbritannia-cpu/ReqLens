"""Opt-in logits collection for the two original SERF notebook architectures.

IMPLEMENTED_NOT_RUN ON ORIGINAL MODELS. No pretrained model/checkpoint loads,
training, tuning or outputs are performed on import. The caller MUST supply
already-loaded models, original dataset bytes and frozen split manifest.

Build the loader with ``make_manifest_loader`` rather than a notebook DataLoader:
notebook datasets omit stable sample IDs, so existing loaders cannot prove
sample alignment. ``collect_pair_logits`` refuses any loader without the
manifest-derived ``source_row_indices`` and a SequentialSampler.

The result is in the input schema of ``export_prediction_cache.export_cache``;
that exporter performs checkpoint hashing and atomic provenance validation.
The caller must independently verify weights/checkpoints used by the models.
"""
import hashlib
import math
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler


VARIANTS = frozenset({"training_separate_fusion", "development_late_fusion"})


def make_manifest_loader(csv_path, manifest, split, dataset_class, batch_size):
    """Rebuild the original notebook dataset in stable split order.

    The supplied dataset_class is the *existing* ElectraRobertaDataset; it uses
    the user's already-initialized notebook tokenizers. This function does not
    download models or tokenizers.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(manifest, dict) or manifest.get("status") != "SPLIT_RECONSTRUCTED_NOT_PREDICTIONS":
        raise ValueError("Expected a frozen split_provenance manifest")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("dataset_sha256"):
        raise ValueError("Dataset SHA-256 differs from the frozen split manifest")
    if split not in ("validation", "test"):
        raise ValueError("Only validation/test splits can be exported")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    expected = manifest.get("splits", {}).get(split)
    labels = manifest.get("label_order")
    if (not isinstance(expected, list) or not expected or not isinstance(labels, list) or
            len(labels) < 2 or len(set(labels)) != len(labels)):
        raise ValueError("Incomplete split or label mapping")

    raw = pd.read_csv(path)
    if not {"RequirementText", "class"} <= set(raw.columns):
        raise ValueError("Missing source CSV fields")
    classes = {label: i for i, label in enumerate(labels)}
    data = []
    ids = []
    for entry in expected:
        if not isinstance(entry, dict) or type(entry.get("source_row_index")) is not int:
            raise ValueError("Manifest has invalid source row ID")
        idx = entry["source_row_index"]
        if idx < 0 or idx >= len(raw) or idx in ids:
            raise ValueError("Out-of-range or duplicated source row ID")
        source = raw.iloc[idx]
        text, label = source["RequirementText"], source["class"]
        if pd.isna(text) or pd.isna(label):
            raise ValueError("Split contains a source row with missing text or label")
        text = re.sub(r"\s+", " ", str(text)).strip()
        actual_label = str(label).strip().upper()
        if not text or actual_label != entry.get("label") or actual_label not in classes:
            raise ValueError("CSV row differs from expected split label or is empty")
        ids.append(idx)
        data.append({"model_text": text, "label": classes[actual_label],
                     "source_row_index": idx})
    dataset = dataset_class(pd.DataFrame(data))
    if len(dataset) != len(data):
        raise ValueError("Notebook Dataset length differs from frozen manifest")
    if getattr(dataset, "source_row_indices", None) is not None:
        raise ValueError("Dataset already carries source indices; refusing overwrite")
    dataset.source_row_indices = tuple(ids)
    dataset.provenance_labels = tuple(row["label"] for row in data)
    # A notebook Dataset can expose a separate .labels list; reject mismatch.
    if hasattr(dataset, "labels") and list(dataset.labels) != list(dataset.provenance_labels):
        raise ValueError("Notebook Dataset labels differ from frozen split")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def collect_pair_logits(selector_model, fusion_model, loader, device, variant, manifest, split):
    """Return a complete list of records, without saving or changing models.

    ``training_separate_fusion`` expects two independently trained models:
    selector_model(**batch) and fusion_model(**batch) return {'logits': Tensor}.

    ``development_late_fusion`` expects selector_model to be the trained hybrid
    with encode_electra/encode_roberta and two independent logits heads;
    fusion_model MUST be None. The returned fusion_logits are the RoBERTa head
    logits; interpolation is performed later in export_prediction_cache.
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown SERF architecture")
    if variant == "development_late_fusion" and fusion_model is not None:
        raise ValueError("Development variant must use a single paired-expert model")
    if variant == "training_separate_fusion" and fusion_model is None:
        raise ValueError("Separate fusion requires the independent fusion model")
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Inference loader must preserve order and all samples")
    expected = manifest.get("splits", {}).get(split) if isinstance(manifest, dict) else None
    labels = manifest.get("label_order") if isinstance(manifest, dict) else None
    ds = loader.dataset
    ids = getattr(ds, "source_row_indices", None)
    observed = getattr(ds, "provenance_labels", None)
    if not isinstance(expected, list) or not expected or not isinstance(labels, list):
        raise ValueError("Invalid split manifest")
    if ids is None or observed is None or len(ids) != len(expected) or len(ds) != len(expected):
        raise ValueError("DataLoader must come from the manifest-backed dataset")
    for i, item in enumerate(expected):
        if ids[i] != item["source_row_index"] or observed[i] != labels.index(item["label"]):
            raise ValueError("Dataset row order or label mapping differs from split manifest")
    for model in (selector_model, fusion_model):
        if model is not None:
            model.eval()
    result = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            if "labels" not in batch:
                raise ValueError("Missing labels in notebook batch")
            y = batch["labels"].detach().cpu().tolist()
            if not isinstance(y, list) or not y or len(y) + offset > len(expected):
                raise ValueError("Unexpected batch length")
            if y != list(observed[offset:offset + len(y)]):
                raise ValueError("Batch labels are not aligned to the manifest")
            move = {k: v.to(device) for k, v in batch.items()}
            if variant == "training_separate_fusion":
                a = selector_model(**move)["logits"]
                b = fusion_model(**move)["logits"]
            else:
                e = selector_model.encode_electra(move["electra_input_ids"],
                                                 move["electra_attention_mask"],
                                                 move.get("electra_token_type_ids"))
                r = selector_model.encode_roberta(move["roberta_input_ids"],
                                                 move["roberta_attention_mask"])
                a = selector_model.selector_logits_from_representation(e)
                b = selector_model.roberta_logits_from_representation(r)
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or tuple(a.shape) != (len(y), len(labels)):
                raise ValueError("Selector and fusion logits have inconsistent shapes")
            if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
                raise ValueError("Model produced non-finite logits")
            for j, label_idx in enumerate(y):
                result.append({"source_row_index": ids[offset + j],
                               "true_label": labels[label_idx],
                               "selector_logits": [float(x) for x in a[j].detach().cpu().tolist()],
                               "fusion_logits": [float(x) for x in b[j].detach().cpu().tolist()]})
            offset += len(y)
    if offset != len(expected):
        raise ValueError("Incomplete logits collection")
    return result

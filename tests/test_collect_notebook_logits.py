"""CPU-only synthetic tests. No notebook imports, downloads, or pretrained weights."""
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from experiments.routing.collect_notebook_logits import make_manifest_loader, collect_pair_logits


class FakeNotebookDataset(Dataset):
    def __init__(self, frame):
        self.labels = frame["label"].astype(int).tolist()
        self.texts = frame["model_text"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Distinct deterministic feature per source text to detect ordering.
        value = 1 if self.texts[index] == "alpha" else 2
        return {"labels": torch.tensor(self.labels[index]),
                "electra_input_ids": torch.tensor([value]),
                "electra_attention_mask": torch.tensor([1]),
                "roberta_input_ids": torch.tensor([value]),
                "roberta_attention_mask": torch.tensor([1])}


class FakeExpert(torch.nn.Module):
    def __init__(self, reverse=False):
        super().__init__()
        self.reverse = reverse
        self.seen_batches = []

    def forward(self, **batch):
        n = len(batch["labels"])
        self.seen_batches.append(n)
        x = batch["electra_input_ids"][:, 0].float()
        v = torch.stack((x, -x), dim=-1)
        return {"logits": torch.flip(v, dims=(-1,)) if self.reverse else v}


class FakeHybrid(torch.nn.Module):
    def encode_electra(self, ids, masks, token_type_ids=None):
        return ids.float()

    def encode_roberta(self, ids, masks):
        return ids.float()

    def selector_logits_from_representation(self, rep):
        return torch.cat([rep, -rep], dim=1)

    def roberta_logits_from_representation(self, rep):
        return torch.cat([-rep, rep], dim=1)


class CollectNotebookLogitsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "data.csv"
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["RequirementText", "class"])
            w.writerow(["  alpha\t", "A"])
            w.writerow(["ignored", "A"])
            w.writerow([" beta  ", "B"])
        self.manifest = {"status": "SPLIT_RECONSTRUCTED_NOT_PREDICTIONS",
                         "dataset_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                         "label_order": ["A", "B"],
                         "splits": {"validation": [{"source_row_index": 2, "label": "B"},
                                                    {"source_row_index": 0, "label": "A"}]}}

    def loader(self):
        return make_manifest_loader(self.path, self.manifest, "validation", FakeNotebookDataset, 1)

    def test_reconstruct_original_source_order_and_normalization(self):
        loader = self.loader()
        self.assertEqual(loader.dataset.texts, ["beta", "alpha"])
        self.assertEqual(loader.dataset.source_row_indices, (2, 0))
        self.assertEqual(loader.dataset.labels, [1, 0])

    def test_training_two_model_logits_all_rows(self):
        s, f = FakeExpert(), FakeExpert(reverse=True)
        result = collect_pair_logits(s, f, self.loader(), "cpu", "training_separate_fusion", self.manifest, "validation")
        self.assertEqual([r["source_row_index"] for r in result], [2, 0])
        self.assertEqual([r["true_label"] for r in result], ["B", "A"])
        self.assertEqual(result[0]["selector_logits"], [2.0, -2.0])
        self.assertEqual(result[1]["fusion_logits"], [-1.0, 1.0])
        self.assertEqual(s.seen_batches, [1, 1])
        self.assertEqual(f.seen_batches, [1, 1])

    def test_development_hybrid_two_heads(self):
        result = collect_pair_logits(FakeHybrid(), None, self.loader(), "cpu", "development_late_fusion", self.manifest, "validation")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["fusion_logits"], [-2.0, 2.0])

    def test_missing_origin_provenance_rejected(self):
        old_loader = DataLoader(FakeNotebookDataset(__import__('pandas').DataFrame({"model_text": ["alpha", "beta"], "label": [0, 1]})), batch_size=1)
        with self.assertRaisesRegex(ValueError, "manifest-backed"):
            collect_pair_logits(FakeExpert(), FakeExpert(True), old_loader, "cpu", "training_separate_fusion", self.manifest, "validation")

    def test_random_shuffle_rejected(self):
        source = self.loader().dataset
        shuffled = DataLoader(source, batch_size=1, shuffle=True)
        with self.assertRaisesRegex(ValueError, "preserve order"):
            collect_pair_logits(FakeExpert(), FakeExpert(True), shuffled, "cpu", "training_separate_fusion", self.manifest, "validation")

    def test_drop_last_rejected(self):
        truncated = DataLoader(self.loader().dataset, batch_size=2, drop_last=True)
        with self.assertRaisesRegex(ValueError, "preserve order"):
            collect_pair_logits(FakeExpert(), FakeExpert(True), truncated, "cpu", "training_separate_fusion", self.manifest, "validation")

    def test_tampered_csv_hash_rejected(self):
        self.path.write_text("bad", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.loader()

    def test_csv_label_mismatch_rejected(self):
        self.manifest["splits"]["validation"][0]["label"] = "A"
        with self.assertRaisesRegex(ValueError, "differs from expected"):
            self.loader()

    def test_manifest_index_tamper_rejected(self):
        loader = self.loader()
        loader.dataset.source_row_indices = (0, 2)
        with self.assertRaisesRegex(ValueError, "row order"):
            collect_pair_logits(FakeExpert(), FakeExpert(True), loader, "cpu", "training_separate_fusion", self.manifest, "validation")

    def test_nonfinite_logits_rejected(self):
        class Bad(FakeExpert):
            def forward(self, **batch):
                return {"logits": torch.full((len(batch["labels"]), 2), float("nan"))}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            collect_pair_logits(Bad(), FakeExpert(True), self.loader(), "cpu", "training_separate_fusion", self.manifest, "validation")

    def test_wrong_architecture_rejected(self):
        with self.assertRaisesRegex(ValueError, "single paired-expert"):
            collect_pair_logits(FakeHybrid(), FakeExpert(), self.loader(), "cpu", "development_late_fusion", self.manifest, "validation")
        with self.assertRaisesRegex(ValueError, "requires"):
            collect_pair_logits(FakeExpert(), None, self.loader(), "cpu", "training_separate_fusion", self.manifest, "validation")

    def test_batch_label_tamper_rejected(self):
        loader = self.loader()
        loader.dataset.labels = [0, 0]
        with self.assertRaisesRegex(ValueError, "Batch labels"):
            collect_pair_logits(FakeExpert(), FakeExpert(True), loader, "cpu", "training_separate_fusion", self.manifest, "validation")


if __name__ == "__main__":
    unittest.main()

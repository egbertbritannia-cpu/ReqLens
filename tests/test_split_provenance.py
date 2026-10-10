"""Check zero-based lineage against the notebooks' legacy split implementation."""
import csv
import hashlib
import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PATH = Path(__file__).resolve().parents[1] / 'experiments/routing/split_provenance.py'
spec = importlib.util.spec_from_file_location('split_provenance', PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class SplitProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.csv_path = Path(self.work.name) / 'promise_exp.csv'
        with self.csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ProjectID', 'RequirementText', 'class'])
            writer.writerow(['p0', '', 'A'])       # dropped after normalization
            writer.writerow(['p1', '  one\n', 'B'])  # preserved original row index
            for i in range(25):
                for cls in 'ABC':
                    writer.writerow(['project', f'  value   {i}  {cls} ', cls.lower()])
            writer.writerow(['p2', 'singleton', 'Z'])
            writer.writerow(['p3', '', 'C'])

    def test_follows_exact_legacy_split_in_membership_and_order(self):
        manifest = mod.build_split_manifest(self.csv_path)
        original = pd.read_csv(self.csv_path)
        old = original[['RequirementText', 'class']].dropna().reset_index(drop=True)
        old['RequirementText'] = old['RequirementText'].astype(str).map(lambda s: re.sub(r'\s+', ' ', str(s)).strip())
        old['class'] = old['class'].astype(str).str.strip().str.upper()
        old = old[old['RequirementText'] != ''].reset_index(drop=True)
        old = old[old['class'].map(old['class'].value_counts()) > 1].reset_index(drop=True)
        old['label'] = LabelEncoder().fit_transform(old['class'])
        train, rem = train_test_split(old, test_size=.3, stratify=old['label'], random_state=42)
        val, test = train_test_split(rem, test_size=.5, stratify=rem['label'], random_state=42)
        # Independently reconstruct raw row IDs via normalized text (unique in
        # the fixture), not from the implementation under test.
        original_text_to_raw_id = {}
        for source_id, text in enumerate(original['RequirementText']):
            if isinstance(text, str) and text.strip():
                original_text_to_raw_id[re.sub(r'\s+', ' ', text).strip()] = source_id
        for name, legacy in [('train', train), ('validation', val), ('test', test)]:
            reconstructed = manifest['splits'][name]
            self.assertEqual([r['label'] for r in reconstructed], legacy['class'].tolist())
            expected_ids = [original_text_to_raw_id[text] for text in legacy['RequirementText']]
            self.assertEqual([r['source_row_index'] for r in reconstructed], expected_ids)
        self.assertEqual(manifest['eligible_rows'], len(old))
        self.assertEqual(manifest['label_order'], ['A', 'B', 'C'])
        self.assertEqual(manifest['dataset_sha256'], hashlib.sha256(self.csv_path.read_bytes()).hexdigest())

    def test_original_index_survives_filtering_and_splits(self):
        m = mod.build_split_manifest(self.csv_path)
        entries = [r for split in m['splits'].values() for r in split]
        indexes = [r['source_row_index'] for r in entries]
        self.assertEqual(len(indexes), len(set(indexes)))
        self.assertNotIn(0, indexes)
        self.assertNotIn(77, indexes)
        self.assertNotIn(78, indexes)
        self.assertIn(1, indexes)
        self.assertEqual(m['total_raw_rows'], 79)

    def test_deterministic_and_seed_changes_membership(self):
        m = mod.build_split_manifest(self.csv_path)
        again = mod.build_split_manifest(self.csv_path)
        self.assertEqual(m, again)
        shifted = mod.build_split_manifest(self.csv_path, split_seed=99)
        self.assertNotEqual(m['splits']['train'], shifted['splits']['train'])

    def test_missing_required_columns_rejected(self):
        with self.csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ProjectID', 'RequirementText'])
            writer.writerow(['p', 'hello'])
        with self.assertRaisesRegex(ValueError, 'Expected RequirementText'):
            mod.build_split_manifest(self.csv_path)

    def test_invalid_seed_rejected(self):
        with self.assertRaises(ValueError):
            mod.build_split_manifest(self.csv_path, True)


if __name__ == '__main__':
    unittest.main()

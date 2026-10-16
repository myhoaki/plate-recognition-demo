import tempfile
import unittest
from pathlib import Path

from app.datasets import json_manifest_sha256, tree_sha256, _split_counts


class DatasetSnapshotHelpersTest(unittest.TestCase):
    def test_split_counts_are_nonempty_and_sum_to_sample_count(self):
        for sample_count in (3, 4, 10, 103):
            counts = _split_counts(sample_count)
            self.assertEqual(sum(counts), sample_count)
            self.assertTrue(all(count > 0 for count in counts))

    def test_manifest_hash_is_stable_and_sensitive_to_content(self):
        first = {"task": "detector", "samples": [{"image_id": "a", "split": "test"}]}
        reordered = {"samples": [{"split": "test", "image_id": "a"}], "task": "detector"}
        changed = {"task": "detector", "samples": [{"image_id": "b", "split": "test"}]}
        self.assertEqual(json_manifest_sha256(first), json_manifest_sha256(reordered))
        self.assertNotEqual(json_manifest_sha256(first), json_manifest_sha256(changed))

    def test_tree_hash_detects_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "train" / "crop.png"
            sample.parent.mkdir()
            sample.write_bytes(b"first")
            original = tree_sha256(root)
            sample.write_bytes(b"second")
            self.assertNotEqual(original, tree_sha256(root))


if __name__ == "__main__":
    unittest.main()

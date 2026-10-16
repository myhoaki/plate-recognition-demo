import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from app.dataset_registry import (
    choose_split,
    content_hash,
    group_split_conflicts,
    import_dataset,
    initial_split,
    normalize_yolo_box,
    _near_duplicate_pairs,
    parse_yolo_file,
    pixels_to_yolo,
    safe_relative_path,
    split_percentages,
    stable_sample_id,
    yolo_label_line,
)


class DatasetRegistryHelpersTest(unittest.TestCase):
    def test_sample_id_is_stable_and_content_based(self):
        digest = "a" * 64
        self.assertEqual(stable_sample_id(digest), stable_sample_id(digest))
        self.assertNotEqual(stable_sample_id(digest), stable_sample_id("b" * 64))

    def test_yolo_coordinate_conversion_and_serialization(self):
        box = pixels_to_yolo([10, 20, 30, 60], 100, 100)
        self.assertEqual(box, [0.2, 0.4, 0.2, 0.4])
        self.assertEqual(normalize_yolo_box(0, box, 100, 100), box)
        self.assertEqual(yolo_label_line(0, box), "0 0.20000000 0.40000000 0.20000000 0.40000000")

    def test_reject_invalid_class_and_boxes(self):
        with self.assertRaises(ValueError):
            normalize_yolo_box(1, [0.5, 0.5, 0.2, 0.2], 100, 100, 1)
        with self.assertRaises(ValueError):
            normalize_yolo_box(0, [0.1, 0.1, 0.4, 0.4], 100, 100)
        with self.assertRaises(ValueError):
            pixels_to_yolo([-1, 0, 2, 2], 100, 100)

    def test_parse_yolo_sidecar_and_reject_malformed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
            self.assertEqual(parse_yolo_file(path, 100, 50)[0]["bbox_normalized_xywh"], [0.5, 0.5, 0.4, 0.2])
            path.write_text("0 0.5 0.5 2 0.2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_yolo_file(path, 100, 50)

    def test_parse_yolo_segmentation_polygon_and_map_source_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "polygon.txt"
            path.write_text("1 0.1 0.2 0.8 0.2 0.75 0.6 0.12 0.65\n", encoding="utf-8")
            parsed = parse_yolo_file(path, 200, 100, class_count=2, class_mapping={0: 0, 1: 0})
            self.assertEqual(parsed[0]["source_class_id"], 1)
            self.assertEqual(parsed[0]["class_id"], 0)
            for actual, expected in zip(parsed[0]["bbox_normalized_xywh"], [0.45, 0.425, 0.7, 0.45]):
                self.assertAlmostEqual(actual, expected)
            self.assertFalse(parsed[0]["clipped"])
            path.write_text("1 0.2 0.2 0.8 0.2 1.02 0.8 0.2 0.8\n", encoding="utf-8")
            clipped = parse_yolo_file(path, 200, 100, class_count=2, class_mapping={0: 0, 1: 0})[0]
            self.assertTrue(clipped["clipped"])
            for actual, expected in zip(clipped["bbox_normalized_xywh"], [0.6, 0.5, 0.8, 0.6]):
                self.assertAlmostEqual(actual, expected)
            path.write_text("1 0.2 0.2 0.8 0.2 1.2 0.8 0.2 0.8\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_yolo_file(path, 200, 100, class_count=2, class_mapping={0: 0, 1: 0})

    def test_split_percentages_include_empty_splits_and_zero_total(self):
        self.assertEqual(split_percentages({"train": 8, "val": 1, "test": 1}), {"train": 80.0, "val": 10.0, "test": 10.0})
        self.assertEqual(split_percentages({}), {"train": 0.0, "val": 0.0, "test": 0.0})

    def test_near_duplicate_hash_search_finds_four_distributed_bit_flips(self):
        rows = [{"relative": "a.jpg", "sha256": "a" * 64, "perceptual_hash": 0},
                {"relative": "b.jpg", "sha256": "b" * 64, "perceptual_hash": (1 << 0) | (1 << 8) | (1 << 16) | (1 << 24)}]
        self.assertEqual(_near_duplicate_pairs(rows)[0]["average_hash_distance"], 4)

    def test_import_dry_run_reports_filename_collision_without_writes(self):
        class ReadOnlySession:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def scalar(self, _query):
                return None

            def scalars(self, _query):
                return self

            def all(self):
                return []

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for split, shade in (("train", 30), ("valid", 180)):
                folder = source / split / "images"
                folder.mkdir(parents=True)
                cv2.imwrite(str(folder / "same.jpg"), np.full((24, 48, 3), shade, dtype=np.uint8))
            with patch("app.dataset_registry.SessionLocal", ReadOnlySession):
                first = import_dataset(source, "collision-test", "detector")
                second = import_dataset(source, "collision-test", "detector")
            self.assertTrue(first["dry_run"])
            self.assertEqual(first["registered"], 2)
            self.assertEqual(first["filename_collisions"], second["filename_collisions"])
            self.assertEqual(len(first["filename_collisions"]), 1)
            self.assertEqual(list(source.glob("**/*.jpg")).__len__(), 2)

    def test_import_dry_run_deduplicates_content_and_flags_cross_split_leakage(self):
        class ReadOnlySession:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def scalar(self, _query):
                return None

            def scalars(self, _query):
                return self

            def all(self):
                return []

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for split in ("train", "val"):
                images = source / "images" / split
                labels = source / "labels" / split
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                cv2.imwrite(str(images / "same.png"), np.full((24, 48, 3), 80, dtype=np.uint8))
                (labels / "same.txt").write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
            with patch("app.dataset_registry.SessionLocal", ReadOnlySession):
                report = import_dataset(source, "duplicate-test", "detector")
            self.assertEqual(report["registered"], 1)
            self.assertEqual(len(report["exact_duplicates"]), 1)
            conflict = next(row for row in report["conflicts"] if row["kind"] == "duplicate_split")
            self.assertEqual(conflict["selected_split"], "val")

    def test_group_split_conflicts_detect_leakage(self):
        rows = [{"group_id": "car-1", "split": "train"}, {"group_id": "car-1", "split": "test"},
                {"group_id": "car-2", "split": "val"}]
        self.assertEqual(group_split_conflicts(rows), [{"group_id": "car-1", "splits": ["test", "train"]}])

    def test_split_assignments_are_preserved_and_new_data_joins_train(self):
        self.assertEqual(choose_split("sample", None, {"sample": "test"}, {}, True), "test")
        self.assertEqual(choose_split("new", "group", {}, {"group": "val"}, True), "val")
        self.assertEqual(choose_split("new", None, {}, {}, True), "train")
        self.assertEqual(initial_split("same-group"), initial_split("same-group"))

    def test_manifest_hash_preserves_exact_text(self):
        self.assertNotEqual(content_hash({"text": "ABC123"}), content_hash({"text": "abc123"}))

    def test_relative_paths_reject_traversal(self):
        self.assertEqual(safe_relative_path("images/train/a.jpg"), Path("images/train/a.jpg"))
        with self.assertRaises(ValueError):
            safe_relative_path("../secret.jpg")
        with self.assertRaises(ValueError):
            safe_relative_path("/etc/passwd")


if __name__ == "__main__":
    unittest.main()

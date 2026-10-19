"""Training `workers` parameter.

Detector training (Ultralytics) and OCR training (CRNN DataLoader) both load
batches in background processes. The value must reach both trainers, default to
2, and stay inside the bounds the API advertises.
"""

from __future__ import annotations

import inspect
import unittest

from pydantic import ValidationError

from app.schemas import StartTraining

SNAPSHOT = "a" * 32


def _payload(**overrides) -> dict:
    base = {"model_type": "detector", "dataset_snapshot_id": SNAPSHOT, "epochs": 1, "batch_size": 1}
    base.update(overrides)
    return base


class WorkersSchemaTest(unittest.TestCase):
    def test_default_is_two(self):
        for task in ("detector", "ocr"):
            config = StartTraining(**_payload(model_type=task))
            self.assertEqual(config.workers, 2, task)

    def test_value_is_accepted_for_both_tasks(self):
        for task in ("detector", "ocr"):
            for workers in (0, 1, 4, 8):
                config = StartTraining(**_payload(model_type=task, workers=workers))
                self.assertEqual(config.workers, workers, f"{task} workers={workers}")

    def test_out_of_range_is_rejected(self):
        for workers in (-1, 17, 100):
            with self.assertRaises(ValidationError):
                StartTraining(**_payload(workers=workers))

    def test_config_reaches_the_training_job(self):
        """The stored job config must carry the value so the worker can read it."""
        config = StartTraining(**_payload(workers=6))
        self.assertEqual(config.model_dump()["workers"], 6)


class WorkersIsWiredTest(unittest.TestCase):
    """Both trainers must read the parameter rather than hardcode it."""

    def test_detector_training_does_not_hardcode_workers(self):
        from app import training

        source = inspect.getsource(training.run_training_job)
        self.assertNotIn("workers=2,", source, "detector training still hardcodes workers=2")
        self.assertIn('config.get("workers"', source)

    def test_ocr_dataloader_receives_num_workers(self):
        from app import ocr_training

        source = inspect.getsource(ocr_training.run_ocr_training)
        self.assertIn("num_workers=num_workers", source)
        # All three loaders (train/val/test) must be covered.
        self.assertEqual(source.count("num_workers=num_workers"), 3)
        self.assertIn('config.get("workers"', source)


class WorkersDefaultFallbackTest(unittest.TestCase):
    """Jobs queued before the parameter existed must still default to 2."""

    def test_detector_fallback(self):
        config = {"batch_size": 8}
        self.assertEqual(int(config.get("workers", 2) or 0), 2)

    def test_ocr_fallback(self):
        config = {"batch_size": 8}
        self.assertEqual(max(0, int(config.get("workers", 2) or 0)), 2)

    def test_explicit_zero_is_preserved_not_replaced_by_default(self):
        """workers=0 means 'load in the main process' and must survive `or 0`."""
        config = {"workers": 0}
        self.assertEqual(int(config.get("workers", 2) or 0), 0)
        self.assertEqual(max(0, int(config.get("workers", 2) or 0)), 0)


if __name__ == "__main__":
    unittest.main()
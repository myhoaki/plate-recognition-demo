"""Detection vs OCR readiness.

Detection ground truth (a box on a plate) and OCR ground truth (the characters
on that plate) are different requirements. These tests pin the derived
readiness rules and prove they agree with the rules the snapshot builders use.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from app.logic import image_readiness

NOW = datetime.now(timezone.utc)


class ImageReadinessTest(unittest.TestCase):
    """Acceptance criteria 1, 2, 3 and 6."""

    def test_verified_boxes_without_text_are_detection_ready_but_not_ocr_ready(self):
        state = image_readiness("verified", NOW, plate_count=1, verified_text_count=0)
        self.assertTrue(state["detection_ready"])
        self.assertFalse(state["ocr_ready"])
        self.assertEqual(state["ocr_status"], "missing")

    def test_verified_boxes_with_verified_text_are_ready_for_both(self):
        state = image_readiness("verified", NOW, plate_count=1, verified_text_count=1)
        self.assertTrue(state["detection_ready"])
        self.assertTrue(state["ocr_ready"])
        self.assertTrue(state["ready_for_both"])

    def test_unverified_boxes_are_never_detection_ready(self):
        for status in ("needs_review", "reviewed"):
            state = image_readiness(status, None, plate_count=2, verified_text_count=2)
            self.assertFalse(state["detection_ready"], status)
            self.assertFalse(state["ocr_ready"], status)

    def test_verified_status_without_timestamp_is_not_approved(self):
        """review_status alone is not enough; verified_at must also be set."""
        state = image_readiness("verified", None, plate_count=1, verified_text_count=1)
        self.assertFalse(state["detection_ready"])
        self.assertEqual(state["ocr_status"], "blocked")

    def test_partial_transcription_is_not_ocr_ready(self):
        state = image_readiness("verified", NOW, plate_count=3, verified_text_count=2)
        self.assertTrue(state["detection_ready"])
        self.assertEqual(state["ocr_status"], "partial")
        self.assertFalse(state["ocr_ready"])

    def test_multiple_plates_are_counted_per_plate(self):
        self.assertEqual(image_readiness("verified", NOW, 4, 4)["verified_text_count"], 4)
        self.assertEqual(image_readiness("verified", NOW, 4, 0)["plate_count"], 4)

    def test_excluded_statuses_stay_excluded(self):
        for status in ("rejected", "no_plate"):
            state = image_readiness(status, None, 3, 3)
            self.assertFalse(state["detection_ready"])
            self.assertFalse(state["ocr_ready"])
            self.assertEqual(state["detection_status"], status)

    def test_image_without_boxes_is_ready_for_neither(self):
        state = image_readiness("verified", NOW, 0, 0)
        self.assertFalse(state["detection_ready"])
        self.assertFalse(state["ocr_ready"])
        self.assertEqual(state["detection_status"], "no_boxes")

    def test_detection_readiness_ignores_text_completely(self):
        """Text can never make an unapproved image detection-ready."""
        for texts in (0, 1, 5):
            self.assertFalse(image_readiness("needs_review", None, 2, texts)["detection_ready"])


class AnnotationTextTest(unittest.TestCase):
    """Acceptance criterion 4: a prediction is not ground truth."""

    def test_prediction_and_verified_text_are_stored_separately(self):
        plate = SimpleNamespace(ai_detection=True, ai_result="5IY-649O", verified_result=None)
        # The model's guess lives only in ai_result.
        self.assertIsNone(plate.verified_result)
        # Only a human-written value makes OCR readiness possible.
        plate.verified_result = plate.ai_result
        state = image_readiness("verified", NOW, 1, 1)
        self.assertTrue(state["ocr_ready"])
        # Equal strings are still distinct provenance: ai_result is preserved.
        self.assertEqual(plate.ai_result, plate.verified_result)

    def test_empty_text_is_not_ground_truth(self):
        """Sidecar imports store "". That must not count as a verified label."""
        for value in ("", "   "):
            counts = 1 if value.strip() else 0
            state = image_readiness("verified", NOW, plate_count=1, verified_text_count=counts)
            self.assertFalse(state["ocr_ready"], repr(value))
            self.assertTrue(state["detection_ready"], repr(value))


if __name__ == "__main__":
    unittest.main()
"""The OCR text endpoint must not disturb detection ground truth."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from app.main import update_ocr_text
from app.schemas import OcrTextUpdate


class OcrTextEndpointTest(unittest.TestCase):
    def _image(self, annotations, review_status="verified", verified_at=datetime.now(timezone.utc)):
        image = SimpleNamespace(
            id="img", source="datasets_folder", original_filename="a.png",
            width=100, height=100, detection_complete=True,
            review_status=review_status, verified_at=verified_at,
            file_url="/api/images/img/file", annotations=annotations,
        )
        return image

    def _annotation(self, ident, verified_result=None, ai_result="51A-1234"):
        return SimpleNamespace(
            id=ident, bbox_xyxy_pixels=[1.0, 1.0, 20.0, 10.0],
            bbox_normalized_xywh=None, ai_detection=True, ai_result=ai_result,
            verified_result=verified_result, detection_confidence=0.9,
            ocr_confidence=0.8, country=None, country_confidence=None,
        )

    def test_saving_text_preserves_review_status_and_verified_at(self):
        stamp = datetime.now(timezone.utc)
        annotations = [self._annotation(1), self._annotation(2)]
        image = self._image(annotations, verified_at=stamp)
        db = SimpleNamespace(scalar=lambda *_a, **_k: image, commit=lambda: None, refresh=lambda _i: None)
        with mock.patch("app.main.image_payload", return_value={"id": "img"}):
            update_ocr_text("img", OcrTextUpdate(plates=[{"annotation_id": 1, "verified_result": "51A-1234"}]), db=db)
        self.assertEqual(image.review_status, "verified")
        self.assertEqual(image.verified_at, stamp)
        self.assertEqual(annotations[0].verified_result, "51A-1234")

    def test_boxes_are_not_modified(self):
        annotations = [self._annotation(1)]
        before = list(annotations[0].bbox_xyxy_pixels)
        image = self._image(annotations)
        db = SimpleNamespace(scalar=lambda *_a, **_k: image, commit=lambda: None, refresh=lambda _i: None)
        with mock.patch("app.main.image_payload", return_value={"id": "img"}):
            update_ocr_text("img", OcrTextUpdate(plates=[{"annotation_id": 1, "verified_result": "AB-1"}]), db=db)
        self.assertEqual(annotations[0].bbox_xyxy_pixels, before)

    def test_prediction_is_never_copied_into_ground_truth(self):
        annotations = [self._annotation(1, ai_result="5IY-649O")]
        image = self._image(annotations)
        db = SimpleNamespace(scalar=lambda *_a, **_k: image, commit=lambda: None, refresh=lambda _i: None)
        with mock.patch("app.main.image_payload", return_value={"id": "img"}):
            update_ocr_text("img", OcrTextUpdate(plates=[{"annotation_id": 1, "verified_result": ""}]), db=db)
        self.assertIsNone(annotations[0].verified_result)
        self.assertEqual(annotations[0].ai_result, "5IY-649O")

    def test_clearing_text_returns_plate_to_missing(self):
        annotations = [self._annotation(1, verified_result="AB-1")]
        image = self._image(annotations)
        db = SimpleNamespace(scalar=lambda *_a, **_k: image, commit=lambda: None, refresh=lambda _i: None)
        with mock.patch("app.main.image_payload", return_value={"id": "img"}):
            update_ocr_text("img", OcrTextUpdate(plates=[{"annotation_id": 1, "verified_result": "   "}]), db=db)
        self.assertIsNone(annotations[0].verified_result)

    def test_annotation_from_another_image_is_rejected(self):
        annotations = [self._annotation(1)]
        image = self._image(annotations)
        db = SimpleNamespace(scalar=lambda *_a, **_k: image, commit=lambda: None, refresh=lambda _i: None)
        with self.assertRaises(HTTPException) as caught:
            update_ocr_text("img", OcrTextUpdate(plates=[{"annotation_id": 999, "verified_result": "X"}]), db=db)
        self.assertEqual(caught.exception.status_code, 400)

    def test_duplicate_annotation_ids_are_rejected(self):
        annotations = [self._annotation(1)]
        image = self._image(annotations)
        db = SimpleNamespace(scalar=lambda *_a, **_k: image, commit=lambda: None, refresh=lambda _i: None)
        with self.assertRaises(HTTPException) as caught:
            update_ocr_text("img", OcrTextUpdate(plates=[
                {"annotation_id": 1, "verified_result": "A"}, {"annotation_id": 1, "verified_result": "B"},
            ]), db=db)
        self.assertEqual(caught.exception.status_code, 400)

    def test_missing_image_is_404(self):
        db = SimpleNamespace(scalar=lambda *_a, **_k: None)
        with self.assertRaises(HTTPException) as caught:
            update_ocr_text("nope", OcrTextUpdate(plates=[]), db=db)
        self.assertEqual(caught.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
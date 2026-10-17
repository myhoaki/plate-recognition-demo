"""CRNN / PaddleOCR checkpoint compatibility.

The OCR trainer emits CRNN weights (ocr_model.PlateOCRNetwork) while recognition
runs PaddleOCR (pipeline._get_paddle_ocr). These tests pin the guarantee that a
CRNN run can never be used as an inference checkpoint, and that this never
degrades into "recognition succeeded with empty text for every plate".

Run inside the api image:
    python -m unittest discover -s /app/tests -t /app -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from app import pipeline
from app.logic import (
    OCR_INFERENCE_ENGINE,
    OCR_TRAINING_ENGINE,
    ocr_checkpoint_is_usable,
    ocr_training_engine,
)
from app.main import activate_training_model, recognize_image, start_evaluation
from app.pipeline import OcrCheckpointError, _get_paddle_ocr, get_models
from app.schemas import RecognitionRequest, StartEvaluation


def _ocr_job(config_overrides: dict | None = None) -> SimpleNamespace:
    """A completed OCR training job whose checkpoint is a CRNN .pt."""
    config = {"model_type": "ocr", "epochs": 5, "batch_size": 8}
    config.update(config_overrides or {})
    return SimpleNamespace(
        id="a" * 32,
        status="completed",
        config=config,
        model_path="/training/jobs/" + "a" * 32 + "/best.pt",
        metrics={"best_epoch": 3},
    )


def _detector_job() -> SimpleNamespace:
    return SimpleNamespace(
        id="d" * 32,
        status="completed",
        config={"model_type": "detector", "epochs": 5, "batch_size": 8},
        model_path="/training/jobs/" + "d" * 32 + "/best.pt",
        metrics={"best_epoch": 4},
    )


class StubSession:
    """Minimal duck-typed stand-in for the SQLAlchemy session.

    Only ``get`` and ``scalar`` are exercised by the endpoints under test, so a
    real database is unnecessary.
    """

    def __init__(self, rows: dict | None = None):
        self.rows = rows or {}

    def get(self, _model, key):
        return self.rows.get(key)

    def scalar(self, *_args, **_kwargs):
        return None

    def commit(self):
        return None


class OcrEngineLabellingTest(unittest.TestCase):
    def test_crnn_run_is_not_a_usable_paddleocr_checkpoint(self):
        self.assertEqual(ocr_training_engine({"model_type": "ocr"}), OCR_TRAINING_ENGINE)
        self.assertFalse(ocr_checkpoint_is_usable({"model_type": "ocr"}))

    def test_engine_is_recorded_explicitly_on_new_ocr_jobs(self):
        self.assertFalse(
            ocr_checkpoint_is_usable({"model_type": "ocr", "ocr_engine": OCR_TRAINING_ENGINE})
        )

    def test_a_hypothetical_paddleocr_trainer_would_be_accepted(self):
        self.assertTrue(
            ocr_checkpoint_is_usable({"model_type": "ocr", "ocr_engine": OCR_INFERENCE_ENGINE})
        )

    def test_detector_jobs_are_not_mislabeled_as_ocr(self):
        self.assertEqual(ocr_training_engine({"model_type": "detector"}), "")
        self.assertEqual(ocr_training_engine({}), "")

    def test_missing_config_is_treated_as_crnn_not_as_usable(self):
        self.assertFalse(ocr_checkpoint_is_usable(None))
        self.assertFalse(ocr_checkpoint_is_usable({}))


class PaddleOcrLoadingTest(unittest.TestCase):
    """Requirement 9: a bad *selection* must not be swallowed."""

    def setUp(self):
        _get_paddle_ocr.cache_clear()

    def tearDown(self):
        _get_paddle_ocr.cache_clear()

    def test_selected_checkpoint_failure_raises_instead_of_returning_none(self):
        # A CRNN .pt is not a PaddleOCR model directory, so the load fails.
        with self.assertRaises(OcrCheckpointError) as caught:
            _get_paddle_ocr("/training/jobs/x/best.pt", 0)
        self.assertIn("could not be loaded by PaddleOCR", str(caught.exception))
        self.assertIn("CRNN", str(caught.exception))

    def test_default_model_failure_still_degrades_to_no_ocr(self):
        """Unrelated failure handling is preserved: detection must keep working."""
        fake_module = SimpleNamespace(
            TextRecognition=mock.Mock(side_effect=RuntimeError("no model assets"))
        )
        with mock.patch.dict("sys.modules", {"paddleocr": fake_module}):
            self.assertIsNone(_get_paddle_ocr(None, 0))

    def test_default_model_loads_paddleocr_by_name(self):
        sentinel = object()
        fake_module = SimpleNamespace(TextRecognition=mock.Mock(return_value=sentinel))
        with mock.patch.dict("sys.modules", {"paddleocr": fake_module}):
            self.assertIs(_get_paddle_ocr(None, 0), sentinel)
        fake_module.TextRecognition.assert_called_with(model_name=pipeline.PADDLE_OCR_MODEL)


class GetModelsTest(unittest.TestCase):
    def setUp(self):
        _get_paddle_ocr.cache_clear()

    def tearDown(self):
        _get_paddle_ocr.cache_clear()

    def test_default_path_still_uses_bundled_paddleocr(self):
        sentinel = object()
        with mock.patch.object(pipeline, "_get_paddle_ocr", return_value=sentinel) as loader:
            _detector, ocr = get_models()
        self.assertIs(ocr, sentinel)
        loader.assert_called_once_with(None, 0)

    def test_active_ocr_file_is_never_auto_loaded(self):
        """Requirement 10: never resolve active_ocr to a CRNN checkpoint.

        Honouring such a file would hand CRNN weights to PaddleOCR, which fails
        to load and would silently disable OCR on every later request.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("active_ocr", "active_ocr.pt"):
                (root / name).write_bytes(b"crnn-weights")
            with mock.patch.dict("os.environ", {"TRAINING_DIR": str(root)}):
                with mock.patch.object(pipeline, "_get_paddle_ocr", return_value=object()) as loader:
                    get_models()
            loader.assert_called_once_with(None, 0)

    def test_explicit_ocr_checkpoint_is_passed_through(self):
        with mock.patch.object(pipeline, "_get_paddle_ocr", return_value=object()) as loader:
            get_models(None, "/some/paddle/model_dir")
        loader.assert_called_once_with("/some/paddle/model_dir", 0)


class RecognitionApiTest(unittest.TestCase):
    """Requirement 3: a CRNN OCR job id must produce a clear API error."""

    IMAGE = SimpleNamespace(
        id="img", source="upload", original_filename="x.png",
        storage_key="uploads/x.png", width=10, height=10,
    )

    def _training_dir_with_checkpoint(self, job: SimpleNamespace) -> str:
        """Create <TRAINING_DIR>/jobs/<job id>/best.pt and point the job at it."""
        directory = tempfile.mkdtemp()
        jobs = Path(directory) / "jobs" / job.id
        jobs.mkdir(parents=True)
        checkpoint = jobs / "best.pt"
        checkpoint.write_bytes(b"weights")
        job.model_path = str(checkpoint)
        return directory

    def test_crnn_ocr_job_id_is_rejected(self):
        job = _ocr_job()
        db = StubSession({job.id: job, self.IMAGE.id: self.IMAGE})
        with mock.patch.dict("os.environ", {"TRAINING_DIR": self._training_dir_with_checkpoint(job)}):
            with mock.patch("app.main.run_detection") as run:
                with self.assertRaises(HTTPException) as caught:
                    recognize_image(RecognitionRequest(image_id="img", ocr_job_id=job.id), db=db)
        run.assert_not_called()  # rejected before any inference
        self.assertEqual(caught.exception.status_code, 400)
        detail = caught.exception.detail
        self.assertIn("CRNN", detail)
        self.assertIn("PADDLEOCR", detail.upper())
        self.assertIn("cannot be interchanged", detail)

    def test_crnn_ocr_job_with_explicit_engine_key_is_rejected(self):
        job = _ocr_job({"ocr_engine": OCR_TRAINING_ENGINE})
        db = StubSession({job.id: job, self.IMAGE.id: self.IMAGE})
        with mock.patch.dict("os.environ", {"TRAINING_DIR": self._training_dir_with_checkpoint(job)}):
            with mock.patch("app.main.run_detection"):
                with self.assertRaises(HTTPException) as caught:
                    recognize_image(RecognitionRequest(image_id="img", ocr_job_id=job.id), db=db)
        self.assertEqual(caught.exception.status_code, 400)

    def test_detector_job_selection_is_still_accepted(self):
        """Requirement 5: the detector path must keep working."""
        import cv2
        import numpy as np

        job = _detector_job()
        db = StubSession({job.id: job, self.IMAGE.id: self.IMAGE})
        detections = [{
            "bbox_xyxy_pixels": [1.0, 1.0, 5.0, 5.0], "text": "AB-1234",
            "ocr_confidence": 0.9, "detection_confidence": 0.8,
            "country": None, "country_confidence": None,
        }]
        directory = tempfile.mkdtemp()
        image_file = Path(directory) / "vehicle.png"
        cv2.imwrite(str(image_file), np.full((10, 10, 3), 128, np.uint8))
        with mock.patch.dict("os.environ", {"TRAINING_DIR": self._training_dir_with_checkpoint(job)}):
            with mock.patch("app.main.image_path", return_value=image_file):
                with mock.patch("app.main.run_detection", return_value=detections) as run:
                    result = recognize_image(
                        RecognitionRequest(image_id="img", detector_job_id=job.id), db=db
                    )
        run.assert_called_once()
        self.assertEqual(run.call_args.args[1], job.model_path)
        self.assertIsNone(run.call_args.args[2])  # no OCR checkpoint requested
        self.assertEqual(result["detections"][0]["text"], "AB-1234")
        self.assertEqual(result["detections"][0]["ocr_confidence"], 0.9)
        self.assertTrue(result["detections"][0]["crop_data_url"].startswith("data:image/jpeg;base64,"))

    def test_ocr_init_error_inside_pipeline_surfaces_as_400(self):
        """A PaddleOCR-compatible selection that fails to load is reported."""
        job = SimpleNamespace(
            id="b" * 32, status="completed",
            config={"model_type": "ocr", "ocr_engine": OCR_INFERENCE_ENGINE},
            model_path="unused", metrics={},
        )
        directory = self._training_dir_with_checkpoint(job)
        db = StubSession({job.id: job, self.IMAGE.id: self.IMAGE})
        with mock.patch.dict("os.environ", {"TRAINING_DIR": directory}):
            with mock.patch("app.main.image_path", return_value=Path("/tmp/x.png")):
                with mock.patch("app.main.run_detection", side_effect=OcrCheckpointError("bad model dir")):
                    with self.assertRaises(HTTPException) as caught:
                        recognize_image(RecognitionRequest(image_id="img", ocr_job_id=job.id), db=db)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("bad model dir", caught.exception.detail)


class EvaluationApiTest(unittest.TestCase):
    def test_crnn_ocr_job_id_is_rejected_at_submission(self):
        job = _ocr_job()
        db = StubSession({job.id: job})
        snapshot = SimpleNamespace(id="s" * 32, task="detector")
        original_get = db.get

        def get(model, key):
            if model.__name__ == "DatasetSnapshot":
                return snapshot
            return original_get(model, key)

        db.get = get
        with mock.patch("app.main.run_evaluation_job") as task:
            with self.assertRaises(HTTPException) as caught:
                start_evaluation(
                    StartEvaluation(dataset_snapshot_id=snapshot.id, ocr_job_id=job.id), db=db
                )
        task.delay.assert_not_called()
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("CRNN", caught.exception.detail)


class ActivationTest(unittest.TestCase):
    """Requirement 10: do not fix the active_ocr.pt / active_ocr name mismatch."""

    def test_activating_a_crnn_model_is_refused(self):
        job = _ocr_job()
        db = StubSession({job.id: job})
        with self.assertRaises(HTTPException) as caught:
            activate_training_model(job.id, db=db)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("CRNN", caught.exception.detail)
        self.assertIn("not activated", caught.exception.detail)


if __name__ == "__main__":
    unittest.main()
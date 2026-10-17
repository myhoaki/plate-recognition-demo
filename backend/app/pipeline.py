"""The license-plate recognition pipeline.

Stage A: YOLO26 plate detection -> Stage B: OpenCV plate crop -> Stage C: PaddleOCR
recognition-only.

Detector and recognizer are independent components with separate configuration,
checkpoints, and evaluation.
"""

from __future__ import annotations

from functools import lru_cache

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
from ultralytics import YOLO


class BoundingBox:
    """Pixel bounding box [x1, y1, x2, y2] produced by the YOLO26 detector."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class OcrResult:
    """OCR recognition output for a single plate crop."""

    def __init__(self, text: str, confidence: float):
        self.text = text
        self.confidence = confidence


class DetectionResult:
    """YOLO26 detection output for a single plate."""

    def __init__(self, label: str, confidence: float, bounding_box: BoundingBox):
        self.label = label
        self.confidence = confidence
        self.bounding_box = bounding_box


class TrainedYoloDetector:
    """Ultralytics YOLO26 detector wrapper for license-plate localization."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, frame: np.ndarray) -> list[DetectionResult]:
        runs_dir = configure_ultralytics_paths()
        result = self.model.predict(
            frame,
            verbose=False,
            save=False,
            project=str(runs_dir),
            name="inference",
            exist_ok=True,
        )[0]
        output: list[DetectionResult] = []
        if result.boxes is None:
            return output
        for box in result.boxes:
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            output.append(
                DetectionResult(
                    label="license_plate",
                    confidence=float(box.conf[0]),
                    bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return output


_ultralytics_root: Path | None = None

# PaddleOCR recognition-only model. PP-OCRv6_tiny_rec is the smallest PP-OCRv6
# tier and is the fastest option on CPU. It covers Latin script, which covers
# Vietnamese license plates.
PADDLE_OCR_MODEL = "PP-OCRv6_tiny_rec"

if TYPE_CHECKING:  # pragma: no cover - typing only
    from paddleocr import TextRecognition


def configure_ultralytics_paths(training_dir: Path | None = None) -> Path:
    """Keep Ultralytics config and implicit run outputs under writable storage."""
    global _ultralytics_root
    training_dir = training_dir or Path(os.environ.get("TRAINING_DIR", "/training"))
    root = training_dir / "ultralytics"
    config_dir = root / "config"
    runs_dir = root / "runs"
    config_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    if _ultralytics_root != root:
        from ultralytics import settings

        settings.update({"runs_dir": str(runs_dir)})
        _ultralytics_root = root
    return runs_dir


class OcrCheckpointError(RuntimeError):
    """A selected OCR checkpoint cannot be used by the PaddleOCR inference path.

    Raised only for an explicitly selected checkpoint. A failure to load the
    bundled default PaddleOCR model still degrades to "no OCR" so that detection
    keeps working.
    """


@lru_cache(maxsize=2)
def _get_paddle_ocr(
    model_dir: str | None, model_mtime_ns: int
) -> "TextRecognition | None":
    """PaddleOCR recognition-only recognizer.

    Uses PaddleOCR's TextRecognition pipeline, which runs *only* the recognizer.
    That is what this project needs: YOLO26 already supplies the plate box, so
    PaddleOCR's text-detection and orientation models are redundant work.

    The import is deferred so an unavailable or misconfigured PaddleOCR install
    cannot stop the API from starting; detection still works without OCR. That
    leniency applies to the *bundled default* only: when a caller explicitly
    selects a checkpoint, a load failure is raised as OcrCheckpointError so an
    incompatible selection (for example a CRNN .pt) is reported instead of
    silently returning empty text for every plate.
    """
    try:
        from paddleocr import TextRecognition  # noqa: PLC0415 - lazy import
    except Exception as exc:
        if model_dir:
            raise OcrCheckpointError(
                f"PaddleOCR is not available to load the selected OCR checkpoint "
                f"({model_dir}): {exc}"
            ) from exc
        return None
    try:
        if model_dir:
            return TextRecognition(model_dir=model_dir)
        return TextRecognition(model_name=PADDLE_OCR_MODEL)
    except Exception as exc:
        if model_dir:
            raise OcrCheckpointError(
                f"The selected OCR checkpoint could not be loaded by PaddleOCR "
                f"({model_dir}). PaddleOCR expects one of its own model directories; a "
                f"CRNN .pt checkpoint is not compatible. Clear the OCR checkpoint "
                f"selection to use the bundled PP-OCRv6-tiny model. Cause: {exc}"
            ) from exc
        return None


def get_models(
    detector_checkpoint: str | None = None,
    ocr_checkpoint: str | None = None,
) -> tuple[TrainedYoloDetector | None, "TextRecognition | None"]:
    """Return (detector, ocr) instances.

    Prefers the active detector checkpoint (active_detector.pt) when present,
    otherwise uses the provided checkpoint path. If no detector checkpoint is
    available, detector is None and callers must handle that (YOLO26 pretrained
    on COCO does not detect license plates — training is required).

    No active *OCR* checkpoint is consulted. The only OCR trainer in this
    project emits CRNN weights, which PaddleOCR cannot load; honouring such a
    file here would silently disable recognition instead of using PaddleOCR.
    OCR inference therefore always uses the bundled PaddleOCR model unless an
    explicit, PaddleOCR-compatible checkpoint is supplied.
    """
    training_dir = Path(os.environ.get("TRAINING_DIR", "/training"))
    active_detector_path = training_dir / "active_detector.pt"

    detector_path = detector_checkpoint or (
        str(active_detector_path) if active_detector_path.is_file() else None
    )

    detector = TrainedYoloDetector(detector_path) if detector_path else None
    ocr = _get_paddle_ocr(ocr_checkpoint, 0) if ocr_checkpoint else _get_paddle_ocr(None, 0)
    return detector, ocr


def confidence_value(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, list):
        return sum(value) / len(value) if value else None
    return float(value)


def crop_plate(frame: np.ndarray, bbox_xyxy: list[float]) -> np.ndarray | None:
    """Stage B: crop the plate from the full image using the detector box.

    Pure image-processing (NumPy slice) — no AI model involved.
    Coordinates are clipped to image boundaries; invalid/empty crops return None.
    The original image is not modified.
    """
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def detect_plates(
    frame: np.ndarray, detector: TrainedYoloDetector | None
) -> list[dict]:
    """Stage A: run YOLO26 plate detection on a full vehicle image.

    Returns a list of detections with pixel bounding boxes and detection
    confidence. No OCR is performed here.
    """
    if detector is None:
        return []
    results = detector.predict(frame)
    detections = []
    for result in results:
        box = result.bounding_box
        detections.append(
            {
                "bbox_xyxy_pixels": [
                    float(box.x1),
                    float(box.y1),
                    float(box.x2),
                    float(box.y2),
                ],
                "detection_confidence": float(result.confidence),
            }
        )
    return detections


def recognize_plate_text(
    crop: np.ndarray | None, ocr: "TextRecognition | None"
) -> OcrResult | None:
    """Stage C: run PaddleOCR recognition-only on a cropped plate image.

    TextRecognition.predict() returns a list of dicts with 'rec_text' and
    'rec_score'. Returns OcrResult(text, confidence), or None when OCR is
    unavailable or the crop is empty.
    """
    if crop is None or crop.size == 0 or ocr is None:
        return None
    try:
        results = ocr.predict(crop)
    except Exception:
        return None
    if not results:
        return OcrResult(text="", confidence=0.0)
    first = results[0] if isinstance(results, list) else results
    if not isinstance(first, dict):
        return OcrResult(text="", confidence=0.0)
    text = first.get("rec_text") or first.get("rec_texts") or ""
    score = first.get("rec_score")
    if score is None:
        scores = first.get("rec_scores")
        score = scores[0] if isinstance(scores, (list, tuple)) and scores else 0.0
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    return OcrResult(text=str(text).strip(), confidence=float(score))


def run_detection(
    path: Path,
    detector_checkpoint: str | None = None,
    ocr_checkpoint: str | None = None,
) -> list[dict]:
    """Run the full Stage A -> B -> C pipeline on one image file.

    Backward-compatible return shape: list of dicts with
    bbox_xyxy_pixels, text, ocr_confidence, detection_confidence, country,
    country_confidence. country/country_confidence are not produced by
    PaddleOCR recognition-only and are returned as None.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Image could not be decoded")

    detector, ocr = get_models(detector_checkpoint, ocr_checkpoint)
    if detector is None:
        raise ValueError(
            "No detector model available. Train a YOLO26 detector first "
            "(Train Detection section), then activate it, or select a "
            "completed detector checkpoint in Recognition / Test."
        )
    detections = detect_plates(image, detector)

    output = []
    for detection in detections:
        bbox = detection["bbox_xyxy_pixels"]
        crop = crop_plate(image, bbox)
        ocr_result = recognize_plate_text(crop, ocr)
        output.append(
            {
                "bbox_xyxy_pixels": bbox,
                "text": ocr_result.text if ocr_result else "",
                "ocr_confidence": ocr_result.confidence if ocr_result else None,
                "detection_confidence": detection["detection_confidence"],
                "country": None,
                "country_confidence": None,
            }
        )
    return output

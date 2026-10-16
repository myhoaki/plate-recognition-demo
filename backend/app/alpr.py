"""Stage A: YOLO26 plate detection + Stage B: OpenCV plate crop + Stage C: PaddleOCR recognition.

This module replaces the previous fast_alpr.ALPR bundled pipeline. Detection and
recognition are now explicitly separated components with independent configuration.

Pipeline:
    full image -> YOLO26 detection (pixel boxes) -> OpenCV crop -> PaddleOCR
    recognition-only (text + confidence) -> final result.
"""

from __future__ import annotations

from functools import lru_cache

import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR


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


@lru_cache(maxsize=2)
def _get_paddle_ocr(
    rec_model_dir: str | None, model_mtime_ns: int
) -> PaddleOCR:
    """PaddleOCR recognition-only instance.

    use_det=False  - skip text detection (we already have YOLO26 boxes)
    use_cls=False  - skip orientation classification
    lang="en"      - Latin-script recognition, covers Vietnamese plates
    rec_model_dir  - optional path to a fine-tuned PaddleOCR recognition model
    """
    return PaddleOCR(
        use_det=False,
        use_cls=False,
        rec_model_dir=rec_model_dir,
        lang="en",
        show_log=False,
    )


def get_alpr(
    detector_checkpoint: str | None = None,
    ocr_checkpoint: str | None = None,
) -> tuple[TrainedYoloDetector | None, PaddleOCR | None]:
    """Return (detector, ocr) instances.

    Prefers active checkpoints (active_detector.pt / active_ocr) when present,
    otherwise uses the provided checkpoint paths. If no detector checkpoint is
    available, detector is None and callers must handle that (YOLO26 pretrained
    on COCO does not detect license plates — training is required).
    """
    training_dir = Path(os.environ.get("TRAINING_DIR", "/training"))
    active_detector_path = training_dir / "active_detector.pt"
    active_ocr_path = training_dir / "active_ocr"

    detector_path = detector_checkpoint or (
        str(active_detector_path) if active_detector_path.is_file() else None
    )
    ocr_path = ocr_checkpoint or (str(active_ocr_path) if active_ocr_path.is_file() else None)

    detector = TrainedYoloDetector(detector_path) if detector_path else None
    ocr = _get_paddle_ocr(ocr_path, 0) if ocr_path else _get_paddle_ocr(None, 0)
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
    crop: np.ndarray | None, ocr: PaddleOCR | None
) -> OcrResult | None:
    """Stage C: run PaddleOCR recognition-only on a cropped plate image.

    Returns OcrResult(text, confidence) or None if the crop is empty/unreadable.
    """
    if crop is None or crop.size == 0 or ocr is None:
        return None
    try:
        result = ocr.ocr(crop, det=False, cls=False)
    except Exception:
        return None
    if not result:
        return OcrResult(text="", confidence=0.0)
    # paddleocr v3 recognition-only returns [[text, score], ...] or [None]
    first = result[0] if isinstance(result, list) else result
    if not first or first is None:
        return OcrResult(text="", confidence=0.0)
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        text = str(first[0])
        score = float(first[1]) if first[1] is not None else 0.0
        return OcrResult(text=text, confidence=score)
    return OcrResult(text="", confidence=0.0)


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

    detector, ocr = get_alpr(detector_checkpoint, ocr_checkpoint)
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

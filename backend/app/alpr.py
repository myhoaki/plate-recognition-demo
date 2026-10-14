from functools import lru_cache
import os
from pathlib import Path

import cv2
import numpy as np
from fast_alpr import ALPR
from open_image_models.detection.core.base import BoundingBox, DetectionResult


class TrainedYoloDetector:
    def __init__(self, model_path: str):
        from ultralytics import YOLO

        self.model = YOLO(model_path)

    def predict(self, frame: np.ndarray) -> list[DetectionResult]:
        result = self.model.predict(frame, verbose=False)[0]
        if result.boxes is None:
            return []
        output = []
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


@lru_cache(maxsize=2)
def _get_alpr(model_path: str | None, model_mtime_ns: int) -> ALPR:
    if model_path:
        detector = TrainedYoloDetector(model_path)
        return ALPR(
            detector=detector,
            ocr_model="cct-xs-v2-global-model",
            ocr_device="cpu",
        )
    return ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
        ocr_device="cpu",
    )


def get_alpr() -> ALPR:
    active_path = Path(os.environ.get("TRAINING_DIR", "/training")) / "active.pt"
    if active_path.is_file():
        return _get_alpr(str(active_path), active_path.stat().st_mtime_ns)
    return _get_alpr(None, 0)


def confidence_value(value):
    if value is None:
        return None
    if isinstance(value, list):
        return sum(value) / len(value) if value else None
    return float(value)


def run_detection(path: Path) -> list[dict]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Image could not be decoded")
    results = get_alpr().predict(image)
    detections = []
    for result in results:
        box = result.detection.bounding_box
        ocr = result.ocr
        detections.append(
            {
                "bbox_xyxy_pixels": [float(box.x1), float(box.y1), float(box.x2), float(box.y2)],
                "text": ocr.text.strip() if ocr and ocr.text else "",
                "ocr_confidence": confidence_value(ocr.confidence) if ocr else None,
                "detection_confidence": float(result.detection.confidence),
                "country": getattr(ocr, "region", None) if ocr else None,
                "country_confidence": confidence_value(getattr(ocr, "region_confidence", None)) if ocr else None,
            }
        )
    return detections

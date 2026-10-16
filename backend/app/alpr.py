from functools import lru_cache
import os
from pathlib import Path

import cv2
import numpy as np
from fast_alpr import ALPR
from open_image_models.detection.core.base import BoundingBox, DetectionResult


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


class TrainedYoloDetector:
    def __init__(self, model_path: str):
        configure_ultralytics_paths()
        from ultralytics import YOLO

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
def _get_alpr(model_path: str | None, model_mtime_ns: int, ocr_path: str | None, ocr_mtime_ns: int) -> ALPR:
    ocr = None
    if ocr_path:
        from .ocr_model import TrainedPlateOCR

        ocr = TrainedPlateOCR(ocr_path)
    if model_path:
        detector = TrainedYoloDetector(model_path)
        return ALPR(
            detector=detector,
            ocr=ocr,
            ocr_model=None if ocr else "cct-xs-v2-global-model",
            ocr_device="cpu",
        )
    return ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr=ocr,
        ocr_model=None if ocr else "cct-xs-v2-global-model",
        ocr_device="cpu",
    )


def get_alpr(detector_checkpoint: str | None = None, ocr_checkpoint: str | None = None) -> ALPR:
    training_dir = Path(os.environ.get("TRAINING_DIR", "/training"))
    active_path = training_dir / "active.pt"
    active_ocr_path = training_dir / "active_ocr.pt"
    detector_path = detector_checkpoint or (str(active_path) if active_path.is_file() else None)
    ocr_path = ocr_checkpoint or (str(active_ocr_path) if active_ocr_path.is_file() else None)
    if detector_path:
        configure_ultralytics_paths(training_dir)
    detector_mtime = Path(detector_path).stat().st_mtime_ns if detector_path and Path(detector_path).is_file() else 0
    ocr_mtime = Path(ocr_path).stat().st_mtime_ns if ocr_path and Path(ocr_path).is_file() else 0
    return _get_alpr(detector_path, detector_mtime, ocr_path, ocr_mtime)


def confidence_value(value):
    if value is None:
        return None
    if isinstance(value, list):
        return sum(value) / len(value) if value else None
    return float(value)


def run_detection(path: Path, detector_checkpoint: str | None = None, ocr_checkpoint: str | None = None) -> list[dict]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Image could not be decoded")
    results = get_alpr(detector_checkpoint, ocr_checkpoint).predict(image)
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

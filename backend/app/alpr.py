from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fast_alpr import ALPR


@lru_cache(maxsize=1)
def get_alpr() -> ALPR:
    return ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
        ocr_device="cpu",
    )


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

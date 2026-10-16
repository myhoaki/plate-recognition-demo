"""Run YOLO26 plate detection + PaddleOCR recognition on one still image (CPU-only).

FastALPR has been removed. This script uses:
  Stage A: YOLO26 detector (ultralytics) for plate localization
  Stage B: OpenCV/NumPy crop from the detector bounding box
  Stage C: PaddleOCR recognition-only on the cropped plate
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from backend.app.alpr import (  # noqa: E402
    crop_plate,
    detect_plates,
    get_alpr,
    recognize_plate_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect license plates in one image and read their text."
    )
    parser.add_argument("image", type=Path, help="Path to a still image")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Annotated output image (default: <input>_alpr.png)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Could not open image: {args.image}")

    output_path = args.output or args.image.with_name(
        f"{args.image.stem}_yolo26_paddleocr.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector, ocr = get_alpr()
    detections = detect_plates(image, detector)

    # Stage B + C: crop each plate and run PaddleOCR recognition-only.
    for detection in detections:
        crop = crop_plate(image, detection["bbox_xyxy_pixels"])
        ocr_result = recognize_plate_text(crop, ocr)
        detection["text"] = ocr_result.text if ocr_result else ""
        detection["ocr_confidence"] = (
            ocr_result.confidence if ocr_result else None
        )

    # Draw boxes and labels on the output image.
    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(v) for v in detection["bbox_xyxy_pixels"])
        text = detection["text"] or "plate"
        dc = detection["detection_confidence"]
        oc = detection["ocr_confidence"]
        label = f"{text} ({dc:.2f})" + (f" / OCR {oc:.2f}" if oc is not None else "")
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    if not cv2.imwrite(str(output_path), annotated):
        raise SystemExit(f"Could not write output image: {output_path}")

    print(f"Found {len(detections)} plate(s).")
    for index, detection in enumerate(detections, start=1):
        box = detection["bbox_xyxy_pixels"]
        print(
            f"Plate {index}: box={box} "
            f"det_conf={detection['detection_confidence']:.3f} "
            f"text='{detection['text']}' "
            f"ocr_conf={detection['ocr_confidence']}"
        )
    print(f"Annotated image saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

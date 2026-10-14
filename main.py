"""Run a pretrained license-plate detector and OCR model on one still image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from fast_alpr import ALPR


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
        f"{args.image.stem}_alpr.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The detector finds plate boxes; OCR reads the text inside each box.
    # Model weights are downloaded automatically the first time this runs.
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
    )
    drawn = alpr.draw_predictions(image)

    if not cv2.imwrite(str(output_path), drawn.image):
        raise SystemExit(f"Could not write output image: {output_path}")

    print(f"Found {len(drawn.results)} plate(s).")
    for index, result in enumerate(drawn.results, start=1):
        print(f"Plate {index}: {result}")
    print(f"Annotated image saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

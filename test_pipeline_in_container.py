"""Run this INSIDE the built Docker container to test the full YOLO26 + PaddleOCR pipeline.

Prerequisites (inside container):
    pip install paddlepaddle paddleocr ultralytics opencv-python-headless fastapi

Usage (from host, after docker compose build):
    docker compose exec api python /app/test_pipeline_in_container.py /path/to/test_image.jpg

Or if you have a synthetic test image:
    python3 -c "import cv2, numpy as np; cv2.imwrite('/tmp/test_car.jpg', np.zeros((480,640,3),dtype=np.uint8))"
    docker cp /tmp/test_car.jpg platelens-api-1:/tmp/test_car.jpg
    docker compose exec api python /app/test_pipeline_in_container.py /tmp/test_car.jpg
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, "/app")
from backend.app.pipeline import get_models, detect_plates, recognize_plate_text, crop_plate


def make_synthetic_plate_image(path):
    """Create a synthetic image with a white rectangle (simulated plate) on dark background."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a white rectangle where a plate would be
    cv2.rectangle(img, (200, 200, 400, 260), (255, 255, 255), -1)
    cv2.imwrite(str(path), img)
    return path


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if image_path is None or not Path(image_path).is_file():
        print("No valid image path given. Creating synthetic test image at /tmp/synthetic_car.jpg")
        image_path = make_synthetic_plate_image(Path("/tmp/synthetic_car.jpg"))

    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    assert image is not None, f"Could not read {image_path}"
    print(f"Image shape: {image.shape}")

    print("\n--- Stage A: YOLO26 detection ---")
    detector, ocr = get_models()
    print(f"Detector: {detector}")
    print(f"OCR: {ocr}")

    detections = detect_plates(image, detector)
    print(f"Detections found: {len(detections)}")
    for i, d in enumerate(detections):
        print(f"  Plate {i+1}: box={d['bbox_xyxy_pixels']}, det_conf={d['detection_confidence']:.3f}")

    print("\n--- Stage B: OpenCV crop ---")
    for i, detection in enumerate(detections):
        crop = crop_plate(image, detection["bbox_xyxy_pixels"])
        print(f"  Crop {i+1}: {crop.shape if crop is not None else None}")

    print("\n--- Stage C: PaddleOCR recognition-only ---")
    for i, detection in enumerate(detections):
        crop = crop_plate(image, detection["bbox_xyxy_pixels"])
        result = recognize_plate_text(crop, ocr)
        print(f"  Plate {i+1}: text='{result.text if result else None}' "
              f"conf={result.confidence if result else None}")

    print("\nPipeline test complete.")


if __name__ == "__main__":
    main()

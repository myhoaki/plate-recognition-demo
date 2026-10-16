"""Test Stage B plate-crop logic using only OpenCV (no ultralytics/paddleocr needed).

Run: python3 test_crop_logic.py
"""
import cv2
import numpy as np


def crop_plate(frame, bbox_xyxy):
    """Copy of backend/app/alpr.py:crop_plate for isolated testing."""
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def test_normal_crop():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    crop = crop_plate(frame, [420, 540, 560, 620])
    assert crop is not None
    assert crop.shape == (80, 140, 3), crop.shape
    print("PASS test_normal_crop:", crop.shape)


def test_box_outside_image_clipped():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # box extends beyond right/bottom edges
    crop = crop_plate(frame, [80, 80, 200, 200])
    assert crop is not None
    assert crop.shape == (20, 20, 3), crop.shape
    print("PASS test_box_outside_image_clipped:", crop.shape)


def test_box_fully_outside_returns_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_plate(frame, [200, 200, 300, 300])
    assert crop is None
    print("PASS test_box_fully_outside_returns_none")


def test_zero_area_box_returns_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_plate(frame, [50, 50, 50, 60]) is None
    assert crop_plate(frame, [50, 50, 60, 50]) is None
    print("PASS test_zero_area_box_returns_none")


def test_negative_coordinates_clipped_to_zero():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_plate(frame, [-50, -50, 50, 50])
    assert crop is not None
    assert crop.shape == (50, 50, 3), crop.shape
    print("PASS test_negative_coordinates_clipped_to_zero")


def test_original_image_not_modified():
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
    original = frame.copy()
    crop_plate(frame, [10, 10, 90, 90])
    assert np.array_equal(frame, original), "original image was modified!"
    print("PASS test_original_image_not_modified")


if __name__ == "__main__":
    test_normal_crop()
    test_box_outside_image_clipped()
    test_box_fully_outside_returns_none()
    test_zero_area_box_returns_none()
    test_negative_coordinates_clipped_to_zero()
    test_original_image_not_modified()
    print("\nAll crop logic tests PASSED.")

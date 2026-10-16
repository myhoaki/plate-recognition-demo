"""Assisted still-image license plate labeling web application."""

from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import statistics
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file, send_from_directory

# FastALPR has been removed. Detection and recognition now live in backend/app/alpr.py
# (YOLO26 detector + OpenCV crop + PaddleOCR recognizer).
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from backend.app.alpr import (  # noqa: E402
    crop_plate,
    detect_plates,
    get_alpr as _backend_get_alpr,
    recognize_plate_text,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_PIXELS = 40_000_000
LABELS_DIR = Path(os.environ.get("LABELS_DIR", Path(__file__).parent / "docker" / "labels"))
LEGACY_LABELS_DIR = Path(os.environ.get("LEGACY_LABELS_DIR", "/legacy-labels"))
RAW_IMAGES_DIR = Path(os.environ.get("RAW_IMAGES_DIR", Path(__file__).parent / "raw-dataset-detector"))
inference_lock = Lock()
label_save_lock = RLock()
migration_lock = Lock()
legacy_labels_migrated = False


@lru_cache(maxsize=1)
def get_alpr():
    """Load YOLO26 detector and PaddleOCR recognizer once and reuse across requests."""
    return _backend_get_alpr()


def confidence_value(value: float | list[float] | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, list):
        return statistics.mean(value) if value else None
    return float(value)


def available_raw_images() -> list[str]:
    if not RAW_IMAGES_DIR.is_dir():
        return []
    return sorted(
        (
            path.name
            for path in RAW_IMAGES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=str.casefold,
    )


def find_label_record(source_type: str, filename: str, image_sha256: str | None = None) -> dict | None:
    if not LABELS_DIR.is_dir():
        return None
    paths = sorted(LABELS_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            image_info = record.get("image", {})
            same_source = (
                image_info.get("source") == source_type
                and image_info.get("original_filename") == filename
            )
            matches = (
                same_source
                if image_sha256 is None
                else image_info.get("sha256") == image_sha256 or (not image_info.get("sha256") and same_source)
            )
            if matches:
                return record
        except (OSError, AttributeError, TypeError, json.JSONDecodeError):
            continue
    return None


def box_iou(first: list[float], second: list[float]) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1.0, first_area + second_area - intersection)


def persist_detection(
    source_type: str,
    filename: str,
    image_bytes: bytes,
    image: np.ndarray,
    detections: list[dict],
) -> str:
    """Persist YOLO26 detections + PaddleOCR results as a label record.

    `detections` is a list of dicts from detect_plates()/recognize_plate_text():
      {bbox_xyxy_pixels, text, detection_confidence, ocr_confidence, ...}
    """
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    with label_save_lock:
        previous = find_label_record(source_type, filename, image_sha256)
        record_id = previous.get("id", "") if previous else ""
        if not isinstance(record_id, str) or len(record_id) != 32:
            record_id = uuid4().hex
        previous_image = previous.get("image", {}) if previous else {}
        image_filename = previous_image.get("file", f"{record_id}.jpg")
        if not isinstance(image_filename, str) or Path(image_filename).name != image_filename:
            image_filename = f"{record_id}.jpg"
        previous_objects = previous.get("objects", []) if previous else []
        if not isinstance(previous_objects, list):
            previous_objects = []
        matched_previous: set[int] = set()
        height, width = image.shape[:2]
        objects = []
        detections_out = []
        for result in detections:
            bounds = [float(value) for value in result["bbox_xyxy_pixels"]]
            ai_text = str(result.get("text") or "").strip()
            confidence = confidence_value(result.get("ocr_confidence"))
            country = result.get("country")
            country_confidence = confidence_value(result.get("country_confidence"))
            detection_confidence = float(result.get("detection_confidence", 0.0))
            normalized = [
                round(((bounds[0] + bounds[2]) / 2) / width, 6),
                round(((bounds[1] + bounds[3]) / 2) / height, 6),
                round((bounds[2] - bounds[0]) / width, 6),
                round((bounds[3] - bounds[1]) / height, 6),
            ]
            best_index = None
            best_overlap = 0.0
            for index, old_object in enumerate(previous_objects):
                old_bounds = old_object.get("bbox_xyxy_pixels") if isinstance(old_object, dict) else None
                if index in matched_previous or not isinstance(old_bounds, list) or len(old_bounds) != 4:
                    continue
                try:
                    overlap = box_iou(bounds, [float(value) for value in old_bounds])
                except (TypeError, ValueError):
                    continue
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_index = index
            old_object = previous_objects[best_index] if best_index is not None and best_overlap >= 0.15 else {}
            if old_object:
                matched_previous.add(best_index)
            old_ocr = old_object.get("ocr", {}) if isinstance(old_object, dict) else {}
            verified = old_ocr.get("verified_result") if isinstance(old_ocr, dict) else None
            if not verified and isinstance(old_object, dict):
                verified = old_object.get("ground_truth")
            verified = verified.strip() if isinstance(verified, str) and verified.strip() else None
            annotation = {
                "class": "license_plate",
                "ai_detection": True,
                "ground_truth": verified or "",
                "ocr": {"ai_result": ai_text or None, "verified_result": verified},
                "ocr_confidence": confidence,
                "detection_confidence": detection_confidence,
                "country": country,
                "country_confidence": country_confidence,
                "bbox_xyxy_pixels": [round(value, 2) for value in bounds],
                "bbox_normalized_xywh": normalized,
            }
            objects.append(annotation)
            detections_out.append(
                {
                    "bbox_xyxy_pixels": bounds,
                    "text": ai_text,
                    "ocr_confidence": confidence,
                    "detection_confidence": detection_confidence,
                    "country": country,
                    "country_confidence": country_confidence,
                }
            )

        # Keep prior AI boxes too: a rerun may miss a plate that an earlier
        # inference found. Preserve verified manual boxes and all unmatched AI
        # suggestions so a transient/partial result cannot erase known boxes.
        for index, old_object in enumerate(previous_objects):
            if index in matched_previous or not isinstance(old_object, dict):
                continue
            old_ocr = old_object.get("ocr", {})
            verified = old_ocr.get("verified_result") if isinstance(old_ocr, dict) else None
            if not verified:
                verified = old_object.get("ground_truth")
            if old_object.get("ai_detection") or (isinstance(verified, str) and verified.strip()):
                preserved_object = dict(old_object)
                preserved_ocr = preserved_object.get("ocr", {})
                if not preserved_object.get("ai_detection"):
                    preserved_object["ai_detection"] = bool(
                        isinstance(preserved_ocr, dict) and preserved_ocr.get("ai_result")
                    )
                objects.append(preserved_object)

        now = datetime.now(timezone.utc).isoformat()
        record = {
            "schema": "platelens-label-v2",
            "id": record_id,
            "created_at": previous.get("created_at", now) if previous else now,
            "updated_at": now,
            "ai_detection_complete": True,
            "models": {"detector": "yolo26n", "ocr": "paddleocr-ppocrv6-tiny"},
            "image": {
                "file": image_filename,
                "original_filename": filename,
                "source": source_type,
                "sha256": image_sha256,
                "width": width,
                "height": height,
            },
            "objects": objects,
        }
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise OSError("Could not encode the source image")
        image_path = LABELS_DIR / image_filename
        json_path = LABELS_DIR / f"{record_id}.json"
        temporary_path = LABELS_DIR / f".{record_id}.tmp"
        image_path.write_bytes(encoded.tobytes())
        try:
            temporary_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary_path.replace(json_path)
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise
    return record_id


@app.get("/")
def home():
    return redirect("/label")


def recent_labels() -> list[dict]:
    migrate_legacy_labels()
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(LABELS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:20]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries.append(
                {
                    "id": data["id"],
                    "filename": data["image"]["original_filename"],
                    "source": data["image"].get("source", "upload"),
                    "count": len(data["objects"]),
                    "created_at": data["created_at"],
                }
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            app.logger.warning("Skipping invalid label record: %s", path.name)
    return entries


def migrate_legacy_labels() -> None:
    """Copy records from the former Compose volume into the project folder once."""
    global legacy_labels_migrated
    if legacy_labels_migrated:
        return
    with migration_lock:
        if legacy_labels_migrated:
            return
        if LEGACY_LABELS_DIR.is_dir():
            LABELS_DIR.mkdir(parents=True, exist_ok=True)
            for source_json in LEGACY_LABELS_DIR.glob("*.json"):
                target_json = LABELS_DIR / source_json.name
                if target_json.exists():
                    continue
                try:
                    record = json.loads(source_json.read_text(encoding="utf-8"))
                    image_name = record.get("image", {}).get("file", "")
                    if not image_name or Path(image_name).name != image_name:
                        continue
                    source_image = LEGACY_LABELS_DIR / image_name
                    if source_image.is_file():
                        shutil.copy2(source_image, LABELS_DIR / image_name)
                        shutil.copy2(source_json, target_json)
                except (OSError, AttributeError, TypeError, json.JSONDecodeError):
                    app.logger.warning("Could not migrate old label record %s", source_json.name)
        legacy_labels_migrated = True


@app.route("/label")
def label_page():
    return render_template("label.html", labels=recent_labels(), raw_images=available_raw_images())


@app.post("/api/label-state")
def label_state():
    """Return combined AI detections and verified labels for an image."""
    payload = request.get_json(silent=True) or {}
    source_type = payload.get("source", "upload")
    filename = payload.get("filename", "")
    if not isinstance(source_type, str) or source_type not in {"upload", "rawimages"} or not isinstance(filename, str):
        return jsonify(error="Invalid image source."), 400
    filename = Path(filename).name
    if not filename:
        return jsonify(error="Choose an image first."), 400

    record = find_label_record(source_type, filename)
    objects = record.get("objects", []) if record else []
    detections = []
    ground_truth = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        ocr = item.get("ocr", {})
        if not isinstance(ocr, dict):
            ocr = {}
        if record and record.get("ai_detection_complete") and item.get("ai_detection", False):
            detections.append(
                {
                    "bbox_xyxy_pixels": item.get("bbox_xyxy_pixels", []),
                    "text": ocr.get("ai_result") or "",
                    "ocr_confidence": item.get("ocr_confidence"),
                    "detection_confidence": item.get("detection_confidence"),
                    "country": item.get("country"),
                    "country_confidence": item.get("country_confidence"),
                }
            )
        verified = ocr.get("verified_result") or item.get("ground_truth")
        if isinstance(verified, str) and verified.strip():
            ground_truth.append(item)
    return jsonify(
        ai_detection_complete=bool(record and record.get("ai_detection_complete")),
        detections=detections,
        ground_truth=ground_truth,
    )


@app.post("/api/label-detections")
def detect_plates_for_labeling():
    """Run AI and store its boxes and OCR guesses in the image's label record."""
    source_type = request.form.get("source", "upload")
    if source_type == "rawimages":
        filename = request.form.get("rawimage", "")
        if Path(filename).name != filename or filename not in available_raw_images():
            return jsonify(error="Choose an image from raw-dataset-detector/"), 400
        source_path = RAW_IMAGES_DIR / filename
        if source_path.stat().st_size > 15 * 1024 * 1024:
            return jsonify(error="Image is too large. The limit is 15 MB."), 400
        image_bytes = source_path.read_bytes()
    else:
        source_type = "upload"
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Choose an image first."), 400
        filename = upload.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
            return jsonify(error="Use a JPG, PNG, WebP, or BMP image."), 400
        image_bytes = upload.read()

    if len(image_bytes) > 15 * 1024 * 1024:
        return jsonify(error="Image is too large. The limit is 15 MB."), 400
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(error="The selected file could not be read as an image."), 400
    height, width = image.shape[:2]
    if width * height > MAX_IMAGE_PIXELS:
        return jsonify(error="That image is too large to process (maximum 40 megapixels)."), 400

    try:
        detector, ocr = get_alpr()
        with inference_lock:
            detections = detect_plates(image, detector)
        # Stage C: run PaddleOCR recognition-only on each plate crop.
        for detection in detections:
            crop = crop_plate(image, detection["bbox_xyxy_pixels"])
            ocr_result = recognize_plate_text(crop, ocr)
            detection["text"] = ocr_result.text if ocr_result else ""
            detection["ocr_confidence"] = ocr_result.confidence if ocr_result else None
        result_id = persist_detection(source_type, filename, image_bytes, image, detections)
        return jsonify(result_id=result_id, detections=detections), 200
    except Exception:
        app.logger.exception("Labeling image detection or result saving failed")
        return jsonify(error="AI detection failed. Check the server logs and try again."), 500


@app.get("/raw-images/<path:filename>")
def raw_image_file(filename: str):
    if Path(filename).name != filename or Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
        return "Not found", 404
    if filename not in available_raw_images():
        return "Not found", 404
    return send_from_directory(RAW_IMAGES_DIR, filename)


@app.post("/api/labels")
def save_label():
    migrate_legacy_labels()
    source_type = request.form.get("source", "upload")
    if source_type == "rawimages":
        original_filename = request.form.get("rawimage", "")
        if Path(original_filename).name != original_filename or original_filename not in available_raw_images():
            return jsonify(error="Choose an image from raw-dataset-detector/"), 400
        source_path = RAW_IMAGES_DIR / original_filename
        if source_path.stat().st_size > 15 * 1024 * 1024:
            return jsonify(error="Image is too large. The limit is 15 MB."), 400
        image_bytes = source_path.read_bytes()
    else:
        source_type = "upload"
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Choose an image first."), 400
        original_filename = upload.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        image_bytes = upload.read()

    extension = Path(original_filename).suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        return jsonify(error="Use a JPG, PNG, WebP, or BMP image."), 400

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(error="The uploaded file is not a readable image."), 400
    height, width = image.shape[:2]
    if width * height > MAX_IMAGE_PIXELS:
        return jsonify(error="Image is too large (maximum 40 megapixels)."), 400
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    existing_label_record = find_label_record(source_type, original_filename, image_sha256)

    try:
        raw_objects = json.loads(request.form.get("annotations", "[]"))
    except json.JSONDecodeError:
        return jsonify(error="Annotation data was not valid JSON."), 400
    if not isinstance(raw_objects, list) or not raw_objects:
        return jsonify(error="Draw at least one plate box before saving."), 400
    if len(raw_objects) > 100:
        return jsonify(error="A single image can contain at most 100 boxes."), 400

    def ai_object_for_box(bounds: list[float]) -> dict | None:
        if not isinstance(existing_label_record, dict):
            return None
        best_overlap = 0.0
        best_object = None
        for candidate in existing_label_record.get("objects", []):
            if not isinstance(candidate, dict):
                continue
            ocr = candidate.get("ocr", {})
            predicted = candidate.get("bbox_xyxy_pixels")
            if not isinstance(ocr, dict) or not isinstance(predicted, list) or len(predicted) != 4:
                continue
            if not candidate.get("ai_detection") and not ocr.get("ai_result"):
                continue
            try:
                overlap = box_iou(bounds, [float(value) for value in predicted])
            except (TypeError, ValueError):
                continue
            if overlap > best_overlap:
                best_overlap = overlap
                best_object = candidate
        return best_object if best_overlap >= 0.15 else None

    objects = []
    for item in raw_objects:
        if not isinstance(item, dict) or not isinstance(item.get("bbox"), list) or len(item["bbox"]) != 4:
            return jsonify(error="One of the plate boxes is invalid."), 400
        try:
            x1, y1, x2, y2 = (float(value) for value in item["bbox"])
        except (TypeError, ValueError):
            return jsonify(error="One of the plate boxes has invalid coordinates."), 400
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            return jsonify(error="One of the plate boxes has invalid coordinates."), 400
        x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
        y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return jsonify(error="Each box must have a positive width and height."), 400
        text = item.get("text", "")
        if not isinstance(text, str) or len(text) > 32:
            return jsonify(error="Plate text must be 32 characters or fewer."), 400
        matched_ai_object = ai_object_for_box([x1, y1, x2, y2])
        matched_ai_ocr = matched_ai_object.get("ocr", {}) if matched_ai_object else {}
        ai_ocr_result = matched_ai_ocr.get("ai_result", "") if isinstance(matched_ai_ocr, dict) else ""
        ai_ocr_result = ai_ocr_result if isinstance(ai_ocr_result, str) else ""
        text = text.strip() or ai_ocr_result
        if not text:
            return jsonify(
                error="Enter the plate text, or leave it blank only when a matching AI guess is available."
            ), 400
        center_x = ((x1 + x2) / 2) / width
        center_y = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        objects.append(
            {
                "class": "license_plate",
                "ai_detection": matched_ai_object is not None,
                "ground_truth": text.strip(),
                "ocr": {
                    "ai_result": ai_ocr_result or None,
                    "verified_result": text.strip(),
                },
                "ocr_confidence": matched_ai_object.get("ocr_confidence") if matched_ai_object else None,
                "detection_confidence": matched_ai_object.get("detection_confidence") if matched_ai_object else None,
                "bbox_xyxy_pixels": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "bbox_normalized_xywh": [
                    round(center_x, 6), round(center_y, 6), round(box_width, 6), round(box_height, 6)
                ],
            }
        )

    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        return jsonify(error="Could not save the uploaded image."), 500
    # Reuse the existing per-image record so detections and verified labels stay together.
    try:
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        app.logger.exception("Could not create label directory %s", LABELS_DIR)
        return jsonify(error="Could not save the label. Check that docker/labels is writable."), 500
    label_save_lock.acquire()
    if existing_label_record:
        record_id = existing_label_record.get("id", "")
        previous_image = existing_label_record.get("image", {})
        image_filename = previous_image.get("file", f"{record_id}.jpg")
        if (
            not isinstance(record_id, str)
            or len(record_id) != 32
            or not isinstance(image_filename, str)
            or Path(image_filename).name != image_filename
        ):
            existing_label_record = None
    if existing_label_record:
        updated_existing = True
    else:
        record_id = uuid4().hex
        image_filename = f"{record_id}.jpg"
        updated_existing = False
    image_path = LABELS_DIR / image_filename
    json_path = LABELS_DIR / f"{record_id}.json"
    record = {
        "schema": "platelens-label-v2",
        "id": record_id,
        "created_at": existing_label_record.get("created_at") if existing_label_record else datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ai_detection_complete": bool(existing_label_record and existing_label_record.get("ai_detection_complete")),
        "image": {
            "file": image_filename,
            "original_filename": original_filename,
            "source": source_type,
            "sha256": image_sha256,
            "width": width,
            "height": height,
        },
        "objects": objects,
    }
    try:
        image_path.write_bytes(encoded.tobytes())
        temporary_json_path = LABELS_DIR / f".{record_id}.tmp"
        temporary_json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_json_path.replace(json_path)
    except OSError:
        app.logger.exception("Could not persist label record in %s", LABELS_DIR)
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            app.logger.warning("Could not remove incomplete label image %s", image_path)
        label_save_lock.release()
        return jsonify(
            error=(
                "Could not save the label. Check that PUID/PGID match your server account "
                "and docker/labels is writable."
            )
        ), 500

    label_save_lock.release()
    return jsonify(id=record_id, json_url=f"/labels/{record_id}.json", updated=updated_existing), (
        200 if updated_existing else 201
    )


@app.get("/labels/<label_id>.json")
def download_label(label_id: str):
    if len(label_id) != 32 or any(char not in "0123456789abcdef" for char in label_id):
        return "Not found", 404
    path = LABELS_DIR / f"{label_id}.json"
    if not path.is_file():
        return "Not found", 404
    return send_file(path, mimetype="application/json", as_attachment=True, download_name=f"{label_id}.json")


@app.errorhandler(413)
def upload_too_large(_error):
    if request.path in {"/api/labels", "/api/label-detections"}:
        return jsonify(error="Image is too large. The upload limit is 15 MB."), 413
    return "Image is too large. The upload limit is 15 MB.", 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

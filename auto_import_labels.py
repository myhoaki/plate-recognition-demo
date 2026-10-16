"""
Auto-import legacy Flask labels (docker/labels/) into the FastAPI PlateLens database.

Reads label JSONs + images from docker/labels/, registers them as
source="registered" images with annotations, marks them verified, and creates
a detector snapshot.

Run inside the API container:
    docker cp auto_import_labels.py platelens-api-1:/app/auto_import_labels.py
    docker compose exec api python /app/auto_import_labels.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Add the app package to the path so we can import models/database
sys.path.insert(0, "/app")

from app.database import SessionLocal, engine
from app.models import Base, ImageRecord, Annotation, DatasetSplitAssignment
from app.storage import DATASETS_DIR
from sqlalchemy import select, func


LABELS_DIR = Path("/app/docker/labels")
RAW_ROOT = DATASETS_DIR / "raw"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_all():
    db = SessionLocal()
    try:
        # Ensure tables exist
        Base.metadata.create_all(bind=engine)

        json_files = sorted(LABELS_DIR.glob("*.json"))
        print(f"Found {len(json_files)} label JSON files in {LABELS_DIR}")

        registered = 0
        skipped = 0
        errors = []

        for json_path in json_files:
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                errors.append((json_path.name, f"JSON parse error: {exc}"))
                skipped += 1
                continue

            image_info = data.get("image", {})
            original_filename = image_info.get("original_filename", "")
            sha256 = image_info.get("sha256", "")
            width = int(image_info.get("width", 0))
            height = int(image_info.get("height", 0))
            source = image_info.get("source", "rawimages")

            if not original_filename or not sha256:
                errors.append((json_path.name, "missing filename or sha256"))
                skipped += 1
                continue

            # Find the corresponding image file
            image_file = LABELS_DIR / image_info.get("file", "")
            if not image_file.is_file():
                # Try by sha256-based name
                image_file = LABELS_DIR / f"{sha256}.jpg"
            if not image_file.is_file():
                # Try by original filename
                image_file = LABELS_DIR / original_filename
            if not image_file.is_file():
                errors.append((json_path.name, f"image file not found: {image_info.get('file', '')}"))
                skipped += 1
                continue

            # Compute actual SHA-256 of the image file
            actual_sha256 = sha256_file(image_file)

            # Check if already registered (by sha256)
            existing = db.scalar(
                select(ImageRecord).where(ImageRecord.sha256 == actual_sha256)
            )
            if existing:
                # Still ensure it's verified
                if existing.review_status != "verified":
                    existing.review_status = "verified"
                    existing.verified_at = datetime.now(timezone.utc)
                    db.commit()
                continue

            # Copy image to datasets/raw/<sha256[:2]>/<sha256><ext>
            ext = image_file.suffix.lower()
            dest_dir = RAW_ROOT / actual_sha256[:2]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{actual_sha256}{ext}"

            if not dest_path.exists():
                shutil.copy2(image_file, dest_path)

            # Verify the copy
            if sha256_file(dest_path) != actual_sha256:
                errors.append((json_path.name, "copy hash mismatch"))
                skipped += 1
                continue

            storage_key = f"datasets/raw/{actual_sha256[:2]}/{actual_sha256}{ext}"

            # Create ImageRecord (verified)
            image_id = uuid4().hex
            image = ImageRecord(
                id=image_id,
                source="registered",
                original_filename=original_filename,
                sha256=actual_sha256,
                storage_key=storage_key,
                width=width,
                height=height,
                review_status="verified",
                verified_at=datetime.now(timezone.utc),
                detection_complete=True,
            )
            db.add(image)
            db.flush()

            # Create annotations from objects
            objects = data.get("objects", [])
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                bbox = obj.get("bbox_xyxy_pixels", [])
                if not bbox or len(bbox) != 4:
                    continue
                norm = obj.get("bbox_normalized_xywh")
                if not norm or len(norm) != 4:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    cx = (x1 + x2) / 2 / width if width else 0
                    cy = (y1 + y2) / 2 / height if height else 0
                    bw = (x2 - x1) / width if width else 0
                    bh = (y2 - y1) / height if height else 0
                    norm = [cx, cy, bw, bh]

                verified_text = ""
                ocr = obj.get("ocr", {}) if isinstance(obj.get("ocr"), dict) else {}
                if isinstance(ocr, dict):
                    verified_text = ocr.get("verified_result") or obj.get("ground_truth") or ""

                annotation = Annotation(
                    image_id=image_id,
                    bbox_xyxy_pixels=[round(float(v), 2) for v in bbox],
                    bbox_normalized_xywh=[round(float(v), 6) for v in norm],
                    ai_detection=bool(obj.get("ai_detection", False)),
                    ai_result=ocr.get("ai_result") if isinstance(ocr, dict) else None,
                    verified_result=verified_text or None,
                    detection_confidence=obj.get("detection_confidence"),
                    ocr_confidence=obj.get("ocr_confidence"),
                    country=obj.get("country"),
                    country_confidence=obj.get("country_confidence"),
                )
                db.add(annotation)

            # Create split assignment (default to train for now)
            db.add(
                DatasetSplitAssignment(
                    image_id=image_id,
                    task="detector",
                    split="train",
                    group_id=None,
                    source_dataset="legacy-labels",
                )
            )

            db.commit()
            registered += 1
            if registered % 50 == 0:
                print(f"  Registered {registered} images...")

        print(f"\n=== Import complete ===")
        print(f"Registered: {registered}")
        print(f"Skipped: {skipped}")
        if errors:
            print(f"Errors ({len(errors)}):")
            for name, err in errors[:10]:
                print(f"  {name}: {err}")

        # Now create detector snapshot
        print("\n=== Creating detector snapshot ===")
        from app.datasets import create_detection_snapshot

        try:
            snapshot = create_detection_snapshot()
            print(f"Snapshot created: {snapshot.id}")
            print(f"  Task: {snapshot.task}")
            print(f"  Split counts: {snapshot.split_counts}")
            print(f"  Path: {snapshot.path}")
            print(f"  Manifest SHA-256: {snapshot.manifest_sha256[:16]}...")
        except ValueError as exc:
            print(f"Snapshot creation failed: {exc}")
            print("You may need more verified images with annotations.")

    finally:
        db.close()


if __name__ == "__main__":
    import_all()

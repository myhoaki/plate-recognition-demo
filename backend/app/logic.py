from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Annotation, ImageRecord


def iou(first: list[float], second: list[float]) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1.0, a + b - intersection)


def image_payload(image: ImageRecord, include_annotations: bool = False) -> dict:
    data = {
        "id": image.id,
        "source": image.source,
        "filename": image.original_filename,
        "width": image.width,
        "height": image.height,
        "detection_complete": image.detection_complete,
        "verified": image.verified_at is not None,
        "verified_at": image.verified_at.isoformat() if image.verified_at else None,
        "file_url": f"/api/images/{image.id}/file",
    }
    if include_annotations:
        data["annotations"] = [annotation_payload(item) for item in image.annotations]
    return data


def annotation_payload(item: Annotation) -> dict:
    return {
        "id": item.id,
        "bbox_xyxy_pixels": item.bbox_xyxy_pixels,
        "bbox_normalized_xywh": item.bbox_normalized_xywh,
        "ai_detection": item.ai_detection,
        "ai_result": item.ai_result,
        "verified_result": item.verified_result,
        "detection_confidence": item.detection_confidence,
        "ocr_confidence": item.ocr_confidence,
        "country": item.country,
        "country_confidence": item.country_confidence,
    }


def sync_raw_images(db: Session, batch_size: int = 500, progress_callback=None) -> int:
    from .storage import IMAGE_EXTENSIONS, RAW_IMAGES_DIR, image_size, sha256_file

    if not RAW_IMAGES_DIR.is_dir():
        return 0
    paths = sorted(
        (path for path in RAW_IMAGES_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda item: item.name.casefold(),
    )
    existing = {
        image.original_filename: image
        for image in db.scalars(select(ImageRecord).where(ImageRecord.source == "rawimages")).all()
    }
    count = 0
    processed = 0
    for start in range(0, len(paths), batch_size):
        for path in paths[start : start + batch_size]:
            processed += 1
            digest = sha256_file(path)
            image = existing.get(path.name)
            try:
                width, height = image_size(path)
            except Exception:
                # A single unreadable file should not prevent the rest of the gallery from loading.
                continue
            if image is None:
                import uuid

                image = ImageRecord(
                    id=uuid.uuid4().hex,
                    source="rawimages",
                    original_filename=path.name,
                    sha256=digest,
                    storage_key=f"rawimages/{path.name}",
                    width=width,
                    height=height,
                )
                db.add(image)
                existing[path.name] = image
                count += 1
            elif image.sha256 != digest:
                image.sha256 = digest
                image.width = width
                image.height = height
                image.storage_key = f"rawimages/{path.name}"
                image.detection_complete = False
                image.verified_at = None
                image.annotations.clear()
                count += 1
        db.commit()
        if progress_callback:
            progress_callback(processed, len(paths), count)
    return count


def import_legacy_records(db: Session) -> int:
    import json
    import uuid
    from pathlib import Path

    from .models import Annotation
    from .storage import LEGACY_LABELS_DIRS, RAW_IMAGES_DIR, import_legacy_image, sha256_file

    imported = 0
    seen = set()
    for legacy_dir in LEGACY_LABELS_DIRS:
        if not legacy_dir.is_dir():
            continue
        for json_path in legacy_dir.glob("*.json"):
            if json_path.name in seen:
                continue
            seen.add(json_path.name)
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                old_image = data.get("image", {})
                source = old_image.get("source", "upload")
                filename = old_image.get("original_filename")
                if not filename or source not in {"rawimages", "upload"}:
                    continue
                present = db.scalar(
                    select(ImageRecord).where(
                        ImageRecord.source == source, ImageRecord.original_filename == filename
                    )
                )
                if present and present.annotations:
                    continue
                storage_key = import_legacy_image(data, legacy_dir)
                if not storage_key:
                    continue
                file_path = RAW_IMAGES_DIR / filename if storage_key.startswith("rawimages/") else Path("/data") / storage_key
                if not file_path.is_file():
                    continue
                digest = old_image.get("sha256") or sha256_file(file_path)
                image = present
                if image is None:
                    record_id = data.get("id")
                    if not isinstance(record_id, str) or len(record_id) != 32:
                        record_id = uuid.uuid4().hex
                    image = ImageRecord(
                        id=record_id,
                        source=source,
                        original_filename=filename,
                        sha256=digest,
                        storage_key=storage_key,
                        width=int(old_image.get("width") or 1),
                        height=int(old_image.get("height") or 1),
                        detection_complete=bool(data.get("ai_detection_complete")),
                    )
                    db.add(image)
                    db.flush()
                else:
                    image.detection_complete = bool(data.get("ai_detection_complete"))
                for old in data.get("objects", []):
                    if not isinstance(old, dict) or not isinstance(old.get("bbox_xyxy_pixels"), list):
                        continue
                    ocr = old.get("ocr") or {}
                    db.add(
                        Annotation(
                            image_id=image.id,
                            bbox_xyxy_pixels=old["bbox_xyxy_pixels"],
                            bbox_normalized_xywh=old.get("bbox_normalized_xywh"),
                            ai_detection=bool(old.get("ai_detection")),
                            ai_result=ocr.get("ai_result"),
                            verified_result=ocr.get("verified_result") or old.get("ground_truth") or None,
                            detection_confidence=old.get("detection_confidence"),
                            ocr_confidence=old.get("ocr_confidence"),
                            country=old.get("country"),
                            country_confidence=old.get("country_confidence"),
                        )
                    )
                imported += 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    db.commit()
    return imported

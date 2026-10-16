from sqlalchemy import func, select
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
        "verified": image.review_status == "verified",
        "review_status": image.review_status,
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


def _normalize_plate_text(value: str) -> str:
    return "".join(value.upper().split())


def _edit_distance(first: str, second: str) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for row, char_first in enumerate(first, 1):
        current = [row]
        for col, char_second in enumerate(second, 1):
            current.append(min(
                current[-1] + 1,
                previous[col] + 1,
                previous[col - 1] + (char_first != char_second),
            ))
        previous = current
    return previous[-1]


def metrics_payload(db: Session) -> dict:
    reviewable = ImageRecord.source.in_(["registered", "datasets_folder"])
    total_images = db.scalar(select(func.count()).select_from(ImageRecord).where(reviewable)) or 0
    images_needing_review = db.scalar(
        select(func.count()).select_from(ImageRecord).where(reviewable, ImageRecord.review_status == "needs_review")
    ) or 0
    verified_plates = db.scalar(
        select(func.count())
        .select_from(Annotation)
        .join(ImageRecord, ImageRecord.id == Annotation.image_id)
        .where(reviewable, ImageRecord.review_status == "verified")
    ) or 0
    detected_verified_plates = db.scalar(
        select(func.count())
        .select_from(Annotation)
        .join(ImageRecord, ImageRecord.id == Annotation.image_id)
        .where(reviewable, ImageRecord.review_status == "verified", Annotation.ai_detection.is_(True))
    ) or 0
    pairs = db.execute(
        select(Annotation.ai_result, Annotation.verified_result)
        .join(ImageRecord, ImageRecord.id == Annotation.image_id)
        .where(
            ImageRecord.review_status == "verified",
            reviewable,
            Annotation.ai_detection.is_(True),
            Annotation.ai_result.is_not(None),
            Annotation.verified_result.is_not(None),
        )
    ).all()
    exact_matches = 0
    edit_errors = 0
    reference_characters = 0
    for prediction, reference in pairs:
        normalized_prediction = _normalize_plate_text(prediction or "")
        normalized_reference = _normalize_plate_text(reference or "")
        if not normalized_reference:
            continue
        exact_matches += normalized_prediction == normalized_reference
        edit_errors += _edit_distance(normalized_prediction, normalized_reference)
        reference_characters += len(normalized_reference)

    return {
        "total_images": total_images,
        "verified_plates": verified_plates,
        "images_needing_review": images_needing_review,
        "ocr_evaluated_plates": len(pairs),
        "ocr_exact_match_accuracy": exact_matches / len(pairs) if pairs else None,
        "character_error_rate": edit_errors / reference_characters if reference_characters else None,
        "detection_evaluated_plates": verified_plates,
        "detection_recall": detected_verified_plates / verified_plates if verified_plates else None,
    }


def sync_raw_images(db: Session, batch_size: int = 500, progress_callback=None) -> int:
    from .models import Annotation
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
            import_sidecar = image is None or image.sha256 != digest
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
                image.review_status = "needs_review"
                image.annotations.clear()
                count += 1
            # Optional imported YOLO sidecar. It is treated as an unverified
            # annotation for review; it never becomes training ground truth until
            # a user verifies the image in the review interface.
            label_path = RAW_IMAGES_DIR / "labels" / f"{path.name}.txt"
            if import_sidecar and label_path.is_file() and not image.annotations:
                imported = []
                try:
                    for line in label_path.read_text(encoding="utf-8").splitlines():
                        values = [float(value) for value in line.split()]
                        if len(values) < 5 or int(values[0]) != 0:
                            continue
                        coordinates = values[1:]
                        if len(coordinates) == 4:
                            cx, cy, bw, bh = coordinates
                            bounds = [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]
                        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
                            xs, ys = coordinates[0::2], coordinates[1::2]
                            bounds = [min(xs), min(ys), max(xs), max(ys)]
                        else:
                            continue
                        x1, y1, x2, y2 = bounds
                        x1, x2 = max(0, min(1, x1)), max(0, min(1, x2))
                        y1, y2 = max(0, min(1, y1)), max(0, min(1, y2))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        imported.append(Annotation(
                            image_id=image.id,
                            bbox_xyxy_pixels=[x1*width, y1*height, x2*width, y2*height],
                            bbox_normalized_xywh=[(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1],
                            ai_detection=False,
                            verified_result="",
                        ))
                    if imported:
                        db.add_all(imported)
                except (OSError, ValueError, OverflowError):
                    # Keep the image available for manual labeling; malformed
                    # external labels are not trusted or applied.
                    pass
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


def sync_datasets_folder(db: Session, batch_size: int = 500, progress_callback=None) -> int:
    """Scan datasets/imports/ for images and register them for review.

    Images are stored with source="datasets_folder" and a storage_key of the
    form "datasets/imports/<relative-path>".  Existing YOLO sidecar labels
    (same directory, ``labels/`` sub-tree, or same stem with .txt extension)
    are imported as unverified annotations so reviewers see pre-drawn boxes.
    Already-registered entries are updated only when the file hash changes.
    """
    from .models import Annotation
    from .storage import DATASETS_IMPORTS_DIR, IMAGE_EXTENSIONS, image_size, sha256_file

    if not DATASETS_IMPORTS_DIR.is_dir():
        return 0

    imports_root = DATASETS_IMPORTS_DIR.resolve()
    image_files = sorted(
        (p for p in imports_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.as_posix().casefold(),
    )

    existing = {
        image.storage_key: image
        for image in db.scalars(select(ImageRecord).where(ImageRecord.source == "datasets_folder")).all()
    }

    count = 0
    processed = 0
    for start in range(0, len(image_files), batch_size):
        for path in image_files[start: start + batch_size]:
            processed += 1
            relative = path.relative_to(imports_root)
            storage_key = f"datasets/imports/{relative.as_posix()}"
            digest = sha256_file(path)
            image = existing.get(storage_key)
            import_sidecar = image is None or image.sha256 != digest
            try:
                width, height = image_size(path)
            except Exception:
                continue
            if image is None:
                import uuid
                image = ImageRecord(
                    id=uuid.uuid4().hex,
                    source="datasets_folder",
                    original_filename=relative.as_posix(),
                    sha256=digest,
                    storage_key=storage_key,
                    width=width,
                    height=height,
                )
                db.add(image)
                existing[storage_key] = image
                count += 1
            elif image.sha256 != digest:
                image.sha256 = digest
                image.width = width
                image.height = height
                image.detection_complete = False
                image.verified_at = None
                image.review_status = "needs_review"
                image.annotations.clear()
                count += 1

            # Try to load a matching YOLO sidecar label file.
            # Supports both flat layouts (labels/<stem>.txt) and nested
            # layouts (…/images/…/<stem>.ext → …/labels/…/<stem>.txt).
            if import_sidecar:
                parts = list(relative.parts)
                images_idx = next(
                    (i for i, p in enumerate(parts) if p.lower() == "images"), None
                )
                if images_idx is not None:
                    label_parts = parts[:]
                    label_parts[images_idx] = "labels"
                    label_path = imports_root.joinpath(*label_parts).with_suffix(".txt")
                else:
                    label_path = path.parent / "labels" / path.with_suffix(".txt").name
                    if not label_path.is_file():
                        label_path = path.with_suffix(".txt")

                db.flush()
                if label_path.is_file() and not image.annotations:
                    imported_anns = []
                    try:
                        for line in label_path.read_text(encoding="utf-8").splitlines():
                            values = [float(v) for v in line.split()]
                            if len(values) < 5:
                                continue
                            coords = values[1:]
                            if len(coords) == 4:
                                cx, cy, bw, bh = coords
                                bounds = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                            elif len(coords) >= 6 and len(coords) % 2 == 0:
                                xs, ys = coords[0::2], coords[1::2]
                                bounds = [min(xs), min(ys), max(xs), max(ys)]
                            else:
                                continue
                            x1, y1, x2, y2 = bounds
                            x1, x2 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))
                            y1, y2 = max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))
                            if x2 <= x1 or y2 <= y1:
                                continue
                            imported_anns.append(Annotation(
                                image_id=image.id,
                                bbox_xyxy_pixels=[x1 * width, y1 * height, x2 * width, y2 * height],
                                bbox_normalized_xywh=[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
                                ai_detection=False,
                                verified_result="",
                            ))
                        if imported_anns:
                            db.add_all(imported_anns)
                    except (OSError, ValueError, OverflowError):
                        pass

        db.commit()
        if progress_callback:
            progress_callback(processed, len(image_files), count)
    return count

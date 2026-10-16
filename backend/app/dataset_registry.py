"""Versioned dataset manifests, canonical imports, audits, and snapshot exports.

Run inside the API/worker environment with ``python -m app.dataset_registry``.
The existing PostgreSQL image and annotation rows remain authoritative; manifests
freeze their verified state and stable split assignments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
from sqlalchemy import select

from .database import Base, SessionLocal, engine
from .models import Annotation, DatasetManifest, DatasetProvenance, DatasetSnapshot, DatasetSplitAssignment, ImageRecord
from .storage import DATASETS_DIR, IMAGE_EXTENSIONS, image_path, sha256_file

SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
TASKS = ("detector", "ocr")


def stable_sample_id(content_sha256: str) -> str:
    """Stable content-derived ID; filenames and source directory names do not affect it."""
    return hashlib.sha256(bytes.fromhex(content_sha256)).hexdigest()[:32]


def content_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_yolo_box(class_id: int, values: list[float], width: int, height: int, class_count: int = 1) -> list[float]:
    if class_id < 0 or class_id >= class_count:
        raise ValueError(f"class_id {class_id} is outside 0..{class_count - 1}")
    if len(values) != 4:
        raise ValueError("only YOLO bounding-box labels with four coordinates are supported")
    cx, cy, box_width, box_height = values
    if width <= 0 or height <= 0 or not all(0 <= value <= 1 for value in values):
        raise ValueError("image dimensions and normalized YOLO values must be positive and within 0..1")
    if box_width <= 0 or box_height <= 0 or cx < box_width / 2 or cy < box_height / 2 or cx + box_width / 2 > 1 or cy + box_height / 2 > 1:
        raise ValueError("YOLO bounding box is empty or extends outside the image")
    return [cx, cy, box_width, box_height]


def pixels_to_yolo(box_xyxy: list[float], width: int, height: int) -> list[float]:
    if len(box_xyxy) != 4 or width <= 0 or height <= 0:
        raise ValueError("expected an xyxy box and positive image dimensions")
    x1, y1, x2, y2 = map(float, box_xyxy)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("pixel box is malformed or outside the image")
    return [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height, (x2 - x1) / width, (y2 - y1) / height]


def yolo_label_line(class_id: int, bbox: list[float]) -> str:
    if len(bbox) != 4 or class_id < 0:
        raise ValueError("invalid class or YOLO box")
    if any(not 0 <= float(value) <= 1 for value in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError("YOLO box values are invalid")
    return f"{class_id} " + " ".join(f"{float(value):.8f}" for value in bbox)


def group_split_conflicts(samples: list[dict]) -> list[dict]:
    values: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.get("group_id") and sample.get("split"):
            values[str(sample["group_id"])].add(str(sample["split"]))
    return [{"group_id": group_id, "splits": sorted(splits)} for group_id, splits in sorted(values.items()) if len(splits) > 1]


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("path must be relative and cannot traverse parent directories")
    return path


def initial_split(group_id: str) -> str:
    """Deterministic 80/10/10 split for first-time registration only."""
    value = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "val" if value < 90 else "test"


def choose_split(sample_id: str, group_id: str | None, existing_splits: dict[str, str], group_splits: dict[str, str], has_manifest: bool) -> str:
    """Preserve an assignment/group; unseen data joins train after registry creation."""
    if sample_id in existing_splits:
        return existing_splits[sample_id]
    if group_id and group_id in group_splits:
        return group_splits[group_id]
    if has_manifest:
        return "train"
    return initial_split(group_id or sample_id)


def split_percentages(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {split: round(100 * counts.get(split, 0) / total, 2) if total else 0.0 for split in SPLITS}


def _manifest_dir(root: Path) -> Path:
    return root / "manifests"


def _load_latest_manifest(db, task: str, root: Path) -> tuple[dict | None, DatasetManifest | None]:
    row = db.scalar(select(DatasetManifest).where(DatasetManifest.task == task).order_by(DatasetManifest.version.desc()).limit(1))
    if not row:
        return None, None
    path = Path(row.path)
    if not path.is_file():
        raise ValueError(f"Registered {task} manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if content_hash(payload) != row.manifest_sha256:
        raise ValueError(f"Registered {task} manifest changed: {path}")
    return payload, row


def _sample_source_path(image: ImageRecord, root: Path) -> str:
    key = image.storage_key
    if key.startswith("datasets/raw/"):
        return "raw/" + key.removeprefix("datasets/raw/")
    if image.source == "rawimages":
        return f"legacy/raw-dataset-detector/{Path(image.original_filename).name}"
    return f"legacy/uploads/{Path(image.storage_key).name}"


def create_manifest(task: str, root: Path | None = None) -> tuple[dict, Path]:
    if task not in TASKS:
        raise ValueError("task must be detector or ocr")
    root = (root or DATASETS_DIR).resolve()
    manifest_dir = _manifest_dir(root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        previous, previous_row = _load_latest_manifest(db, task, root)
        assignments = {
            row.image_id: row
            for row in db.scalars(select(DatasetSplitAssignment).where(DatasetSplitAssignment.task == task)).all()
        }
        query = select(ImageRecord).where(ImageRecord.source == "registered").order_by(ImageRecord.id)
        images = db.scalars(query).all()
        annotations_by_image: dict[str, list[Annotation]] = defaultdict(list)
        for annotation in db.scalars(select(Annotation).order_by(Annotation.image_id, Annotation.id)).all():
            annotations_by_image[annotation.image_id].append(annotation)
        provenance_by_image: dict[str, list[DatasetProvenance]] = defaultdict(list)
        for origin in db.scalars(select(DatasetProvenance).order_by(DatasetProvenance.image_id, DatasetProvenance.id)).all():
            provenance_by_image[origin.image_id].append(origin)
        previous_by_id = {sample["sample_id"]: sample for sample in (previous or {}).get("samples", [])}
        group_splits: dict[str, str] = {}
        for assignment in assignments.values():
            if assignment.group_id and assignment.split in SPLITS:
                if assignment.group_id in group_splits and group_splits[assignment.group_id] != assignment.split:
                    raise ValueError(f"Group {assignment.group_id!r} is assigned to multiple splits")
                group_splits[assignment.group_id] = assignment.split
        for sample in previous_by_id.values():
            if sample.get("group_id") and sample.get("split") in SPLITS:
                group_splits.setdefault(sample["group_id"], sample["split"])
        samples = []
        no_group_count = 0
        for image in images:
            anns = annotations_by_image.get(image.id, [])
            provenance = provenance_by_image.get(image.id, [])
            provenance_meta = [row.metadata_json or {} for row in provenance]
            ocr_annotations = [item for meta in provenance_meta for item in meta.get("ocr_annotations", [])]
            image_verified = image.review_status == "verified" and image.verified_at is not None
            if task == "detector" and (not image_verified or not anns):
                continue
            verified_ocr = [ann for ann in anns if image_verified and ann.verified_result is not None and ann.verified_result != ""]
            verified_imported_ocr = [item for item in ocr_annotations if item.get("verification_status") == "verified"]
            if task == "ocr" and not verified_ocr and not verified_imported_ocr:
                continue
            previous_sample = previous_by_id.get(image.id)
            assignment = assignments.get(image.id)
            group_id = ((previous_sample or {}).get("group_id") or (assignment.group_id if assignment else None)
                        or next((meta.get("group_id") for meta in provenance_meta if meta.get("group_id")), None))
            if not group_id:
                no_group_count += 1
            existing_splits = {sample_id: sample["split"] for sample_id, sample in previous_by_id.items()}
            existing_splits.update({image_id: assignment.split for image_id, assignment in assignments.items()})
            split = choose_split(image.id, group_id, existing_splits, group_splits, bool(previous_row))
            if split not in SPLITS:
                raise ValueError(f"Invalid split {split!r} for sample {image.id}")
            if group_id:
                existing = group_splits.get(group_id)
                if existing and existing != split:
                    raise ValueError(f"Group {group_id!r} is assigned to multiple splits; fix assignments before snapshot creation")
                group_splits[group_id] = split
            annotation_rows = []
            selected_annotations = anns if task == "detector" else verified_ocr
            for ann in selected_annotations:
                norm = ann.bbox_normalized_xywh or pixels_to_yolo(ann.bbox_xyxy_pixels, image.width, image.height)
                x, y, w, h = map(float, norm)
                if not (0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0 and x >= w/2 and y >= h/2 and x+w/2 <= 1 and y+h/2 <= 1):
                    raise ValueError(f"Invalid verified box on sample {image.id}")
                annotation_rows.append({"annotation_id": ann.id, "class_id": 0, "class_name": "license_plate", "bbox_normalized_xywh": [x, y, w, h],
                                        "coordinate_format": "normalized_xywh", "image_width": image.width, "image_height": image.height,
                                        "verification_status": "verified", "verified_text": ann.verified_result or ""})
            selected_ocr = ([{"exact_text": ann.verified_result, "annotation_id": ann.id, "bbox_xyxy_pixels": ann.bbox_xyxy_pixels} for ann in verified_ocr]
                            if verified_ocr else verified_imported_ocr)
            annotation_digest = content_hash({"annotations": annotation_rows, "ocr_annotations": selected_ocr})
            source_dataset = next((row.source_dataset for row in provenance), assignment.source_dataset if assignment else "legacy")
            import_batch = next((row.import_batch for row in provenance), None)
            samples.append({"sample_id": image.id, "relative_image_path": _sample_source_path(image, root), "image_sha256": image.sha256,
                            "annotation_version": annotation_digest, "annotation_sha256": annotation_digest,
                            "verification_status": "verified", "split": split, "source_dataset": source_dataset,
                            "import_batch": import_batch, "group_id": group_id, "tasks": [task], "width": image.width, "height": image.height,
                            "original_filename": image.original_filename, "annotations": annotation_rows,
                            "ocr_annotations": selected_ocr})
            if not assignment:
                assignment = DatasetSplitAssignment(image_id=image.id, task=task, split=split, group_id=group_id,
                                                    source_dataset="legacy" if image.storage_key.startswith("rawimages/") else "registered")
                db.add(assignment)
            else:
                if assignment.split != split:
                    assignment.split = split
                if group_id:
                    assignment.group_id = group_id
        if not samples:
            raise ValueError(f"No eligible verified samples are available for the {task} manifest.")
        # Avoid generating a new version when the frozen sample state is unchanged.
        if previous and previous.get("samples") == samples:
            db.commit()
            return previous, Path(previous_row.path)
        version = (previous_row.version + 1) if previous_row else 1
        payload = {"schema_version": SCHEMA_VERSION, "task": task, "version": version,
                   "created_at": datetime.now(timezone.utc).isoformat(), "split_policy": "Existing assignments are preserved; new verified samples default to train; related known groups share a split.",
                   "grouping_warning": f"{no_group_count} verified samples have no known group ID; leakage prevention cannot be guaranteed for them.",
                   "split_counts": dict(Counter(row["split"] for row in samples)),
                   "split_percentages": split_percentages(dict(Counter(row["split"] for row in samples))),
                   "samples": samples}
        digest = content_hash(payload)
        destination = manifest_dir / f"{task}-v{version:06d}.json"
        with destination.open("x", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        row = DatasetManifest(id=uuid4().hex, task=task, version=version, path=str(destination), manifest_sha256=digest, sample_count=len(samples))
        db.add(row)
        db.commit()
        return payload, destination


def reassign_split(task: str, sample_id: str, split: str, *, confirm_test_change: bool = False, root: Path | None = None) -> dict:
    if task not in TASKS or split not in SPLITS:
        raise ValueError("task and split must be valid")
    with SessionLocal() as db:
        current = db.scalar(select(DatasetSplitAssignment).where(DatasetSplitAssignment.task == task, DatasetSplitAssignment.image_id == sample_id))
        image = db.get(ImageRecord, sample_id)
        if not image or image.source != "registered":
            raise ValueError(f"Sample {sample_id} does not exist")
        group_id = current.group_id if current else None
        group_rows = list(db.scalars(select(DatasetSplitAssignment).where(DatasetSplitAssignment.task == task,
                    DatasetSplitAssignment.group_id == group_id)).all()) if group_id else ([current] if current else [])
        if any(row.split == "test" for row in group_rows) and split != "test" and not confirm_test_change:
            raise ValueError("Changing a test assignment requires --confirm-test-change")
        if current and current.split == "test" and split != "test" and not confirm_test_change:
            raise ValueError("Changing a test assignment requires --confirm-test-change")
        if not current:
            current = DatasetSplitAssignment(image_id=sample_id, task=task, split=split, source_dataset="manual")
            db.add(current)
        else:
            group_rows = group_rows or [current]
            for row in group_rows:
                row.split = split
        db.commit()
    manifest, path = create_manifest(task, root)
    return {"path": str(path), "version": manifest["version"], "sample_id": sample_id, "split": split}


def parse_yolo_file(path: Path, width: int, height: int, class_count: int = 1,
                    class_mapping: dict[int, int] | None = None) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) < 5:
            raise ValueError(f"{path}:{line_no}: expected a class ID and YOLO box or polygon coordinates")
        try:
            class_value = float(values[0])
            coords = [float(value) for value in values[1:]]
        except ValueError:
            raise ValueError(f"{path}:{line_no}: non-numeric YOLO value") from None
        if not class_value.is_integer():
            raise ValueError(f"{path}:{line_no}: class ID must be an integer")
        class_id = int(class_value)
        if class_id < 0 or class_id >= class_count:
            raise ValueError(f"{path}:{line_no}: class ID {class_id} is outside 0..{class_count - 1}")
        if len(coords) == 4:
            bbox = normalize_yolo_box(class_id, coords, width, height, class_count)
            clipped = False
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            if not all(math.isfinite(value) for value in coords):
                raise ValueError(f"{path}:{line_no}: polygon coordinates must be finite")
            if min(coords) < -0.05 or max(coords) > 1.05:
                raise ValueError(f"{path}:{line_no}: polygon extends more than 5% outside the image")
            xs, ys = coords[0::2], coords[1::2]
            x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
            y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
            clipped = min(coords) < 0 or max(coords) > 1
            bbox = normalize_yolo_box(class_id, [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], width, height, class_count)
        else:
            raise ValueError(f"{path}:{line_no}: expected four box values or at least three polygon points")
        rows.append({"class_id": (class_mapping or {}).get(class_id, class_id), "source_class_id": class_id, "clipped": clipped,
                     "bbox_normalized_xywh": bbox})
    return rows


def _source_split(relative: Path) -> str | None:
    for part in relative.parts:
        lowered = part.lower()
        if lowered in {"train", "training"}:
            return "train"
        if lowered in {"valid", "validation", "val"}:
            return "val"
        if lowered in {"test", "testing"}:
            return "test"
    return None


def _label_path(source: Path, relative: Path) -> Path:
    parts = list(relative.parts)
    images_index = next((i for i, part in enumerate(parts) if part.lower() == "images"), None)
    if images_index is not None:
        parts[images_index] = "labels"
        return source.joinpath(*parts).with_suffix(".txt")
    return source / "labels" / relative.with_suffix(".txt")


def _average_hash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("unreadable image")
    small = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    value = 0
    for pixel in small.flatten():
        value = (value << 1) | int(float(pixel) >= mean)
    return value


def _near_duplicate_pairs(items: list[dict], max_distance: int = 4) -> list[dict]:
    """Use four-band LSH so likely close 64-bit hashes are compared efficiently."""
    bands: dict[tuple[int, int], set[int]] = defaultdict(set)
    pairs: set[tuple[int, int]] = set()
    for index, item in enumerate(items):
        value = item["perceptual_hash"]
        candidates: set[int] = set()
        # Eight 8-bit bands guarantee that hashes within four bit flips share
        # at least one band, while avoiding an all-pairs comparison.
        for band in range(8):
            key = (band, (value >> (band * 8)) & 0xFF)
            candidates.update(bands[key])
            bands[key].add(index)
        for other in candidates:
            distance = (value ^ items[other]["perceptual_hash"]).bit_count()
            if distance <= max_distance and item["sha256"] != items[other]["sha256"]:
                pairs.add((other, index))
    return [{"first": Path(items[a]["relative"]).as_posix(), "second": Path(items[b]["relative"]).as_posix(),
             "average_hash_distance": (items[a]["perceptual_hash"] ^ items[b]["perceptual_hash"]).bit_count()} for a, b in sorted(pairs)]


def import_dataset(source: Path, source_dataset: str, task: str, *, import_batch: str | None = None, apply: bool = False, conflict_policy: str = "error") -> dict:
    source = source.resolve(strict=True)
    import_batch = import_batch or source_dataset
    if not source.is_dir() or not source_dataset.strip():
        raise ValueError("source must be a directory and source_dataset must be a non-empty ID")
    root = DATASETS_DIR.resolve()
    image_files = sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.as_posix().casefold())
    report = {"source_dataset": source_dataset, "import_batch": import_batch, "task": task, "dry_run": not apply, "images_seen": len(image_files), "registered": 0, "provenance_records": 0,
              "exact_duplicates": [], "filename_collisions": [], "missing_labels": [], "invalid_annotations": [], "clipped_annotations": [], "conflicts": [], "near_duplicates": [], "errors": []}
    if task not in TASKS:
        raise ValueError("task must be detector or ocr")
    source_class_count = 1
    source_class_names = {0: "license_plate"}
    class_mapping = {0: 0}
    if task == "detector":
        metadata_path = next((candidate for candidate in (source / "dataset.yaml", source / "dataset.yml", source / "data.yaml") if candidate.is_file()), None)
        if metadata_path:
            import yaml
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            names = metadata.get("names", {})
            if isinstance(names, list):
                source_class_names = {index: str(name) for index, name in enumerate(names)}
            elif isinstance(names, dict):
                source_class_names = {int(index): str(name) for index, name in names.items()}
            source_class_count = int(metadata.get("nc") or len(source_class_names) or 1)
            if source_class_names and set(source_class_names) != set(range(source_class_count)):
                raise ValueError(f"{metadata_path.name}: class names must define IDs 0 through {source_class_count - 1}")
            # This application trains a single license_plate-position class. Source
            # subclasses (such as plate shape) are retained in provenance and mapped
            # to that one detection class.
            class_mapping = {class_id: 0 for class_id in range(source_class_count)}
        report["class_mapping"] = {"source_classes": source_class_names, "target": {"0": "license_plate"}}
    ocr_rows: dict[str, list[dict]] = defaultdict(list)
    group_rows: dict[str, str] = {}
    group_csv = source / "groups.csv"
    if group_csv.is_file():
        with group_csv.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"image_path", "group_id"}.issubset(reader.fieldnames or []):
                raise ValueError("groups.csv must contain image_path and group_id columns")
            for row in reader:
                relative = safe_relative_path(row["image_path"]).as_posix()
                if row["group_id"].strip():
                    group_rows[relative] = row["group_id"].strip()
    if task == "ocr":
        annotations_csv = source / "annotations.csv"
        if not annotations_csv.is_file():
            raise ValueError("OCR CSV import requires annotations.csv with image_path and plate_text columns")
        with annotations_csv.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"image_path", "plate_text"}.issubset(reader.fieldnames or []):
                raise ValueError("OCR CSV must contain image_path and plate_text columns")
            for line_no, row in enumerate(reader, 2):
                try:
                    relative = safe_relative_path(row.get("image_path", ""))
                except ValueError:
                    report["errors"].append({"line": line_no, "error": "unsafe image_path"})
                    continue
                text_value = row.get("plate_text", "")
                if not text_value.strip():
                    report["errors"].append({"line": line_no, "error": "plate_text must be non-empty and exact"})
                    continue
                ocr_rows[relative.as_posix()].append({"plate_text": text_value,
                    "verification_status": "verified" if str(row.get("verified", "")).strip().lower() in {"1", "true", "yes", "verified"} else "pending",
                    "is_crop": str(row.get("is_crop", "")).strip().lower() in {"1", "true", "yes"},
                    "group_id": row.get("group_id", "") or None,
                    "object_index": row.get("object_index", ""), "x1": row.get("x1", ""), "y1": row.get("y1", ""),
                    "x2": row.get("x2", ""), "y2": row.get("y2", "")})
    plans = []
    for path in image_files:
        relative = path.relative_to(source)
        try:
            path.resolve(strict=True).relative_to(source)
            digest = sha256_file(path)
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("unreadable image")
            height, width = image.shape[:2]
            annotations = []
            ocr_annotations = []
            source_class_counts: Counter[str] = Counter()
            label_path = _label_path(source, relative)
            if task == "detector":
                if label_path.is_file():
                    annotations = parse_yolo_file(label_path, width, height, source_class_count, class_mapping)
                    source_class_counts.update(str(row["source_class_id"]) for row in annotations)
                    report["clipped_annotations"].extend(
                        {"path": relative.as_posix(), "line": index + 1}
                        for index, row in enumerate(annotations) if row["clipped"]
                    )
                else:
                    report["missing_labels"].append(relative.as_posix())
            else:
                for row in ocr_rows.get(relative.as_posix(), []):
                    coords = [row.get(key, "") for key in ("x1", "y1", "x2", "y2")]
                    bbox = None
                    if all(str(value).strip() for value in coords):
                        bbox = [float(value) for value in coords]
                    elif any(str(value).strip() for value in coords):
                        raise ValueError(f"{relative}: OCR crop requires all four coordinates")
                    elif row["is_crop"]:
                        bbox = [0.0, 0.0, float(width), float(height)]
                    else:
                        try:
                            object_index = int(row.get("object_index") or 0)
                            detected_boxes = parse_yolo_file(label_path, width, height)
                            x, y, bw, bh = detected_boxes[object_index]["bbox_normalized_xywh"]
                            bbox = [(x-bw/2)*width, (y-bh/2)*height, (x+bw/2)*width, (y+bh/2)*height]
                        except (ValueError, IndexError):
                            raise ValueError(f"{relative}: OCR row needs is_crop, pixel coordinates, or a matching YOLO box") from None
                    x1, y1, x2, y2 = bbox
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        raise ValueError(f"{relative}: OCR crop coordinates are outside the image")
                    ocr_annotations.append({**row, "bbox_xyxy_pixels": bbox})
            group_id = group_rows.get(relative.as_posix()) or next((item.get("group_id") for item in ocr_annotations if item.get("group_id")), None)
            plans.append({"path": path, "relative": relative, "sha256": digest, "width": width, "height": height,
                          "annotations": annotations, "ocr_annotations": ocr_annotations, "group_id": group_id,
                          "source_class_counts": dict(source_class_counts),
                          "split": _source_split(relative), "perceptual_hash": _average_hash(path)})
        except Exception as exc:
            report["errors"].append({"path": relative.as_posix(), "error": str(exc)})
    found_paths = {plan["relative"].as_posix() for plan in plans}
    for relative in ocr_rows:
        if relative not in found_paths:
            report["errors"].append({"path": relative, "error": "OCR annotations reference a missing image"})
    basename_paths: dict[str, list[dict]] = defaultdict(list)
    for plan in plans:
        basename_paths[plan["path"].name.casefold()].append(plan)
    report["filename_collisions"] = [{"filename": rows[0]["path"].name, "paths": [row["relative"].as_posix() for row in rows]}
                                      for rows in basename_paths.values() if len(rows) > 1 and len({row["sha256"] for row in rows}) > 1]
    duplicate_groups: dict[str, list[dict]] = defaultdict(list)
    for plan in plans:
        duplicate_groups[plan["sha256"]].append(plan)
    unique_plans = []
    split_priority = {"train": 1, "val": 2, "test": 3, None: 0}
    for digest, origins in duplicate_groups.items():
        splits = sorted({origin["split"] for origin in origins if origin["split"]})
        groups = sorted({origin["group_id"] for origin in origins if origin["group_id"]})
        signatures = {
            tuple(sorted((int(row["class_id"]), tuple(round(float(value), 6) for value in row["bbox_normalized_xywh"]))
                         for row in origin["annotations"]))
            for origin in origins
        }
        selected = max(origins, key=lambda origin: (split_priority[origin["split"]], origin["relative"].as_posix()))
        selected["origins"] = origins
        if splits:
            selected["split"] = max(splits, key=lambda split: split_priority[split])
        if len(origins) > 1:
            report["exact_duplicates"].append({
                "kind": "source_duplicate", "sha256": digest,
                "paths": [origin["relative"].as_posix() for origin in origins],
                "splits": splits, "selected_split": selected["split"] or "train",
            })
        if len(splits) > 1:
            report["conflicts"].append({"kind": "duplicate_split", "sha256": digest,
                                        "paths": [origin["relative"].as_posix() for origin in origins],
                                        "splits": splits, "selected_split": selected["split"]})
        if len(groups) > 1:
            report["conflicts"].append({"kind": "duplicate_group", "sha256": digest, "group_ids": groups})
        if len(signatures) > 1:
            report["conflicts"].append({"kind": "duplicate_annotation", "sha256": digest,
                                        "paths": [origin["relative"].as_posix() for origin in origins]})
        selected["source_class_counts"] = dict(Counter(
            source_class for origin in origins for source_class, count in origin["source_class_counts"].items()
            for _ in range(count)
        ))
        unique_plans.append(selected)
    report["near_duplicates"] = _near_duplicate_pairs(unique_plans)
    if task == "detector":
        report["source_class_counts"] = dict(Counter(
            class_id for plan in unique_plans for class_id, count in plan["source_class_counts"].items()
            for _ in range(count)
        ))
    group_splits: dict[str, set[str]] = defaultdict(set)
    for plan in unique_plans:
        if plan["group_id"] and plan["split"]:
            group_splits[plan["group_id"]].add(plan["split"])
    report["group_split_conflicts"] = [{"group_id": group, "splits": sorted(splits)} for group, splits in group_splits.items() if len(splits) > 1]
    report["conflicts"].extend({"kind": "group_split", **row} for row in report["group_split_conflicts"])
    existing_hashes: set[str] = set()
    with SessionLocal() as db:
        existing_registered = {
            image.sha256: image.id for image in db.scalars(select(ImageRecord).where(ImageRecord.source == "registered")).all()
        }
        for plan in unique_plans:
            existing_id = existing_registered.get(plan["sha256"])
            if existing_id:
                existing_hashes.add(plan["sha256"])
                report["exact_duplicates"].append({"source_path": plan["relative"].as_posix(), "sample_id": existing_id})
        incoming_group_splits: dict[str, set[str]] = defaultdict(set)
        for plan in unique_plans:
            if plan["group_id"] and plan["split"]:
                incoming_group_splits[plan["group_id"]].add(plan["split"])
        current_group_splits: dict[str, set[str]] = defaultdict(set)
        for assignment in db.scalars(select(DatasetSplitAssignment).where(DatasetSplitAssignment.task == task)).all():
            if assignment.group_id:
                current_group_splits[assignment.group_id].add(assignment.split)
        for group_id, incoming_splits in incoming_group_splits.items():
            current = current_group_splits.get(group_id, set())
            all_splits = current | incoming_splits
            if len(all_splits) > 1:
                conflict = {"kind": "group_split", "group_id": group_id, "splits": sorted(all_splits)}
                if conflict not in report["conflicts"]:
                    report["conflicts"].append(conflict)
    if not apply:
        report["registered"] = len(unique_plans) - len(existing_hashes)
        report["provenance_records"] = sum(len(plan["origins"]) for plan in unique_plans if plan["sha256"] not in existing_hashes)
        if conflict_policy == "prefer-holdout":
            for conflict in report["conflicts"]:
                if conflict.get("kind") == "duplicate_split":
                    conflict["resolution"] = "prefer-holdout"
        return report
    root.joinpath("raw").mkdir(parents=True, exist_ok=True)
    root.joinpath("imports").mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        current_assignments = db.scalars(select(DatasetSplitAssignment).where(DatasetSplitAssignment.task == task)).all()
        assignment_by_image = {row.image_id: row for row in current_assignments}
        group_assignments: dict[str, str] = {}
        for current in current_assignments:
            if current.group_id:
                group_assignments.setdefault(current.group_id, current.split)
        incoming_group_defaults = {
            group: next(iter(splits)) if len(splits) == 1 else "train"
            for group, splits in incoming_group_splits.items()
        }
        if conflict_policy == "prefer-holdout":
            for conflict in report["conflicts"]:
                if conflict.get("kind") == "duplicate_split":
                    conflict["resolution"] = "prefer-holdout"
        images_by_hash = {}
        for existing_image in db.scalars(select(ImageRecord).where(ImageRecord.source == "registered").order_by(ImageRecord.id)).all():
            images_by_hash.setdefault(existing_image.sha256, existing_image)
        target_hashes = {plan["sha256"] for plan in unique_plans}
        existing_ids = [image.id for digest, image in images_by_hash.items() if digest in target_hashes]
        annotations_by_image: dict[str, list[Annotation]] = defaultdict(list)
        for start in range(0, len(existing_ids), 10000):
            for annotation in db.scalars(select(Annotation).where(Annotation.image_id.in_(existing_ids[start:start+10000]))).all():
                annotations_by_image[annotation.image_id].append(annotation)
        provenance_keys = {
            (row.image_id, row.source_relative_path)
            for row in db.scalars(select(DatasetProvenance).where(
                DatasetProvenance.source_dataset == source_dataset, DatasetProvenance.import_batch == import_batch
            )).all()
        }
        existing_filenames = {name for name in db.scalars(select(ImageRecord.original_filename).where(ImageRecord.source == "registered")).all()}
        for plan in unique_plans:
            digest = plan["sha256"]
            sample_id = stable_sample_id(digest)
            image = images_by_hash.get(digest)
            if image:
                incoming = plan["annotations"]
                old = annotations_by_image.get(image.id, [])
                if task == "detector" and old and incoming:
                    old_boxes = sorted(tuple(round(float(value), 5) for value in (ann.bbox_normalized_xywh or pixels_to_yolo(ann.bbox_xyxy_pixels, image.width, image.height))) for ann in old)
                    new_boxes = sorted(tuple(round(float(value), 5) for value in row["bbox_normalized_xywh"]) for row in incoming)
                    if old_boxes != new_boxes:
                        report["conflicts"].append({"sample_id": image.id, "kind": "detection_annotation", "existing": old_boxes, "incoming": new_boxes})
                        if conflict_policy == "error":
                            continue
                elif task == "detector" and not old and incoming:
                    for row in incoming:
                        cx, cy, bw, bh = row["bbox_normalized_xywh"]
                        bounds = [(cx-bw/2)*plan["width"], (cy-bh/2)*plan["height"], (cx+bw/2)*plan["width"], (cy+bh/2)*plan["height"]]
                        db.add(Annotation(image_id=image.id, bbox_xyxy_pixels=bounds, bbox_normalized_xywh=[cx,cy,bw,bh], ai_detection=False))
                if task == "ocr" and plan["ocr_annotations"]:
                    prior = db.scalars(select(DatasetProvenance).where(DatasetProvenance.image_id == image.id)).all()
                    prior_texts = {item.get("plate_text") for item in (p.metadata_json or {}).get("ocr_annotations", [])
                                   if item.get("verification_status") == "verified"}
                    incoming_texts = {item.get("plate_text") for item in plan["ocr_annotations"]
                                      if item.get("verification_status") == "verified"}
                    if prior_texts and incoming_texts and prior_texts != incoming_texts:
                        report["conflicts"].append({"sample_id": image.id, "kind": "verified_ocr_text", "existing": sorted(prior_texts), "incoming": sorted(incoming_texts)})
                        if conflict_policy == "error":
                            continue
            else:
                extension = plan["path"].suffix.lower()
                destination = root / "raw" / digest[:2] / f"{digest}{extension}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(plan["path"], destination)
                if sha256_file(destination) != digest:
                    report["errors"].append({"path": plan["relative"].as_posix(), "error": "canonical copy hash mismatch"})
                    continue
                display_name = plan["path"].name
                if display_name in existing_filenames:
                    display_name = f"{plan['path'].stem}__{sample_id[:8]}{extension}"
                existing_filenames.add(display_name)
                image = ImageRecord(id=sample_id, source="registered", original_filename=display_name, sha256=digest,
                                    storage_key=f"datasets/raw/{digest[:2]}/{digest}{extension}", width=plan["width"], height=plan["height"])
                db.add(image)
                db.flush()
                images_by_hash[digest] = image
                report["registered"] += 1
                for row in plan["annotations"]:
                    cx, cy, bw, bh = row["bbox_normalized_xywh"]
                    bounds = [(cx-bw/2)*plan["width"], (cy-bh/2)*plan["height"], (cx+bw/2)*plan["width"], (cy+bh/2)*plan["height"]]
                    db.add(Annotation(image_id=image.id, bbox_xyxy_pixels=bounds, bbox_normalized_xywh=[cx,cy,bw,bh], ai_detection=False))
            for origin in plan["origins"]:
                provenance_key = (image.id, origin["relative"].as_posix())
                if provenance_key not in provenance_keys:
                    db.add(DatasetProvenance(image_id=image.id, source_dataset=source_dataset, import_batch=import_batch,
                                             source_relative_path=origin["relative"].as_posix(), metadata_json={"group_id": origin["group_id"], "source_split": origin["split"],
                                             "task": task, "ocr_annotations": origin["ocr_annotations"],
                                             "source_class_counts": origin["source_class_counts"], "class_mapping": report.get("class_mapping")}))
                    report["provenance_records"] += 1
                    provenance_keys.add(provenance_key)
            assignment = assignment_by_image.get(image.id)
            inherited_split = group_assignments.get(plan["group_id"]) if plan["group_id"] else None
            assigned_split = inherited_split or incoming_group_defaults.get(plan["group_id"]) or plan["split"] or "train"
            if assignment and plan["split"] and assignment.split != plan["split"]:
                report["conflicts"].append({"sample_id": image.id, "existing_split": assignment.split, "incoming_split": plan["split"]})
                if conflict_policy == "error":
                    continue
            elif not assignment:
                new_assignment = DatasetSplitAssignment(image_id=image.id, task=task, split=assigned_split,
                                                        group_id=plan["group_id"], source_dataset=source_dataset)
                db.add(new_assignment)
                assignment_by_image[image.id] = new_assignment
                if plan["group_id"]:
                    group_assignments.setdefault(plan["group_id"], assigned_split)
        unresolved_conflicts = [row for row in report["conflicts"] if not (
            conflict_policy == "prefer-holdout" and row.get("kind") == "duplicate_split" and row.get("resolution") == "prefer-holdout"
        )]
        if report["errors"] or unresolved_conflicts:
            db.rollback()
        else:
            db.commit()
    if not report["errors"] and not unresolved_conflicts:
        report["dry_run"] = False
    return report


def audit_dataset(task: str = "detector") -> dict:
    if task not in TASKS:
        raise ValueError("task must be detector or ocr")
    report = {"task": task, "generated_at": datetime.now(timezone.utc).isoformat(), "total_unique_samples": 0,
              "exact_duplicate_groups": [], "near_duplicates": [], "missing_images": [], "modified_images": [], "missing_annotations": [],
              "invalid_boxes": [], "split_counts": {split: 0 for split in SPLITS},
              "task_counts": {split: 0 for split in SPLITS}, "source_counts": {}, "import_batch_counts": {}, "image_dimensions": {}, "class_distribution": {"license_plate": 0},
              "ocr_character_counts": {}, "ocr_text_lengths": {}, "unverified_samples": 0, "unknown_group_samples": 0,
              "split_leakage": [], "group_split_conflicts": [], "manifest_changes": [], "snapshot_changes": [],
              "warnings": ["No capture/session group metadata is available for some samples; split leakage checks cannot guarantee separation."]}
    with SessionLocal() as db:
        samples = db.scalars(select(ImageRecord).where(ImageRecord.source == "registered").order_by(ImageRecord.id)).all()
        annotations_by_image: dict[str, list[Annotation]] = defaultdict(list)
        for annotation in db.scalars(select(Annotation).order_by(Annotation.image_id, Annotation.id)).all():
            annotations_by_image[annotation.image_id].append(annotation)
        provenance_by_image: dict[str, list[DatasetProvenance]] = defaultdict(list)
        for origin in db.scalars(select(DatasetProvenance).order_by(DatasetProvenance.image_id, DatasetProvenance.id)).all():
            provenance_by_image[origin.image_id].append(origin)
        assignments_by_image = {
            row.image_id: row for row in db.scalars(
                select(DatasetSplitAssignment).where(DatasetSplitAssignment.task == task)
            ).all()
        }
        report["total_unique_samples"] = len({sample.sha256 for sample in samples})
        hashes: dict[str, list[str]] = defaultdict(list)
        group_splits: dict[str, set[str]] = defaultdict(set)
        hash_splits: dict[str, set[str]] = defaultdict(set)
        perceptual_items = []
        dim_counts: Counter[str] = Counter()
        sources: Counter[str] = Counter()
        batches: Counter[str] = Counter()
        for image in samples:
            hashes[image.sha256].append(image.id)
            dim_counts[f"{image.width}x{image.height}"] += 1
            annotations = annotations_by_image.get(image.id, [])
            if image.review_status != "verified" or image.verified_at is None:
                report["unverified_samples"] += 1
            if task == "detector" and image.review_status == "verified" and image.verified_at is not None and not annotations:
                report["missing_annotations"].append(image.id)
            for ann in annotations:
                try:
                    pixels_to_yolo(ann.bbox_xyxy_pixels, image.width, image.height)
                    report["class_distribution"]["license_plate"] += 1
                except ValueError:
                    report["invalid_boxes"].append({"sample_id": image.id, "annotation_id": ann.id})
                text_value = ann.verified_result or ""
                if text_value:
                    report["ocr_text_lengths"][str(len(text_value))] = report["ocr_text_lengths"].get(str(len(text_value)), 0) + 1
                    for char in text_value:
                        report["ocr_character_counts"][char] = report["ocr_character_counts"].get(char, 0) + 1
            try:
                source_path = image_path(image.source, image.original_filename, image.storage_key)
                if sha256_file(source_path) != image.sha256:
                    report["modified_images"].append(image.id)
                perceptual_items.append({"relative": image.id, "sha256": image.sha256, "perceptual_hash": _average_hash(source_path)})
            except FileNotFoundError:
                report["missing_images"].append(image.id)
            prov = provenance_by_image.get(image.id, [])
            for record in prov:
                sources[record.source_dataset] += 1
                batches[record.import_batch] += 1
            assignment = assignments_by_image.get(image.id)
            if assignment:
                report["split_counts"][assignment.split] += 1
                hash_splits[image.sha256].add(assignment.split)
                if assignment.group_id:
                    group_splits[assignment.group_id].add(assignment.split)
                else:
                    report["unknown_group_samples"] += 1
                if task == "detector":
                    eligible = image.review_status == "verified" and image.verified_at is not None and bool(annotations)
                else:
                    eligible = (image.review_status == "verified" and image.verified_at is not None
                                and any(ann.verified_result for ann in annotations)) or any(
                                    item.get("verification_status") == "verified" for record in prov
                                    for item in (record.metadata_json or {}).get("ocr_annotations", []))
                if eligible:
                    report["task_counts"][assignment.split] += 1
        report["exact_duplicate_groups"] = [ids for ids in hashes.values() if len(ids) > 1]
        report["near_duplicates"] = _near_duplicate_pairs(perceptual_items)
        report["split_leakage"] = [{"sha256": digest, "splits": sorted(splits)} for digest, splits in hash_splits.items() if len(splits) > 1]
        report["split_percentages"] = split_percentages(report["split_counts"])
        report["task_split_percentages"] = split_percentages(report["task_counts"])
        report["group_split_conflicts"] = group_split_conflicts(
            [{"group_id": assignment.group_id, "split": assignment.split} for assignment in assignments_by_image.values()]
        )
        report["source_counts"] = dict(sources)
        report["import_batch_counts"] = dict(batches)
        report["image_dimensions"] = dict(dim_counts)
        for manifest in db.scalars(select(DatasetManifest).where(DatasetManifest.task == task)).all():
            try:
                payload = json.loads(Path(manifest.path).read_text(encoding="utf-8"))
                if content_hash(payload) != manifest.manifest_sha256:
                    report["manifest_changes"].append({"id": manifest.id, "path": manifest.path})
            except (OSError, ValueError):
                report["manifest_changes"].append({"id": manifest.id, "path": manifest.path})
        for snapshot in db.scalars(select(DatasetSnapshot).where(DatasetSnapshot.task == task)).all():
            path = Path(snapshot.path)
            valid = path.is_dir()
            if valid:
                from .datasets import json_manifest_sha256, tree_sha256
                try:
                    payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                    valid = (tree_sha256(path) == snapshot.manifest_sha256 or
                             task == "detector" and json_manifest_sha256(payload) == snapshot.manifest_sha256)
                except (OSError, ValueError):
                    valid = False
            if not valid:
                report["snapshot_changes"].append({"id": snapshot.id, "path": snapshot.path})
    return report


def report_path(root: Path, name: str, payload: dict) -> Path:
    directory = root / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Register, audit, split, and snapshot PlateLens datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("import", "merge"):
        imp = subparsers.add_parser(command, help="Import/merge an extracted YOLO detector or OCR CSV dataset")
        imp.add_argument("source", type=Path)
        imp.add_argument("--source-dataset", required=True)
        imp.add_argument("--import-batch")
        imp.add_argument("--task", choices=TASKS, default="detector")
        imp.add_argument("--apply", action="store_true", help="Commit import; default is a dry run")
        imp.add_argument("--conflict-policy", choices=("error", "keep-existing", "prefer-holdout"), default="error")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--task", choices=TASKS, default="detector")
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--task", choices=TASKS, default="detector")
    reassign = subparsers.add_parser("assign")
    reassign.add_argument("sample_id")
    reassign.add_argument("--task", choices=TASKS, required=True)
    reassign.add_argument("--split", choices=SPLITS, required=True)
    reassign.add_argument("--confirm-test-change", action="store_true")
    snapshot = subparsers.add_parser("snapshot", help="Create a copied, hash-verified YOLO detector snapshot")
    snapshot.add_argument("--task", choices=("detector",), default="detector")
    args = parser.parse_args()
    Base.metadata.create_all(bind=engine)
    try:
        if args.command in {"import", "merge"}:
            report = import_dataset(args.source, args.source_dataset, args.task, import_batch=args.import_batch, apply=args.apply, conflict_policy=args.conflict_policy)
        elif args.command == "audit":
            report = audit_dataset(args.task)
        elif args.command == "manifest":
            payload, path = create_manifest(args.task)
            report = {"manifest": str(path), "version": payload["version"], "samples": len(payload["samples"]), "split_counts": dict(Counter(row["split"] for row in payload["samples"]))}
        elif args.command == "assign":
            report = reassign_split(args.task, args.sample_id, args.split, confirm_test_change=args.confirm_test_change)
        else:
            from .datasets import create_detection_snapshot
            row = create_detection_snapshot()
            report = {"snapshot_id": row.id, "task": row.task, "path": row.path, "split_counts": row.split_counts,
                      "snapshot_sha256": row.manifest_sha256}
        saved = report_path(DATASETS_DIR, args.command, report)
        if args.command in {"import", "merge"}:
            summary = {key: value for key, value in report.items() if key not in {
                "exact_duplicates", "filename_collisions", "missing_labels", "invalid_annotations",
                "clipped_annotations", "conflicts", "near_duplicates", "errors"
            }}
            summary.update({
                "exact_duplicate_count": len(report["exact_duplicates"]),
                "filename_collision_count": len(report["filename_collisions"]),
                "missing_label_count": len(report["missing_labels"]),
                "invalid_annotation_count": len(report["invalid_annotations"]),
                "clipped_annotation_count": len(report["clipped_annotations"]),
                "conflict_count": len(report["conflicts"]),
                "conflicts_preview": report["conflicts"][:10],
                "near_duplicate_candidate_count": len(report["near_duplicates"]),
                "error_count": len(report["errors"]),
                "errors_preview": report["errors"][:10],
            })
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"Summary: {report['images_seen']} images scanned, {report['registered']} registered, "
                  f"{len(report['exact_duplicates'])} exact duplicates, {len(report['near_duplicates'])} near-duplicate candidates, "
                  f"{len(report['clipped_annotations'])} clipped boxes, {len(report['errors'])} input errors, dry_run={report['dry_run']}.")
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.command == "audit":
            print(f"Summary: {report['total_unique_samples']} unique images, {len(report['missing_images'])} missing, "
                  f"{len(report['modified_images'])} modified, {len(report['near_duplicates'])} near-duplicate candidates, "
                  f"{len(report['split_leakage'])} split leakage groups.")
        print(f"Report: {saved}")
        fatal_conflicts = any(not (
            getattr(args, "conflict_policy", "error") == "prefer-holdout"
            and row.get("kind") == "duplicate_split"
            and row.get("resolution") == "prefer-holdout"
        ) for row in report.get("conflicts", []))
        return 1 if report.get("errors") or fatal_conflicts else 0
    except Exception as exc:
        print(f"dataset registry: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())

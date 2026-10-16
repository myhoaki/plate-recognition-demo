"""Immutable, hash-verified dataset snapshots built directly from datasets/imports/.

This replaces the previous dataset-registry workflow. There is no manifest
versioning, no provenance table, and no import CLI: the datasets/imports/ folder
is the single source of truth.

Splitting is deterministic and leakage-aware. Frames from the same capture
sequence (for example carlong_0001..carlong_0189) are grouped together by
chunking the numeric part of the filename, so adjacent video frames never land
in different splits.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from .database import SessionLocal
from .models import DatasetSnapshot, ImageRecord, utcnow
from .storage import DATASETS_DIR, sha256_file

# How many consecutive frames of one capture sequence form a single group.
# 50 keeps near-duplicate burst frames together while leaving enough groups
# to build balanced splits.
GROUP_CHUNK = 50

SPLIT_RATIOS = (("train", 0.8), ("val", 0.1), ("test", 0.1))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_NAME_NUMBER = re.compile(r"^(?P<stem>.*?)[-_]?(?P<number>\d+)$")


def json_manifest_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def group_key(relative: Path) -> str:
    """Group adjacent frames of one capture sequence together.

    carlong_0001.png -> carlong_0000
    carlong_0051.png -> carlong_0050
    """
    stem = relative.stem
    match = _NAME_NUMBER.match(stem)
    if match:
        number = int(match.group("number"))
        return f"{match.group('stem') or 'img'}_{number // GROUP_CHUNK:06d}"
    return stem


def initial_split(group_id: str) -> str:
    """Deterministic 80/10/10 split for a group key.

    Kept for the OCR crop splitter in ocr_training.py so both pipelines use the
    same deterministic hash-based assignment.
    """
    return split_for_group(group_id)


def split_for_group(key: str) -> str:
    """Deterministic 80/10/10 split derived from the group key."""
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 1000
    if bucket < 800:
        return "train"
    if bucket < 900:
        return "val"
    return "test"


def read_class_names(imports_root: Path) -> tuple[int, dict[int, str]]:
    """Read nc / names from a dataset.yaml (or data.yaml) anywhere under imports/.

    Datasets in the wild ship this file at the dataset root, which may be a
    subfolder of imports/, so the search is recursive and the shallowest match
    wins.
    """
    candidates: list[Path] = []
    for filename in ("dataset.yaml", "dataset.yml", "data.yaml"):
        candidates.extend(sorted(imports_root.rglob(filename)))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            import yaml

            metadata = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue
        names = metadata.get("names")
        resolved: dict[int, str] = {}
        if isinstance(names, list):
            resolved = {index: str(name) for index, name in enumerate(names)}
        elif isinstance(names, dict):
            try:
                resolved = {int(index): str(name) for index, name in names.items()}
            except (TypeError, ValueError):
                continue
        count = int(metadata.get("nc") or len(resolved) or 1)
        if resolved or count > 1:
            return count, resolved
    return 1, {}


def yolo_boxes_from_label(label_path: Path, class_count: int = 1) -> list[list[float]]:
    """Read a YOLO label file and return bounding boxes [cx, cy, w, h].

    Supports both four-coordinate boxes and segmentation polygons. Polygons are
    reduced to their enclosing axis-aligned bounding box because the detector is
    trained as a detection model, not a segmentation model.

    Every source class is mapped to the single license_plate position class, so a
    dataset that splits plates into subclasses (for example BSD / BSV) still
    trains one detector.
    """
    boxes: list[list[float]] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            continue
        if len(values) < 5:
            continue
        class_id = int(values[0])
        if class_id < 0 or class_id >= class_count:
            continue
        coords = values[1:]
        if len(coords) == 4:
            cx, cy, width, height = coords
            if width <= 0 or height <= 0:
                continue
            boxes.append([cx, cy, width, height])
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs = coords[0::2]
            ys = coords[1::2]
            x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
            y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, width, height])
    return boxes


def label_path_for(image_path: Path, imports_root: Path) -> Path | None:
    """Locate the YOLO sidecar label for an image.

    Handles the standard .../images/<split>/<file>.png -> .../labels/<split>/<file>.txt
    layout and a flat <file>.txt next to the image.
    """
    parts = list(image_path.relative_to(imports_root).parts)
    images_index = next((index for index, part in enumerate(parts) if part.lower() == "images"), None)
    if images_index is not None:
        parts[images_index] = "labels"
        candidate = imports_root.joinpath(*parts).with_suffix(".txt")
        if candidate.is_file():
            return candidate
    for candidate in (
        image_path.with_suffix(".txt"),
        image_path.parent / "labels" / image_path.with_suffix(".txt").name,
    ):
        if candidate.is_file():
            return candidate
    return None


def discover_samples(imports_root: Path, class_count: int = 1) -> tuple[list[dict], list[dict]]:
    """Find every image with a readable YOLO label under datasets/imports/."""
    samples: list[dict] = []
    skipped: list[dict] = []
    for image_path in sorted(imports_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_path_for(image_path, imports_root)
        if label_path is None:
            skipped.append({"path": image_path.relative_to(imports_root).as_posix(), "reason": "no label file"})
            continue
        boxes = yolo_boxes_from_label(label_path, class_count)
        if not boxes:
            skipped.append({"path": image_path.relative_to(imports_root).as_posix(), "reason": "no valid boxes"})
            continue
        samples.append({"image_path": image_path, "relative": image_path.relative_to(imports_root), "boxes": boxes})
    return samples, skipped


def create_detection_snapshot() -> DatasetSnapshot:
    """Build a frozen, hash-verified YOLO snapshot from datasets/imports/."""
    imports_root = (DATASETS_DIR / "imports").resolve()
    if not imports_root.is_dir():
        raise ValueError(f"No datasets/imports directory found at {imports_root}")

    class_count, class_names = read_class_names(imports_root)
    samples, skipped = discover_samples(imports_root, class_count)
    if not samples:
        raise ValueError(
            "No images with readable YOLO labels were found under datasets/imports/. "
            "Expected <name>/images/**.png paired with <name>/labels/**.txt."
        )

    snapshot_id = uuid4().hex
    destination = (DATASETS_DIR / "snapshots" / "detector" / snapshot_id).resolve()
    if destination.exists():
        raise FileExistsError(f"Snapshot path already exists: {destination}")

    # Group-aware split keeps adjacent video frames in the same split.
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        grouped.setdefault(group_key(sample["relative"]), []).append(sample)

    split_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for key in sorted(grouped):
        split_rows[split_for_group(key)].extend(grouped[key])

    split_counts = {split: len(rows) for split, rows in split_rows.items()}
    missing = [split for split, count in split_counts.items() if count == 0]
    if missing:
        raise ValueError(
            f"Split(s) {', '.join(missing)} are empty. Add more capture sequences so "
            "each split receives at least one frame group."
        )

    manifest_rows = []
    try:
        for split, rows in split_rows.items():
            image_dir = destination / "images" / split
            label_dir = destination / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for row in rows:
                source = row["image_path"]
                name = f"{source.stem}{source.suffix.lower()}"
                exported = image_dir / name
                shutil.copy2(source, exported)
                labels = [
                    "0 " + " ".join(f"{value:.8f}" for value in box) for box in row["boxes"]
                ]
                (label_dir / f"{source.stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
                manifest_rows.append(
                    {
                        "sample_id": source.stem,
                        "image_path": f"images/{split}/{name}",
                        "split": split,
                        "group_id": group_key(row["relative"]),
                        "source_relative_path": row["relative"].as_posix(),
                        "sha256": sha256_file(exported),
                        "boxes": row["boxes"],
                    }
                )

        (destination / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: license_plate\n",
            encoding="utf-8",
        )
        payload = {
            "snapshot_id": snapshot_id,
            "task": "detector",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(imports_root),
            "split_counts": split_counts,
            "group_chunk": GROUP_CHUNK,
            "class_mapping": {"0": "license_plate"},
            "source_class_count": class_count,
            "source_class_names": {str(key): value for key, value in sorted(class_names.items())},
            "source_class_collapse": "all source classes map to the single license_plate position class",
            "validation_report": {
                "images": len(manifest_rows),
                "boxes": sum(len(row["boxes"]) for row in manifest_rows),
                "skipped": skipped[:50],
            },
            "samples": manifest_rows,
        }
        (destination / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        record = DatasetSnapshot(
            id=snapshot_id,
            task="detector",
            path=str(destination),
            split_counts=split_counts,
            manifest_sha256=tree_sha256(destination),
            created_at=utcnow(),
        )
        with SessionLocal() as db:
            db.add(record)
            db.commit()
            db.refresh(record)

        for item in destination.rglob("*"):
            if item.is_file():
                item.chmod(0o444)
        for item in sorted(
            (item for item in destination.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            item.chmod(0o555)
        destination.chmod(0o555)
        return record
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

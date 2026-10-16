"""Immutable, traceable dataset snapshots used by training jobs."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from .database import SessionLocal
from .models import DatasetSnapshot, ImageRecord, utcnow
from .storage import DATASETS_DIR, image_path, sha256_file


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


def _split_counts(total: int, ratios=(0.8, 0.1, 0.1)) -> list[int]:
    if total < 3:
        raise ValueError("At least 3 verified samples are required to create train, validation, and test splits.")
    remainder = total - 3
    exact = [remainder * ratio for ratio in ratios]
    counts = [1 + int(value) for value in exact]
    for index in sorted(range(3), key=lambda i: exact[i] - int(exact[i]), reverse=True)[: remainder - sum(int(value) for value in exact)]:
        counts[index] += 1
    return counts


def create_detection_snapshot() -> DatasetSnapshot:
    from .dataset_registry import create_manifest, yolo_label_line

    manifest, _manifest_path = create_manifest("detector", DATASETS_DIR)
    snapshot_id = uuid4().hex
    root = (DATASETS_DIR / "snapshots" / "detector").resolve()
    destination = root / snapshot_id
    if destination.exists():
        raise FileExistsError(f"Snapshot path already exists: {destination}")
    split_counts = {split: sum(sample["split"] == split for sample in manifest["samples"]) for split in ("train", "val", "test")}
    with SessionLocal() as db:
        images = {image.id: image for image in db.scalars(select(ImageRecord)).all()}
        try:
            manifest_rows = []
            for sample in manifest["samples"]:
                image = images.get(sample["sample_id"])
                if not image:
                    raise ValueError(f"Manifest sample {sample['sample_id']} is missing from the image registry")
                source = image_path(image.source, image.original_filename, image.storage_key)
                if sha256_file(source) != sample["image_sha256"]:
                    raise ValueError(f"Source image {sample['sample_id']} changed after manifest creation")
                name = f"{sample['sample_id']}{source.suffix.lower()}"
                split = sample["split"]
                image_dir, label_dir = destination / "images" / split, destination / "labels" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                exported_image = image_dir / name
                shutil.copy2(source, exported_image)
                if sha256_file(exported_image) != sample["image_sha256"]:
                    raise ValueError(f"Snapshot copy verification failed for {sample['sample_id']}")
                boxes = sample["annotations"]
                labels = []
                for box in boxes:
                    x, y, width, height = map(float, box["bbox_normalized_xywh"])
                    labels.append(yolo_label_line(int(box["class_id"]), [x, y, width, height]))
                (label_dir / f"{sample['sample_id']}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                manifest_rows.append({"image_id": sample["sample_id"], "image_path": f"images/{split}/{name}", "width": image.width,
                                      "height": image.height, "split": split, "sha256": sample["image_sha256"], "annotation_sha256": sample["annotation_sha256"],
                                      "source_dataset": sample["source_dataset"], "group_id": sample["group_id"],
                                      "boxes": [{"bbox_normalized_xywh": box["bbox_normalized_xywh"], "text": box["verified_text"]} for box in boxes]})
            if min(split_counts.values()) < 1:
                raise ValueError("Verified detector data must contain train, val, and test samples")
            (destination / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: license_plate\n", encoding="utf-8")
            payload = {"snapshot_id": snapshot_id, "task": "detector", "source_manifest_version": manifest["version"],
                       "source_manifest_sha256": json_manifest_sha256(manifest), "created_at": datetime.now(timezone.utc).isoformat(),
                       "split_counts": split_counts, "class_mapping": {"0": "license_plate"},
                       "validation_report": {"copied_images": len(manifest_rows), "labels": sum(len(row["boxes"]) for row in manifest_rows)}, "samples": manifest_rows}
            (destination / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            (destination / "export_manifest.json").write_text(json.dumps({"snapshot_id": snapshot_id, "samples": [
                {"exported_filename": Path(row["image_path"]).name, "sample_id": row["image_id"], "sha256": row["sha256"]} for row in manifest_rows]}, indent=2), encoding="utf-8")
            record = DatasetSnapshot(id=snapshot_id, task="detector", path=str(destination), split_counts=split_counts,
                                     manifest_sha256=tree_sha256(destination), created_at=utcnow())
            db.add(record)
            db.commit()
            db.refresh(record)
            for item in destination.rglob("*"):
                if item.is_file():
                    item.chmod(0o444)
            for item in sorted((path for path in destination.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
                item.chmod(0o555)
            destination.chmod(0o555)
            return record
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise

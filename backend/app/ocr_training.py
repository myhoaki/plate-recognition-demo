"""Server-side fine-tuning of the lightweight plate OCR network."""

from __future__ import annotations

import math
import os
import shutil
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from sqlalchemy import select
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .database import SessionLocal
from .models import Annotation, DatasetManifest, DatasetProvenance, DatasetSnapshot, DatasetSplitAssignment, ImageRecord, TrainingJob
from .ocr_model import (
    OCR_HEIGHT,
    OCR_WIDTH,
    PLATE_ALPHABET,
    PlateOCRNetwork,
    clean_plate_text,
    decode_greedy,
    prepare_plate_crop,
)
from .storage import DATASETS_DIR, image_path, sha256_file
from .training import _record_epoch_metrics, _seed_checkpoint, _split_sizes
from .datasets import tree_sha256
from .dataset_registry import initial_split


class PlateCropDataset(Dataset):
    def __init__(self, examples: list[tuple[Path, str]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        path, text = self.examples[index]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read OCR crop: {path.name}")
        return torch.from_numpy(prepare_plate_crop(image)), torch.tensor(
            [PLATE_ALPHABET.index(char) for char in text], dtype=torch.long
        )


def _collate(batch):
    images, labels = zip(*batch)
    lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return torch.stack(images), torch.cat(labels), lengths


def _cancelled(job_id: str) -> bool:
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        return not job or job.status == "cancelled" or job.cancel_requested


def _save_job(job_id: str, **values):
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        if job:
            for key, value in values.items():
                setattr(job, key, value)
            db.commit()


def _current_job_metrics(job_id: str) -> dict:
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        return dict(job.metrics or {}) if job else {}


def _collect_crops(job_id: str, training_dir: Path, snapshot_id: str | None = None, output_dir: Path | None = None) -> tuple[dict[str, list[tuple[Path, str]]], dict[str, int]]:
    raw_root = Path(os.environ.get("RAW_OCR_DATASET_DIR", training_dir / "raw-dataset-ocr")).resolve()
    if snapshot_id:
        with SessionLocal() as db:
            snapshot = db.get(DatasetSnapshot, snapshot_id)
            if not snapshot or snapshot.task != "ocr":
                raise ValueError("OCR dataset snapshot is not registered or has an invalid path.")
            snapshot_dir = Path(snapshot.path).resolve()
            allowed = ((DATASETS_DIR / "snapshots" / "ocr").resolve(),
                       (Path(os.environ.get("VERIFIED_OCR_DATASET_DIR", training_dir / "verified-dataset-ocr")).resolve() / "jobs").resolve())
            try:
                snapshot_dir.relative_to(next(root for root in allowed if snapshot_dir == root or root in snapshot_dir.parents))
            except (ValueError, StopIteration):
                raise ValueError("OCR dataset snapshot path is outside the configured dataset directory.") from None
            if not (snapshot_dir / "manifest.json").is_file():
                raise ValueError("OCR dataset snapshot files are missing.")
            if tree_sha256(snapshot_dir) != snapshot.manifest_sha256:
                raise ValueError("OCR dataset snapshot changed after it was created. Create a new snapshot from verified data.")
        job_dir = training_dir / "jobs" / job_id / "ocr-dataset"
        job_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot_dir, job_dir)
        result: dict[str, list[tuple[Path, str]]] = {}
        for split in ("train", "val", "test"):
            rows = []
            manifest = job_dir / f"{split}.csv"
            with manifest.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    path = (job_dir / row["image_path"]).resolve()
                    path.relative_to(job_dir.resolve())
                    if not path.is_file() or not row.get("plate_text", "").strip():
                        raise ValueError(f"OCR snapshot contains an invalid {split} sample.")
                    rows.append((path, row["plate_text"]))
            result[split] = rows
        return result, {split: len(rows) for split, rows in result.items()}
    with SessionLocal() as db:
        rows = db.execute(
            select(Annotation, ImageRecord)
            .join(ImageRecord, ImageRecord.id == Annotation.image_id)
            .where(
                ImageRecord.review_status == "verified",
                ImageRecord.verified_at.is_not(None),
                Annotation.verified_result.is_not(None),
            )
            .order_by(ImageRecord.id, Annotation.id)
        ).all()
        imported_rows = db.execute(
            select(DatasetProvenance, ImageRecord)
            .join(ImageRecord, ImageRecord.id == DatasetProvenance.image_id)
            .order_by(ImageRecord.id, DatasetProvenance.id)
        ).all()
        group_metadata = {row.image_id: row.group_id for row in db.scalars(select(DatasetSplitAssignment).where(DatasetSplitAssignment.task == "ocr")).all() if row.group_id}
        for provenance, image_record in imported_rows:
            group_metadata.setdefault(image_record.id, (provenance.metadata_json or {}).get("group_id"))

    raw_job_dir = DATASETS_DIR / "imports" / "ocr-preparation" / job_id
    verified_job_dir = output_dir or (DATASETS_DIR / "snapshots" / "ocr" / job_id)
    if raw_job_dir.exists() or verified_job_dir.exists():
        raise FileExistsError("OCR preparation destination already exists; refusing to overwrite it")
    crops_dir = raw_job_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    raw_manifest = raw_job_dir / "annotations.csv"
    raw_rows = []
    verified_examples: list[tuple[Path, str, int, str, str]] = []
    def add_raw_example(source: Path, text: str, bbox: tuple[float, float, float, float] | None, index: int, group_id: str | None = None):
        try:
            source = source.resolve(strict=True)
            source.relative_to(raw_root)
        except (OSError, ValueError):
            return
        image = cv2.imread(str(source))
        if image is None:
            return
        if bbox is not None:
            x1, y1, x2, y2 = (int(round(value)) for value in bbox)
            height, width = image.shape[:2]
            x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
            y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                return
            image = image[y1:y2, x1:x2]
        label = clean_plate_text(text)
        if label != text:
            raise ValueError(f"OCR text {text!r} contains characters unsupported by the current CRNN alphabet; source text was preserved")
        if not label or len(label) > 12:
            return
        crop_path = crops_dir / f"external_{index:06d}.png"
        if not cv2.imwrite(str(crop_path), image):
            return
        raw_rows.append({
            "image_path": str(source.relative_to(raw_root)),
            "crop_path": f"jobs/{job_id}/crops/{crop_path.name}",
            "crop_coordinates": list(bbox) if bbox is not None else None,
            "source_image_id": str(source.relative_to(raw_root)),
            "review_status": "verified",
            "verified_text": label,
        })
        source_id = str(source.relative_to(raw_root))
        verified_examples.append((crop_path, label, index, source_id, group_id or source_id))

    # Imported OCR datasets use a verified transcription manifest. The matching
    # YOLO sidecar supplies plate bounds; it is never treated as OCR text.
    annotations_csv = raw_root / "annotations.csv"
    if annotations_csv.is_file():
        with annotations_csv.open(newline="", encoding="utf-8-sig") as manifest_file:
            for index, row in enumerate(csv.DictReader(manifest_file), 1):
                verified = str(row.get("verified", "")).strip().lower() in {"1", "true", "yes", "verified"}
                review_status = str(row.get("review_status", "")).strip().lower()
                if not verified or review_status == "rejected":
                    continue
                relative = Path(row.get("image_path", ""))
                if relative.is_absolute() or ".." in relative.parts:
                    continue
                source = (raw_root / relative).resolve()
                bbox = None
                coords = [row.get(key, "").strip() for key in ("x1", "y1", "x2", "y2")]
                if all(coords):
                    try:
                        bbox = tuple(float(value) for value in coords)
                    except ValueError:
                        continue
                elif any(coords):
                    continue
                else:
                    try:
                        parts = list(source.relative_to(raw_root).parts)
                        image_index = next(i for i, part in enumerate(parts) if part.lower() == "images")
                        parts[image_index] = "labels"
                        label_path = raw_root.joinpath(*parts).with_suffix(".txt")
                        label_lines = label_path.read_text(encoding="utf-8").splitlines()
                    except (OSError, ValueError):
                        label_lines = []
                    objects = []
                    image = cv2.imread(str(source))
                    if image is None:
                        continue
                    image_height, image_width = image.shape[:2]
                    try:
                        object_index = int(row.get("object_index") or 0)
                    except ValueError:
                        continue
                    for line in label_lines:
                        try:
                            values = [float(value) for value in line.split()]
                        except ValueError:
                            continue
                        coords_yolo = values[1:]
                        if len(coords_yolo) == 4:
                            center_x, center_y, box_width, box_height = coords_yolo
                            box = ((center_x-box_width/2)*image_width, (center_y-box_height/2)*image_height,
                                   (center_x+box_width/2)*image_width, (center_y+box_height/2)*image_height)
                        elif len(coords_yolo) >= 6 and len(coords_yolo) % 2 == 0:
                            xs = coords_yolo[0::2]
                            ys = coords_yolo[1::2]
                            box = (min(xs)*image_width, min(ys)*image_height,
                                   max(xs)*image_width, max(ys)*image_height)
                        else:
                            continue
                        objects.append(box)
                    if objects:
                        if object_index < 0 or object_index >= len(objects):
                            continue
                        bbox = objects[object_index]
                    elif str(row.get("is_crop", "")).strip().lower() not in {"1", "true", "yes"}:
                        continue
                add_raw_example(source, row.get("plate_text", ""), bbox, index, row.get("group_id") or None)

    for annotation, image_record in rows:
        try:
            source = image_path(image_record.source, image_record.original_filename, image_record.storage_key)
        except FileNotFoundError:
            continue
        image = cv2.imread(str(source))
        if image is None:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in annotation.bbox_xyxy_pixels)
        height, width = image.shape[:2]
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        crop_path = crops_dir / f"{image_record.id}_{annotation.id}.png"
        if cv2.imwrite(str(crop_path), crop):
            verified_text = clean_plate_text(annotation.verified_result or "")
            raw_rows.append({
                "image_path": f"crops/{crop_path.name}",
                "source_image_id": image_record.id,
                "source_annotation_id": annotation.id,
                "crop_coordinates": [x1, y1, x2, y2],
                "image_id": image_record.id,
                "annotation_id": annotation.id,
                "review_status": image_record.review_status,
                "ai_text": annotation.ai_result or "",
                "verified_text": annotation.verified_result or "",
            })
            if (
                image_record.review_status == "verified"
                and image_record.verified_at is not None
                and verified_text
                and len(verified_text) <= 12
            ):
                verified_examples.append((crop_path, verified_text, annotation.id, image_record.id,
                                          group_metadata.get(image_record.id) or image_record.id))

    imported_index = 0
    for provenance, image_record in imported_rows:
        for imported in (provenance.metadata_json or {}).get("ocr_annotations", []):
            if imported.get("verification_status") != "verified":
                continue
            try:
                source = image_path(image_record.source, image_record.original_filename, image_record.storage_key)
            except FileNotFoundError:
                continue
            image = cv2.imread(str(source))
            box = imported.get("bbox_xyxy_pixels")
            if image is None or not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = (int(round(float(value))) for value in box)
            height, width = image.shape[:2]
            x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
            y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"OCR crop coordinates are invalid for sample {image_record.id}")
            exact_text = imported.get("plate_text", "")
            train_text = clean_plate_text(exact_text)
            if not exact_text or train_text != exact_text:
                raise ValueError(f"OCR text {exact_text!r} for sample {image_record.id} is outside the current CRNN alphabet; the verified text remains unchanged")
            if len(train_text) > 12:
                raise ValueError(f"OCR text for sample {image_record.id} exceeds the current CRNN limit of 12 characters")
            imported_index += 1
            crop_path = crops_dir / f"registered_{image_record.id}_{imported_index:04d}.png"
            if not cv2.imwrite(str(crop_path), image[y1:y2, x1:x2]):
                raise ValueError(f"Could not prepare OCR crop for sample {image_record.id}")
            raw_rows.append({"image_path": image_record.storage_key, "crop_path": f"jobs/{job_id}/crops/{crop_path.name}",
                             "source_image_id": image_record.id, "source_annotation_id": provenance.id,
                             "crop_coordinates": [x1, y1, x2, y2], "review_status": "verified", "verified_text": exact_text})
            verified_examples.append((crop_path, exact_text, imported_index, image_record.id,
                                      group_metadata.get(image_record.id) or image_record.id))

    with raw_manifest.open("w", newline="", encoding="utf-8") as manifest_file:
        columns = ["image_path", "crop_path", "source_image_id", "source_annotation_id", "crop_coordinates", "image_id", "annotation_id", "review_status", "ai_text", "verified_text"]
        writer = csv.DictWriter(manifest_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(raw_rows)

    if len(verified_examples) < 30:
        raise ValueError(
            f"OCR training needs at least 30 verified plate crops with readable text in raw-dataset-ocr; found {len(verified_examples)}. "
            f"Add raw-dataset-ocr/annotations.csv rows with image_path, plate_text, verified=true, and object_index "
            f"(or x1,y1,x2,y2). The imported YOLO region labels locate plates but do not contain their text. "
            f"Job crops were saved to {raw_job_dir}."
        )
    grouped: dict[str, list[tuple[Path, str, int, str, str]]] = {}
    for example in verified_examples:
        grouped.setdefault(example[4], []).append(example)
    if len(grouped) < 3:
        raise ValueError("OCR splits require at least 3 distinct source images so crops from one image cannot leak across splits.")
    groups = list(grouped.items())
    split_rows = {"train": [], "val": [], "test": []}
    source_splits = {}
    with SessionLocal() as db:
        current_assignments = {row.image_id: row for row in db.scalars(select(DatasetSplitAssignment).where(DatasetSplitAssignment.task == "ocr")).all()}
        has_manifest = db.scalar(select(DatasetManifest.id).where(DatasetManifest.task == "ocr").limit(1)) is not None
        previous_snapshot = db.scalar(select(DatasetSnapshot).where(DatasetSnapshot.task == "ocr").order_by(DatasetSnapshot.created_at.desc()).limit(1))
        previous_splits = {}
        if previous_snapshot:
            previous_dir = Path(previous_snapshot.path)
            if not previous_dir.is_dir() or tree_sha256(previous_dir) != previous_snapshot.manifest_sha256:
                raise ValueError("Previous OCR snapshot changed; refusing to derive new split assignments from it")
            previous_manifest = __import__("json").loads((previous_dir / "manifest.json").read_text(encoding="utf-8"))
            previous_splits = previous_manifest.get("split_assignments", {})
        group_assignment_splits = {row.group_id: row.split for row in current_assignments.values() if row.group_id}
        for group_id, samples in groups:
            sample_ids = {sample[3] for sample in samples}
            assigned = {current_assignments[sample_id].split for sample_id in sample_ids if sample_id in current_assignments}
            if len(assigned) > 1:
                raise ValueError(f"OCR group {group_id!r} already spans multiple splits")
            source_splits_for_group = set()
            for sample in samples:
                source_parts = {part.lower() for part in Path(sample[3]).parts}
                if source_parts & {"train", "training"}:
                    source_splits_for_group.add("train")
                if source_parts & {"val", "valid", "validation"}:
                    source_splits_for_group.add("val")
                if source_parts & {"test", "testing"}:
                    source_splits_for_group.add("test")
            if len(source_splits_for_group) > 1:
                raise ValueError(f"OCR group {group_id!r} appears in multiple source splits")
            split = (next(iter(assigned)) if assigned else group_assignment_splits.get(group_id)
                     or previous_splits.get(group_id)
                     or (next(iter(source_splits_for_group)) if source_splits_for_group else None)
                     or ("train" if has_manifest or previous_snapshot else initial_split(group_id)))
            source_splits[group_id] = split
            split_rows[split].extend(samples)
            for sample_id in sample_ids:
                image_record = db.get(ImageRecord, sample_id)
                assignment = current_assignments.get(sample_id)
                if image_record and not assignment:
                    db.add(DatasetSplitAssignment(image_id=sample_id, task="ocr", split=split, group_id=group_id, source_dataset="verified-annotations"))
                elif assignment:
                    assignment.group_id = group_id
                    assignment.split = split
        db.commit()
    splits: dict[str, list[tuple[Path, str]]] = {}
    manifest_rows = []
    for split, rows_for_split in split_rows.items():
        split_dir = verified_job_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        splits[split] = []
        with (verified_job_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as manifest_file:
            writer = csv.writer(manifest_file)
            writer.writerow(["crop_id", "image_path", "plate_text", "source_image_id", "source_dataset", "split", "verification_status"])
            for source_path, label, annotation_id, image_id, group_id in rows_for_split:
                destination = split_dir / source_path.name
                shutil.copy2(source_path, destination)
                source_id = str(image_id)
                writer.writerow([destination.stem, f"{split}/{destination.name}", label, source_id, "verified-annotations", split, "verified"])
                manifest_rows.append({"crop_id": destination.stem, "image_path": f"{split}/{destination.name}", "source_image_id": source_id,
                                      "source_dataset": "registered" if len(source_id) == 32 else "raw-dataset-ocr",
                                      "group_id": group_id, "split": split, "verification_status": "verified",
                                      "exact_text_sha256": hashlib.sha256(label.encode("utf-8")).hexdigest(),
                                      "crop_sha256": sha256_file(destination),
                                      "annotation_version": hashlib.sha256(f"{source_id}\0{label}".encode("utf-8")).hexdigest()})
                splits[split].append((destination, label))
    counts = {name: len(rows) for name, rows in splits.items()}
    with SessionLocal() as db:
        source_manifest = db.scalar(select(DatasetManifest).where(DatasetManifest.task == "ocr").order_by(DatasetManifest.version.desc()).limit(1))
    (verified_job_dir / "manifest.json").write_text(
        __import__("json").dumps({"snapshot_id": job_id, "task": "ocr", "split_counts": counts,
                                  "source_manifest_version": source_manifest.version if source_manifest else None,
                                  "split_assignments": source_splits, "samples": manifest_rows}, indent=2),
        encoding="utf-8",
    )
    return splits, {
        "train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"])
    }


def _edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for column, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (left != right)))
        previous = current
    return previous[-1]


def _evaluate(model, loader, device, progress=None) -> dict:
    model.eval()
    exact = 0
    errors = 0
    characters = 0
    processed = 0
    with torch.inference_mode():
        for images, targets, lengths in loader:
            images = images.to(device)
            logits = model(images)
            offset = 0
            for batch_index, length in enumerate(lengths.tolist()):
                target = "".join(PLATE_ALPHABET[index] for index in targets[offset : offset + length].tolist())
                offset += length
                prediction, _ = decode_greedy(logits[batch_index])
                exact += prediction == target
                errors += _edit_distance(prediction, target)
                characters += len(target)
                processed += 1
            if progress:
                progress(processed)
    return {
        "plate_accuracy": exact / max(1, processed),
        "character_error_rate": errors / max(1, characters),
        "evaluated_plates": processed,
    }


def run_ocr_training(job_id: str, config: dict):
    training_dir = Path(os.environ.get("TRAINING_DIR", "/training"))
    try:
        splits, split_counts = _collect_crops(job_id, training_dir, config.get("dataset_snapshot_id"))
        _save_job(
            job_id,
            split_counts=split_counts,
            current_val=0,
            total_val=split_counts["val"],
            current_test=0,
            total_test=split_counts["test"],
        )
        train_rows, val_rows, test_rows = splits["train"], splits["val"], splits["test"]
        batch_size = int(config["batch_size"])
        train_loader = DataLoader(PlateCropDataset(train_rows), batch_size=batch_size, shuffle=True, collate_fn=_collate)
        val_loader = DataLoader(PlateCropDataset(val_rows), batch_size=batch_size, shuffle=False, collate_fn=_collate)
        test_loader = DataLoader(PlateCropDataset(test_rows), batch_size=batch_size, shuffle=False, collate_fn=_collate)
        configured_device = config.get("device", "auto")
        if configured_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif configured_device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{configured_device}" if torch.cuda.is_available() else "cpu")

        global_best = training_dir / "best_ocr.pt"
        _seed_checkpoint("ocr", global_best)
        model = PlateOCRNetwork().to(device)
        if global_best.is_file():
            checkpoint = torch.load(global_best, map_location=device, weights_only=False)
            if checkpoint.get("alphabet") != PLATE_ALPHABET:
                raise ValueError("Saved OCR checkpoint uses a different character alphabet.")
            model.load_state_dict(checkpoint["state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_function = nn.CTCLoss(blank=len(PLATE_ALPHABET), zero_infinity=True)
        total_batches = max(1, math.ceil(len(train_rows) / batch_size))
        _save_job(job_id, total_batches=total_batches, current_batch=0)
        job_dir = training_dir / "jobs" / job_id
        best_model_path = job_dir / "best.pt"
        best_accuracy = -1.0
        metrics: dict[str, float] = {}

        for epoch in range(int(config["epochs"])):
            model.train()
            _save_job(job_id, current_epoch=epoch + 1, current_batch=0, current_val=0)
            loss_total = 0.0
            batch_count = 0
            for batch_index, (images, targets, target_lengths) in enumerate(train_loader, 1):
                if _cancelled(job_id):
                    _save_job(job_id, status="cancelled", finished_at=datetime.now(timezone.utc))
                    return
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images).log_softmax(dim=2).transpose(0, 1)
                input_lengths = torch.full((len(images),), logits.shape[0], dtype=torch.long)
                loss = loss_function(logits, targets, input_lengths, target_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_total += float(loss.item())
                batch_count += 1
                _save_job(job_id, current_batch=batch_index, total_batches=total_batches)

            def update_val_progress(completed):
                _save_job(job_id, current_val=min(completed, split_counts["val"]))

            validation = _evaluate(model, val_loader, device, update_val_progress)
            metrics = {
                "train_loss": loss_total / max(1, batch_count),
                "val_plate_accuracy": validation["plate_accuracy"],
                "val_character_error_rate": validation["character_error_rate"],
            }
            _record_epoch_metrics(job_id, epoch + 1, metrics, validation["plate_accuracy"])
            _save_job(job_id, current_val=split_counts["val"])
            if validation["plate_accuracy"] >= best_accuracy:
                best_accuracy = validation["plate_accuracy"]
                torch.save(
                    {"state_dict": model.cpu().state_dict(), "alphabet": PLATE_ALPHABET,
                     "image_height": OCR_HEIGHT, "image_width": OCR_WIDTH,
                     "epoch": epoch + 1, "validation_metrics": metrics},
                    best_model_path,
                )
                model.to(device)

        if _cancelled(job_id):
            _save_job(job_id, status="cancelled", finished_at=datetime.now(timezone.utc))
            return
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

        def update_test_progress(completed):
            _save_job(job_id, current_test=min(completed, split_counts["test"]))

        test_metrics = _evaluate(model, test_loader, device, update_test_progress)
        metrics.update({
            "test_plate_accuracy": test_metrics["plate_accuracy"],
            "test_character_error_rate": test_metrics["character_error_rate"],
        })
        _save_job(
            job_id,
            status="completed",
            current_epoch=int(config["epochs"]),
            current_batch=total_batches,
            total_batches=total_batches,
            current_test=split_counts["test"],
            metrics={
                **metrics,
                **{key: value for key, value in _current_job_metrics(job_id).items() if key in {"epoch_history", "best_epoch", "best_score", "best_metrics"}},
            },
            model_path=str(best_model_path),
            finished_at=datetime.now(timezone.utc),
        )
        shutil.copy2(best_model_path, global_best)
    except Exception as exc:
        _save_job(job_id, status="failed", error=str(exc)[:2000], finished_at=datetime.now(timezone.utc))

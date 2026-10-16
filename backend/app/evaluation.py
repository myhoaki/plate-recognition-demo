"""Background evaluation of the end-to-end pipeline on a frozen detector test split."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from .pipeline import configure_ultralytics_paths, run_detection
from .celery_app import celery_app
from .database import SessionLocal
from .logic import iou
from .models import DatasetSnapshot, EvaluationJob, ImageRecord, TrainingJob
from .datasets import json_manifest_sha256, tree_sha256
from .storage import DATASETS_DIR, sha256_file


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for column, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (left != right)))
        previous = current
    return previous[-1]


def _checkpoint(job_id: str | None, model_type: str, training_dir: Path) -> str | None:
    if not job_id:
        return None
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        if not job or job.status != "completed" or (job.config or {}).get("model_type", "detector") != model_type or not job.model_path:
            raise ValueError(f"The selected {model_type} checkpoint is unavailable.")
        path = Path(job.model_path).resolve()
        try:
            path.relative_to((training_dir / "jobs" / job.id).resolve())
        except ValueError:
            raise ValueError("A selected model path is invalid.") from None
        if not path.is_file():
            raise ValueError(f"The selected {model_type} checkpoint file is missing.")
        return str(path)


def _save(job_id: str, **values):
    with SessionLocal() as db:
        job = db.get(EvaluationJob, job_id)
        if job:
            for name, value in values.items():
                setattr(job, name, value)
            db.commit()


@celery_app.task(name="platelens.run_evaluation_job")
def run_evaluation_job(job_id: str):
    training_dir = Path(os.environ.get("TRAINING_DIR", "/training")).resolve()
    try:
        with SessionLocal() as db:
            job = db.get(EvaluationJob, job_id)
            if not job or job.status != "queued":
                return
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            config = dict(job.config)
            snapshot = db.get(DatasetSnapshot, config["dataset_snapshot_id"])
            if not snapshot or snapshot.task != "detector":
                raise ValueError("The fixed detector test snapshot is unavailable.")
            snapshot_dir = Path(snapshot.path).resolve()
            allowed_roots = ((DATASETS_DIR / "snapshots" / "detector").resolve(),
                             (Path(os.environ.get("VERIFIED_DETECTOR_DATASET_DIR", "/verified-dataset-detector")).resolve() / "snapshots").resolve())
            if not any(_within(snapshot_dir, root) for root in allowed_roots):
                raise ValueError("The selected dataset snapshot path is invalid.") from None
            manifest_path = snapshot_dir / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError("The selected dataset snapshot manifest is missing.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if json_manifest_sha256(manifest) != snapshot.manifest_sha256 and tree_sha256(snapshot_dir) != snapshot.manifest_sha256:
                raise ValueError("The selected detector test snapshot manifest changed after creation.")
            samples = [row for row in manifest["samples"] if row["split"] == "test"]
            if not samples:
                raise ValueError("The selected test split is empty.")
            job.total = len(samples)
            db.commit()
        detector_path = _checkpoint(config.get("detector_job_id"), "detector", training_dir)
        ocr_path = _checkpoint(config.get("ocr_job_id"), "ocr", training_dir)
        if detector_path:
            configure_ultralytics_paths(training_dir)
        true_positive = false_positive = false_negative = exact = char_errors = char_count = ocr_count = 0
        end_to_end_correct = end_to_end_total = 0
        examples = []
        for index, sample in enumerate(samples, 1):
            relative = Path(sample["image_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("The test snapshot contains an unsafe sample path.")
            image_path = (snapshot_dir / relative).resolve()
            try:
                image_path.relative_to(snapshot_dir)
            except ValueError:
                raise ValueError("The test snapshot contains an unsafe sample path.") from None
            if not image_path.is_file():
                raise ValueError(f"Test image {relative.name} is missing from the snapshot.")
            if sha256_file(image_path) != sample["sha256"]:
                raise ValueError(f"Test image {relative.name} changed after the snapshot was created.")
            truth = []
            for box in sample.get("boxes", []):
                x, y, w, h = box["bbox_normalized_xywh"]
                truth.append(([x-w/2, y-h/2, x+w/2, y+h/2], str(box.get("text", "")).strip().upper()))
            try:
                predictions = run_detection(image_path, detector_path, ocr_path)
            except Exception as exc:
                raise RuntimeError(f"Recognition failed on test image {relative.name}: {exc}") from exc
            candidates = []
            for gt_index, (gt_box, gt_text) in enumerate(truth):
                for pred_index, pred in enumerate(predictions):
                    x1, y1, x2, y2 = pred["bbox_xyxy_pixels"]
                    pred_box = [x1/sample["width"], y1/sample["height"], x2/sample["width"], y2/sample["height"]]
                    candidates.append((iou(gt_box, pred_box), gt_index, pred_index))
            matched_gt, matched_pred = set(), set()
            matches = []
            for score, gt_index, pred_index in sorted(candidates, reverse=True):
                if score >= 0.5 and gt_index not in matched_gt and pred_index not in matched_pred:
                    matched_gt.add(gt_index); matched_pred.add(pred_index); matches.append((gt_index, pred_index))
            tp = len(matches)
            fp, fn = len(predictions) - tp, len(truth) - tp
            true_positive += tp; false_positive += fp; false_negative += fn
            is_correct = True
            for gt_index, pred_index in matches:
                expected = truth[gt_index][1]
                predicted = str(predictions[pred_index].get("text", "")).strip().upper()
                if expected:
                    exact += predicted == expected
                    ocr_count += 1
                    char_errors += _edit_distance(predicted, expected)
                    char_count += len(expected)
                    is_correct = is_correct and predicted == expected
            is_correct = is_correct and fn == 0 and fp == 0 and all(text for _, text in truth)
            if truth:
                end_to_end_total += 1
                end_to_end_correct += is_correct
            if not is_correct and len(examples) < 500:
                examples.append({"image_id": sample["image_id"], "filename": relative.name, "missed_plates": fn, "false_detections": fp,
                                 "ground_truth": [text for _, text in truth], "predictions": [{"text": item.get("text", ""), "bbox_xyxy_pixels": item["bbox_xyxy_pixels"], "detection_confidence": item.get("detection_confidence"), "ocr_confidence": item.get("ocr_confidence")} for item in predictions]})
            _save(job_id, processed=index)
        ground_truth_plates = true_positive + false_negative
        map_metrics = {"detection_map50": None, "detection_map50_95": None}
        if detector_path:
            try:
                from ultralytics import YOLO
                configure_ultralytics_paths(training_dir)
                validation = YOLO(detector_path).val(data=str(snapshot_dir / "data.yaml"), split="test", imgsz=640, verbose=False,
                                                    project=str(training_dir / "evaluation_runs"), name=job_id, exist_ok=True)
                map_metrics = {"detection_map50": float(validation.box.map50), "detection_map50_95": float(validation.box.map)}
            except Exception as exc:
                map_metrics["detection_map_error"] = str(exc)[:300]
        results = {
            "matching_policy": "Greedy one-to-one matching by descending IoU at 0.50; plate text is trimmed and compared case-insensitively, with all other characters preserved.",
            "detection_precision": true_positive / max(1, true_positive + false_positive),
            "detection_recall": true_positive / max(1, ground_truth_plates), **map_metrics,
            "true_positives": true_positive, "false_detections": false_positive, "missed_plates": false_negative,
            "ocr_exact_accuracy": exact / max(1, ocr_count), "ocr_evaluated_plates": ocr_count,
            "character_error_rate": char_errors / max(1, char_count),
            "end_to_end_accuracy": end_to_end_correct / max(1, end_to_end_total),
            "end_to_end_evaluated_images": end_to_end_total,
            "incorrect_examples": examples,
        }
        _save(job_id, results=results, status="completed", finished_at=datetime.now(timezone.utc))
    except Exception as exc:
        _save(job_id, status="failed", error=str(exc)[:2000], finished_at=datetime.now(timezone.utc))

import math
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from .celery_app import celery_app
from .database import SessionLocal
from .models import Annotation, DatasetSnapshot, ImageRecord, TrainingJob
from .storage import DATASETS_DIR, image_path, sha256_file
from .datasets import json_manifest_sha256, tree_sha256


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _split_sizes(total: int, ratios: list[float]) -> list[int]:
    sizes = [1, 1, 1]
    remaining = total - 3
    exact = [remaining * ratio for ratio in ratios]
    additions = [int(value) for value in exact]
    for index, count in enumerate(additions):
        sizes[index] += count
    leftover = remaining - sum(additions)
    order = sorted(range(3), key=lambda index: exact[index] - additions[index], reverse=True)
    for index in order[:leftover]:
        sizes[index] += 1
    return sizes


def _prepare_dataset(job_id: str, config: dict, training_dir: Path) -> tuple[Path, dict]:
    if config.get("dataset_snapshot_id"):
        with SessionLocal() as db:
            snapshot = db.get(DatasetSnapshot, config["dataset_snapshot_id"])
            if not snapshot or snapshot.task != "detector":
                raise ValueError("Selected detector dataset snapshot is missing or has the wrong task type.")
            dataset_dir = Path(snapshot.path).resolve()
            allowed_roots = ((DATASETS_DIR / "snapshots" / "detector").resolve(),
                             (Path(os.environ.get("VERIFIED_DETECTOR_DATASET_DIR", "/verified-dataset-detector")).resolve() / "snapshots").resolve())
            if not any(_is_relative_to(dataset_dir, allowed) for allowed in allowed_roots):
                raise ValueError("Detector dataset snapshot path is outside the configured dataset directory.") from None
            yaml_path = dataset_dir / "data.yaml"
            if not yaml_path.is_file():
                raise ValueError(
                    f"Selected detector dataset snapshot is missing data.yaml ({dataset_dir}). "
                    "The snapshot may still be building or was deleted. Rebuild it and try again."
                )
            manifest_path = dataset_dir / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError("Selected detector dataset snapshot is missing its manifest.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if json_manifest_sha256(manifest) != snapshot.manifest_sha256 and tree_sha256(dataset_dir) != snapshot.manifest_sha256:
                raise ValueError("Detector dataset snapshot manifest changed after it was created.")
            for sample in manifest.get("samples", []):
                sample_path = (dataset_dir / sample["image_path"]).resolve()
                try:
                    sample_path.relative_to(dataset_dir)
                except ValueError:
                    raise ValueError("Detector snapshot contains an unsafe image path.") from None
                if not sample_path.is_file() or sha256_file(sample_path) != sample["sha256"]:
                    raise ValueError("A detector snapshot image changed or is missing. Create a new snapshot.")
            return yaml_path, dict(snapshot.split_counts)

    with SessionLocal() as db:
        images = db.scalars(
            select(ImageRecord)
            .where(ImageRecord.review_status == "verified", ImageRecord.verified_at.is_not(None))
            .order_by(ImageRecord.id)
        ).all()
        records = []
        for image in images:
            boxes = db.scalars(
                select(Annotation).where(Annotation.image_id == image.id)
            ).all()
            try:
                source = image_path(image.source, image.original_filename, image.storage_key)
            except FileNotFoundError:
                continue
            records.append((image.id, source, boxes, image.width, image.height))

    if len(records) < 3:
        raise ValueError("Verify at least 3 images with available image files before training.")
    random.Random(42).shuffle(records)
    counts = _split_sizes(len(records), [config["train_ratio"], config["validation_ratio"], config["test_ratio"]])
    verified_root = Path(
        os.environ.get("VERIFIED_DETECTOR_DATASET_DIR", training_dir / "verified-dataset-detector")
    )
    job_dir = verified_root / "jobs" / job_id
    dataset_dir = job_dir / "dataset"
    shutil.rmtree(dataset_dir, ignore_errors=True)
    names = ("train", "val", "test")
    cursor = 0
    for split, count in zip(names, counts):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_id, source, boxes, image_width, image_height in records[cursor : cursor + count]:
            destination_name = f"{image_id}{source.suffix.lower()}"
            destination = image_dir / destination_name
            try:
                os.link(source, destination)
            except OSError:
                # Keep the exported dataset usable from the host if bind mounts prevent hard links.
                shutil.copy2(source, destination)
            lines = []
            for box in boxes:
                normalized = box.bbox_normalized_xywh
                if normalized and len(normalized) == 4:
                    x, y, width, height = (float(value) for value in normalized)
                else:
                    x1, y1, x2, y2 = (float(value) for value in box.bbox_xyxy_pixels)
                    x, y = (x1 + x2) / (2 * image_width), (y1 + y2) / (2 * image_height)
                    width, height = (x2 - x1) / image_width, (y2 - y1) / image_height
                if width > 0 and height > 0:
                    lines.append(f"0 {x:.6f} {y:.6f} {width:.6f} {height:.6f}")
            (label_dir / f"{image_id}.txt").write_text("\n".join(lines), encoding="utf-8")
        cursor += count

    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: license_plate",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return yaml_path, dict(zip(names, counts))


def _save_job(job_id: str, **values) -> None:
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        if job:
            for key, value in values.items():
                setattr(job, key, value)
            db.commit()


def _record_epoch_metrics(job_id: str, epoch: int, row: dict[str, float], score: float) -> bool:
    """Persist epoch history and return whether this is the best epoch so far."""
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        if not job:
            return False
        metrics = dict(job.metrics or {})
        history = list(metrics.get("epoch_history", []))
        best_score = metrics.get("best_score")
        is_best = best_score is None or score >= float(best_score)
        history.append({"epoch": epoch, **row, "is_best": is_best})
        metrics.update(row)
        metrics["epoch_history"] = history[-300:]
        if is_best:
            metrics["best_epoch"] = epoch
            metrics["best_score"] = score
            metrics["best_metrics"] = dict(row)
        job.metrics = metrics
        db.commit()
        return is_best


def _seed_checkpoint(model_type: str, destination: Path) -> None:
    """Keep older completed runs usable when migrating to persistent best checkpoints."""
    if destination.is_file():
        return
    with SessionLocal() as db:
        jobs = db.scalars(
            select(TrainingJob)
            .where(TrainingJob.status == "completed")
            .order_by(TrainingJob.finished_at.desc())
            .limit(100)
        ).all()
        source = next(
            (
                Path(job.model_path)
                for job in jobs
                if (job.config or {}).get("model_type", "detector") == model_type
                and job.model_path
                and Path(job.model_path).is_file()
            ),
            None,
        )
    if source is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


@celery_app.task(name="platelens.run_training_job")
def run_training_job(job_id: str):
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        if not job or job.status not in {"queued", "running"}:
            return
        job.status = "running"
        job.started_at = job.started_at or datetime.now(timezone.utc)
        config = dict(job.config)
        db.commit()

    if config.get("model_type", "detector") == "ocr":
        try:
            from .ocr_training import run_ocr_training

            return run_ocr_training(job_id, config)
        except Exception as exc:
            _save_job(job_id, status="failed", error=str(exc)[:2000], finished_at=datetime.now(timezone.utc))
            return

    try:
        import os

        training_dir = Path(os.environ.get("TRAINING_DIR", "/training"))
        config_dir = training_dir / "ultralytics-config"
        pretrained_dir = training_dir / "pretrained"
        config_dir.mkdir(parents=True, exist_ok=True)
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
        # Keep matplotlib and any other library cache inside the writable
        # training volume; HOME may belong to a different uid than this process.
        os.environ.setdefault("MPLCONFIGDIR", str(config_dir / "mpl"))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        from ultralytics import YOLO

        training_dir.mkdir(parents=True, exist_ok=True)
        dataset_yaml, split_counts = _prepare_dataset(job_id, config, training_dir)
        _save_job(job_id, split_counts=split_counts)
        expected_batches = max(1, math.ceil(split_counts["train"] / config["batch_size"]))
        progress_phase = {"name": "val", "total": split_counts["val"]}
        _save_job(
            job_id,
            current_val=0,
            total_val=split_counts["val"],
            current_test=0,
            total_test=split_counts["test"],
        )
        run_dir = training_dir / "jobs" / job_id / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        global_best = training_dir / "best_detector.pt"
        _seed_checkpoint("detector", global_best)
        initial_weights = global_best if global_best.is_file() else pretrained_dir / "yolo26n.pt"
        model = YOLO(str(initial_weights))

        def cancellation_requested() -> bool:
            with SessionLocal() as db:
                job = db.get(TrainingJob, job_id)
                return not job or job.status == "cancelled" or job.cancel_requested

        # Progress counter for on_train_batch_end, which receives no batch index.
        batch_counter = {"n": 0}

        def record_train_start(trainer):
            batch_counter["n"] = 0
            _save_job(job_id, current_batch=0, total_batches=expected_batches)

        def record_epoch_start(trainer):
            batch_counter["n"] = 0
            _save_job(
                job_id,
                current_epoch=min(int(trainer.epoch) + 1, config["epochs"]),
                current_batch=0,
            )

        def record_batch(trainer):
            # Ultralytics keeps the batch index in a loop-local variable (`for i,
            # batch in pbar`) and does not expose it on the trainer, and it does
            # not pass it to on_train_batch_end. Count the callbacks instead.
            batch_counter["n"] += 1
            total_batches = expected_batches
            batch = min(batch_counter["n"], total_batches)
            if cancellation_requested():
                trainer.stop = True
                return
            update_every = max(1, total_batches // 100)
            if batch == total_batches or batch % update_every == 0:
                _save_job(
                    job_id,
                    current_epoch=min(int(trainer.epoch) + 1, config["epochs"]),
                    current_batch=batch,
                    total_batches=total_batches,
                )

        def record_epoch(trainer):
            metrics = {}
            for key, value in (getattr(trainer, "metrics", {}) or {}).items():
                try:
                    metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            epoch = min(int(trainer.epoch) + 1, config["epochs"])
            losses = getattr(trainer, "tloss", None)
            try:
                loss_values = losses.detach().cpu().flatten().tolist()
                loss_names = list(getattr(trainer, "loss_names", []) or [])
                for index, loss in enumerate(loss_values):
                    name = str(loss_names[index]) if index < len(loss_names) else f"train_loss_{index + 1}"
                    metrics[f"train/{name}"] = float(loss)
                if loss_values:
                    metrics["train/loss"] = float(sum(loss_values))
            except (AttributeError, TypeError, ValueError):
                pass
            map50 = metrics.get("metrics/mAP50(B)", 0.0)
            map50_95 = metrics.get("metrics/mAP50-95(B)", map50)
            score = 0.1 * map50 + 0.9 * map50_95
            is_best = _record_epoch_metrics(job_id, epoch, metrics, score)
            current_best = run_dir / "detector" / "weights" / "best.pt"
            job_best = training_dir / "jobs" / job_id / "best.pt"
            if is_best and current_best.is_file():
                job_best.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current_best, job_best)
            _save_job(
                job_id,
                current_epoch=epoch,
                current_batch=expected_batches,
                total_batches=expected_batches,
            )

        def record_validation_start(validator):
            split = progress_phase["name"]
            _save_job(job_id, **{f"current_{split}": 0, f"total_{split}": progress_phase["total"]})

        def record_validation_batch(validator):
            split = progress_phase["name"]
            total = progress_phase["total"]
            seen = int(getattr(validator, "seen", 0) or 0)
            if seen <= 0:
                seen = (int(getattr(validator, "batch_i", -1)) + 1) * int(getattr(validator, "batch_size", 1) or 1)
            completed = min(seen, total)
            update_every = max(1, total // 50)
            if completed == total or completed % update_every == 0:
                _save_job(job_id, **{f"current_{split}": completed, f"total_{split}": total})

        def record_validation_end(validator):
            split = progress_phase["name"]
            _save_job(job_id, **{f"current_{split}": progress_phase["total"]})

        model.add_callback("on_train_start", record_train_start)
        model.add_callback("on_train_epoch_start", record_epoch_start)
        model.add_callback("on_train_batch_end", record_batch)
        model.add_callback("on_fit_epoch_end", record_epoch)
        model.add_callback("on_val_start", record_validation_start)
        model.add_callback("on_val_batch_end", record_validation_batch)
        model.add_callback("on_val_end", record_validation_end)
        model.train(
            data=str(dataset_yaml),
            epochs=config["epochs"],
            batch=config["batch_size"],
            imgsz=config["image_size"],
            device=None if config["device"] == "auto" else config["device"],
            project=str(run_dir),
            name="detector",
            exist_ok=True,
            workers=2,
            seed=42,
            plots=False,
            verbose=False,
        )
        if cancellation_requested():
            _save_job(job_id, status="cancelled", finished_at=datetime.now(timezone.utc))
            return
        best_weights = run_dir / "detector" / "weights" / "best.pt"
        if not best_weights.is_file():
            raise RuntimeError("Training finished without producing best.pt")
        if cancellation_requested():
            _save_job(job_id, status="cancelled", finished_at=datetime.now(timezone.utc))
            return

        trained_model = YOLO(str(best_weights))
        progress_phase["name"] = "test"
        progress_phase["total"] = split_counts["test"]
        trained_model.add_callback("on_val_start", record_validation_start)
        trained_model.add_callback("on_val_batch_end", record_validation_batch)
        trained_model.add_callback("on_val_end", record_validation_end)
        evaluation = trained_model.val(
            data=str(dataset_yaml),
            split="test",
            imgsz=config["image_size"],
            batch=config["batch_size"],
            device=None if config["device"] == "auto" else config["device"],
            project=str(run_dir),
            name="test",
            exist_ok=True,
            plots=False,
        )
        if cancellation_requested():
            _save_job(job_id, status="cancelled", finished_at=datetime.now(timezone.utc))
            return
        test_metrics = {}
        for key, value in (getattr(evaluation, "results_dict", {}) or {}).items():
            try:
                test_metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        final_metrics = dict(test_metrics)
        final_metrics["test_images"] = split_counts["test"]
        saved_model = training_dir / "jobs" / job_id / "best.pt"
        shutil.copy2(best_weights, saved_model)
        shutil.copy2(saved_model, global_best)
        with SessionLocal() as db:
            job = db.get(TrainingJob, job_id)
            saved_metrics = dict(job.metrics or {}) if job else {}
        final_metrics.update({key: value for key, value in saved_metrics.items() if key in {"epoch_history", "best_epoch", "best_score", "best_metrics"}})
        _save_job(
            job_id,
            status="completed",
            current_epoch=config["epochs"],
            current_batch=expected_batches,
            total_batches=expected_batches,
            metrics=final_metrics,
            model_path=str(saved_model),
            finished_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        _save_job(job_id, status="failed", error=str(exc)[:2000], finished_at=datetime.now(timezone.utc))

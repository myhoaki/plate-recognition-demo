import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from .celery_app import celery_app
from .database import SessionLocal
from .models import Annotation, ImageRecord, TrainingJob
from .storage import image_path


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
    with SessionLocal() as db:
        images = db.scalars(
            select(ImageRecord)
            .where(ImageRecord.verified_at.is_not(None))
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
    job_dir = training_dir / "jobs" / job_id
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
            (image_dir / destination_name).symlink_to(source.resolve())
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
                f"path: {dataset_dir.as_posix()}",
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

    try:
        import os

        training_dir = Path(os.environ.get("TRAINING_DIR", "/training"))
        config_dir = training_dir / "ultralytics-config"
        pretrained_dir = training_dir / "pretrained"
        config_dir.mkdir(parents=True, exist_ok=True)
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

        from ultralytics import YOLO

        training_dir.mkdir(parents=True, exist_ok=True)
        dataset_yaml, split_counts = _prepare_dataset(job_id, config, training_dir)
        _save_job(job_id, split_counts=split_counts)
        run_dir = training_dir / "jobs" / job_id / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(pretrained_dir / "yolo11n.pt"))

        def record_epoch(trainer):
            metrics = {}
            for key, value in (getattr(trainer, "metrics", {}) or {}).items():
                try:
                    metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            _save_job(job_id, current_epoch=min(int(trainer.epoch) + 1, config["epochs"]), metrics=metrics)

        model.add_callback("on_fit_epoch_end", record_epoch)
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
        best_weights = run_dir / "detector" / "weights" / "best.pt"
        if not best_weights.is_file():
            raise RuntimeError("Training finished without producing best.pt")

        trained_model = YOLO(str(best_weights))
        evaluation = trained_model.val(
            data=str(dataset_yaml),
            split="test",
            imgsz=config["image_size"],
            batch=config["batch_size"],
            device=None if config["device"] == "auto" else config["device"],
            project=str(run_dir),
            name="test",
            exist_ok=True,
        )
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
        _save_job(
            job_id,
            status="completed",
            current_epoch=config["epochs"],
            metrics=final_metrics,
            model_path=str(saved_model),
            finished_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        _save_job(job_id, status="failed", error=str(exc)[:2000], finished_at=datetime.now(timezone.utc))

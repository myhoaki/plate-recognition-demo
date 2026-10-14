from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import case, func, select, text
from sqlalchemy.orm import Session, selectinload

from . import models  # noqa: F401 - register ORM metadata
from .celery_app import celery_app
from .database import Base, SessionLocal, engine, get_db
from .logic import annotation_payload, image_payload, import_legacy_records, iou, metrics_payload, sync_raw_images
from .models import Annotation, DetectionJob, ImageRecord, TrainingJob, utcnow
from .schemas import SaveAnnotations, SetReviewStatus, SetVerification, StartDetection, StartTraining
from .storage import RAW_IMAGES_DIR, UPLOADS_DIR, image_path, save_upload
from .tasks import run_detection_job  # noqa: F401 - register Celery task
from .training import run_training_job

_refresh_lock = Lock()
_batch_queue_lock = Lock()
_refresh_state = {"running": False, "processed": 0, "total": 0, "added_or_changed": 0, "error": None}


def _worker_detection_tasks():
    try:
        inspector = celery_app.control.inspect(timeout=1)
        active = inspector.active()
        reserved = inspector.reserved()
    except Exception:
        return None
    if active is None and reserved is None:
        return None
    tasks = []
    for state, workers in (("running", active or {}), ("queued", reserved or {})):
        for worker_tasks in workers.values():
            for task in worker_tasks:
                args = task.get("args") or []
                if args and isinstance(args[0], str):
                    tasks.append({"id": args[0], "status": state})
    return tasks


def _run_raw_image_refresh() -> None:
    with _refresh_lock:
        _refresh_state.update(running=True, processed=0, total=0, added_or_changed=0, error=None)
    try:
        with SessionLocal() as db:
            def update_progress(processed: int, total: int, changed: int) -> None:
                with _refresh_lock:
                    _refresh_state.update(processed=processed, total=total, added_or_changed=changed)

            sync_raw_images(db, batch_size=500, progress_callback=update_progress)
    except Exception as exc:
        with _refresh_lock:
            _refresh_state["error"] = str(exc)
    finally:
        with _refresh_lock:
            _refresh_state["running"] = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE images ADD COLUMN IF NOT EXISTS verified_at TIMESTAMPTZ"))
        connection.execute(text("ALTER TABLE images ADD COLUMN IF NOT EXISTS review_status VARCHAR(20) NOT NULL DEFAULT 'needs_review'"))
        connection.execute(text("UPDATE images SET review_status = 'verified' WHERE verified_at IS NOT NULL AND review_status = 'needs_review'"))
    with SessionLocal() as db:
        import_legacy_records(db)
    Thread(target=_run_raw_image_refresh, daemon=True).start()
    yield


app = FastAPI(title="PlateLens API", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/api/metrics")
def dashboard_metrics(db: Session = Depends(get_db)):
    return metrics_payload(db)


@app.get("/api/images")
def list_images(
    source: str = "rawimages",
    limit: int = 500,
    offset: int = 0,
    review: str = "all",
    threshold: float = 0.7,
    review_status: str = "all",
    db: Session = Depends(get_db),
):
    if source not in {"rawimages", "upload"}:
        raise HTTPException(400, "Unsupported image source")
    if review not in {"all", "detector", "ocr", "both"}:
        raise HTTPException(400, "Unsupported confidence review filter")
    if review_status not in {"all", "needs_review", "verified", "rejected", "no_plate"}:
        raise HTTPException(400, "Unsupported image review status filter")
    if not 0 <= threshold <= 1:
        raise HTTPException(400, "Confidence threshold must be between 0 and 1")
    limit = min(max(limit, 1), 500)
    offset = max(offset, 0)
    query = select(ImageRecord).where(ImageRecord.source == source)
    if review_status != "all":
        query = query.where(ImageRecord.review_status == review_status)
    detection_confidence = (
        select(func.min(Annotation.detection_confidence))
        .where(Annotation.image_id == ImageRecord.id, Annotation.ai_detection.is_(True))
        .scalar_subquery()
    )
    ocr_confidence = (
        select(func.min(Annotation.ocr_confidence))
        .where(Annotation.image_id == ImageRecord.id, Annotation.ai_detection.is_(True))
        .scalar_subquery()
    )
    if review == "detector":
        query = query.where(detection_confidence.is_not(None), detection_confidence <= threshold)
        confidence_order = detection_confidence
    elif review == "ocr":
        query = query.where(ocr_confidence.is_not(None), ocr_confidence <= threshold)
        confidence_order = ocr_confidence
    elif review == "both":
        query = query.where(
            (detection_confidence.is_not(None) & (detection_confidence <= threshold))
            | (ocr_confidence.is_not(None) & (ocr_confidence <= threshold))
        )
        confidence_order = func.least(
            func.coalesce(detection_confidence, 2.0),
            func.coalesce(ocr_confidence, 2.0),
        )
    else:
        confidence_order = func.least(
            func.coalesce(detection_confidence, 2.0),
            func.coalesce(ocr_confidence, 2.0),
        )
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db.scalars(
        query
        .order_by(
            case(
                (ImageRecord.review_status == "needs_review", 0),
                (ImageRecord.review_status == "rejected", 1),
                (ImageRecord.review_status == "no_plate", 2),
                else_=3,
            ),
            confidence_order.asc().nulls_last(),
            ImageRecord.original_filename.collate("C"),
        )
        .offset(offset)
        .limit(limit)
    ).all()
    return {"images": [image_payload(item) for item in rows], "total": total, "limit": limit, "offset": offset}


@app.post("/api/images/refresh")
def refresh_images():
    with _refresh_lock:
        if not _refresh_state["running"]:
            _refresh_state.update(running=True, processed=0, total=0, added_or_changed=0, error=None)
            Thread(target=_run_raw_image_refresh, daemon=True).start()
        return dict(_refresh_state)


@app.get("/api/images/refresh")
def refresh_status():
    with _refresh_lock:
        return dict(_refresh_state)


@app.post("/api/images/upload")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = Path(file.filename or "image").name[:255]
    image_id = uuid4().hex
    suffix = Path(filename).suffix.lower()
    storage_key = f"{image_id}{suffix}"
    path, digest, width, height = await save_upload(file, storage_key)
    image = db.scalar(
        select(ImageRecord).where(ImageRecord.source == "upload", ImageRecord.original_filename == filename)
    )
    if image:
        old_path = UPLOADS_DIR / Path(image.storage_key).name
        image.annotations.clear()
        image.sha256 = digest
        image.width = width
        image.height = height
        image.storage_key = f"uploads/{storage_key}"
        image.detection_complete = False
        image.verified_at = None
        image.review_status = "needs_review"
        image_id = image.id
        db.commit()
        if old_path != path:
            old_path.unlink(missing_ok=True)
    else:
        image = ImageRecord(
            id=image_id,
            source="upload",
            original_filename=filename,
            sha256=digest,
            storage_key=f"uploads/{storage_key}",
            width=width,
            height=height,
        )
        db.add(image)
        db.commit()
    return image_payload(image)


@app.get("/api/images/{image_id}")
def image_state(image_id: str, db: Session = Depends(get_db)):
    image = db.scalar(
        select(ImageRecord)
        .options(selectinload(ImageRecord.annotations))
        .where(ImageRecord.id == image_id)
    )
    if not image:
        raise HTTPException(404, "Image not found")
    payload = image_payload(image, include_annotations=True)
    payload["annotations"] = [
        annotation_payload(item) for item in image.annotations if item.ai_detection or item.verified_result
    ]
    return payload


@app.post("/api/images/{image_id}/verification")
def set_image_verification(image_id: str, payload: SetVerification, db: Session = Depends(get_db)):
    image = db.get(ImageRecord, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    image.verified_at = utcnow() if payload.verified else None
    image.review_status = "verified" if payload.verified else "needs_review"
    db.commit()
    return {"image": image_payload(image)}


@app.post("/api/images/{image_id}/review-status")
def set_image_review_status(image_id: str, payload: SetReviewStatus, db: Session = Depends(get_db)):
    image = db.get(ImageRecord, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    image.review_status = payload.status
    image.verified_at = utcnow() if payload.status == "verified" else None
    db.commit()
    return {"image": image_payload(image)}


@app.get("/api/images/{image_id}/file")
def image_file(image_id: str, db: Session = Depends(get_db)):
    image = db.get(ImageRecord, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    try:
        path = image_path(image.source, image.original_filename, image.storage_key)
    except FileNotFoundError:
        raise HTTPException(404, "Image file not found") from None
    return FileResponse(path)


@app.post("/api/images/{image_id}/annotations")
def save_annotations(image_id: str, payload: SaveAnnotations, db: Session = Depends(get_db)):
    image = db.scalar(
        select(ImageRecord)
        .options(selectinload(ImageRecord.annotations))
        .where(ImageRecord.id == image_id)
    )
    if not image:
        raise HTTPException(404, "Image not found")
    ai_objects = [item for item in image.annotations if item.ai_detection]
    used_ai: set[int] = set()
    output = []
    for item in payload.annotations:
        x1, y1, x2, y2 = item.bbox_xyxy_pixels
        x1, x2 = sorted((max(0.0, min(x1, image.width)), max(0.0, min(x2, image.width))))
        y1, y2 = sorted((max(0.0, min(y1, image.height)), max(0.0, min(y2, image.height))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise HTTPException(400, "Each box needs a positive width and height")
        bounds = [x1, y1, x2, y2]
        candidate = None
        overlap = 0.0
        for index, ai in enumerate(ai_objects):
            if index in used_ai:
                continue
            score = iou(bounds, ai.bbox_xyxy_pixels)
            if score > overlap:
                candidate, overlap = (index, ai), score
        matched = candidate[1] if candidate and overlap >= 0.5 else None
        if matched:
            used_ai.add(candidate[0])
        text_value = item.verified_result.strip()
        if not text_value:
            raise HTTPException(400, "Enter the verified plate text; OCR guesses are not saved as ground truth automatically")
        output.append(
            Annotation(
                image_id=image.id,
                bbox_xyxy_pixels=[round(value, 2) for value in bounds],
                bbox_normalized_xywh=[
                    round((x1 + x2) / (2 * image.width), 6),
                    round((y1 + y2) / (2 * image.height), 6),
                    round((x2 - x1) / image.width, 6),
                    round((y2 - y1) / image.height, 6),
                ],
                ai_detection=matched is not None,
                ai_result=matched.ai_result if matched else None,
                verified_result=text_value,
                detection_confidence=matched.detection_confidence if matched else None,
                ocr_confidence=matched.ocr_confidence if matched else None,
                country=matched.country if matched else None,
                country_confidence=matched.country_confidence if matched else None,
            )
        )
    for annotation in list(image.annotations):
        db.delete(annotation)
    db.flush()
    db.add_all(output)
    image.verified_at = None
    image.review_status = "needs_review"
    db.commit()
    db.refresh(image)
    return {"image": image_payload(image, include_annotations=True)}


def queue_job(db: Session, kind: str, image_ids: list[str]) -> dict:
    if db.scalar(select(TrainingJob.id).where(TrainingJob.status.in_(["queued", "running"])).limit(1)):
        raise HTTPException(409, "Detector training is active. Wait for training to finish before running detection.")
    job = DetectionJob(id=uuid4().hex, kind=kind, image_ids=image_ids, total=len(image_ids), status="queued")
    db.add(job)
    db.commit()
    run_detection_job.delay(job.id)
    return {"job_id": job.id, "kind": job.kind, "status": job.status, "total": job.total}


@app.post("/api/detection-jobs")
def detect_one(payload: StartDetection, db: Session = Depends(get_db)):
    image = db.get(ImageRecord, payload.image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    return queue_job(db, "single", [image.id])


@app.post("/api/detection-jobs/batch")
def detect_batch(db: Session = Depends(get_db)):
    with _batch_queue_lock:
        worker_tasks = _worker_detection_tasks()
        active = None
        active_status = None
        if worker_tasks is None:
            active = db.scalars(
                select(DetectionJob)
                .where(DetectionJob.kind == "batch", DetectionJob.status.in_(["queued", "running"]))
                .order_by(case((DetectionJob.status == "running", 0), else_=1), DetectionJob.created_at.asc())
            ).first()
            if active is not None:
                active_status = active.status
        else:
            for task in worker_tasks:
                candidate = db.get(DetectionJob, task["id"])
                if candidate and candidate.kind == "batch" and candidate.status in {"queued", "running"}:
                    active = candidate
                    active_status = task["status"]
                    break
        if active is not None:
            return {
                "job_id": active.id,
                "kind": active.kind,
                "status": active_status,
                "total": active.total,
                "completed": active.completed,
                "existing": True,
            }
        images = db.scalars(
            select(ImageRecord)
            .where(ImageRecord.source == "rawimages")
            .order_by(ImageRecord.original_filename.collate("C"))
        ).all()
        if not images:
            raise HTTPException(400, "No images are available in rawimages/")
        return queue_job(db, "batch", [item.id for item in images])


@app.get("/api/detection-jobs/active")
def active_detection_jobs(db: Session = Depends(get_db)):
    worker_tasks = _worker_detection_tasks()
    if worker_tasks is None:
        jobs = db.scalars(
            select(DetectionJob)
            .where(DetectionJob.status.in_(["queued", "running"]))
            .order_by(DetectionJob.created_at.asc())
            .limit(50)
        ).all()
        states = {job.id: job.status for job in jobs}
    else:
        jobs = [db.get(DetectionJob, task["id"]) for task in worker_tasks]
        jobs = [job for job in jobs if job is not None and job.status in {"queued", "running"}]
        states = {task["id"]: task["status"] for task in worker_tasks}
    return {
        "jobs": [
            {
                "id": job.id,
                "kind": job.kind,
                "status": states[job.id],
                "total": job.total,
                "completed": job.completed,
                "detected": job.detected,
                "failed": job.failed,
            }
            for job in jobs
        ]
    }


@app.get("/api/jobs/{job_id}")
def job_state(job_id: str, db: Session = Depends(get_db)):
    job = db.get(DetectionJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "total": job.total,
        "completed": job.completed,
        "detected": job.detected,
        "failed": job.failed,
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


def training_payload(job: TrainingJob) -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "config": job.config,
        "split_counts": job.split_counts,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "metrics": job.metrics,
        "model_ready": bool(job.model_path and Path(job.model_path).is_file()),
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


@app.post("/api/training-jobs")
def start_training(payload: StartTraining, db: Session = Depends(get_db)):
    existing = db.scalar(
        select(TrainingJob)
        .where(TrainingJob.status.in_(["queued", "running"]))
        .order_by(TrainingJob.created_at.desc())
    )
    if existing:
        return training_payload(existing)
    detection_tasks = _worker_detection_tasks()
    if detection_tasks:
        raise HTTPException(409, "Wait for active image detection to finish before training the detector.")
    if detection_tasks is None and db.scalar(
        select(DetectionJob.id).where(DetectionJob.status.in_(["queued", "running"])).limit(1)
    ):
        raise HTTPException(409, "Wait for active image detection to finish before training the detector.")
    config = payload.model_dump()
    job = TrainingJob(
        id=uuid4().hex,
        status="queued",
        config=config,
        total_epochs=config["epochs"],
    )
    db.add(job)
    db.commit()
    run_training_job.delay(job.id)
    return training_payload(job)


@app.get("/api/training-jobs/active")
def active_training_job(db: Session = Depends(get_db)):
    job = db.scalar(
        select(TrainingJob)
        .where(TrainingJob.status.in_(["queued", "running"]))
        .order_by(TrainingJob.created_at.desc())
    )
    return {"job": training_payload(job) if job else None}


@app.get("/api/training-jobs/latest")
def latest_training_job(db: Session = Depends(get_db)):
    job = db.scalar(select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(1))
    return {"job": training_payload(job) if job else None}


@app.get("/api/training-jobs/{job_id}")
def get_training_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    return training_payload(job)


@app.post("/api/training-jobs/{job_id}/activate")
def activate_training_model(job_id: str, db: Session = Depends(get_db)):
    import os
    import shutil

    job = db.get(TrainingJob, job_id)
    if not job or job.status != "completed" or not job.model_path:
        raise HTTPException(400, "A completed training job is required.")
    training_dir = Path(os.environ.get("TRAINING_DIR", "/training")).resolve()
    model_path = Path(job.model_path).resolve()
    try:
        model_path.relative_to(training_dir / "jobs" / job.id)
    except ValueError:
        raise HTTPException(400, "Training model path is invalid.") from None
    if not model_path.is_file():
        raise HTTPException(404, "Trained model file not found.")
    active_path = training_dir / "active.pt"
    temporary_path = training_dir / "active.pt.tmp"
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, temporary_path)
    os.replace(temporary_path, active_path)
    return {"status": "active", "model": "trained detector", "job_id": job.id}

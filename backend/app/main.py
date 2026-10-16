from contextlib import asynccontextmanager
import csv
import hashlib
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy import case, func, or_, select, text
from sqlalchemy.orm import Session, selectinload

from . import models  # noqa: F401 - register ORM metadata
from .celery_app import celery_app
from .database import Base, SessionLocal, engine, get_db
from .logic import annotation_payload, image_payload, iou, metrics_payload, sync_datasets_folder
from .models import Annotation, DatasetSnapshot, DatasetManifest, DetectionJob, EvaluationJob, ImageRecord, TrainingJob, utcnow
from .schemas import RecognitionRequest, SaveAnnotations, SetReviewStatus, SetVerification, StartDetection, StartEvaluation, StartTraining, UpdateOcrSample
from .storage import DATASETS_DIR, UPLOADS_DIR, image_path, save_upload
from .tasks import run_detection_job  # noqa: F401 - register Celery task
from .training import run_training_job
from .datasets import create_detection_snapshot, tree_sha256
from .dataset_registry import audit_dataset, create_manifest, report_path
from .alpr import run_detection
from .evaluation import run_evaluation_job  # noqa: F401 - register Celery task

_batch_queue_lock = Lock()
_ocr_csv_lock = Lock()


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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE images ADD COLUMN IF NOT EXISTS verified_at TIMESTAMPTZ"))
        connection.execute(text("ALTER TABLE images ADD COLUMN IF NOT EXISTS review_status VARCHAR(20) NOT NULL DEFAULT 'needs_review'"))
        connection.execute(text("UPDATE images SET review_status = 'verified' WHERE verified_at IS NOT NULL AND review_status = 'needs_review'"))
        connection.execute(text("ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS current_batch INTEGER NOT NULL DEFAULT 0"))
        connection.execute(text("ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS total_batches INTEGER NOT NULL DEFAULT 0"))
        connection.execute(text("ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS current_val INTEGER NOT NULL DEFAULT 0"))
        connection.execute(text("ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS total_val INTEGER NOT NULL DEFAULT 0"))
        connection.execute(text("ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS current_test INTEGER NOT NULL DEFAULT 0"))
        connection.execute(text("ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS total_test INTEGER NOT NULL DEFAULT 0"))
        connection.execute(text("ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS cancel_requested BOOLEAN NOT NULL DEFAULT FALSE"))
    yield


app = FastAPI(title="PlateLens API", version="1.0.0", lifespan=lifespan)


@app.get("/api/models")
def list_models(db: Session = Depends(get_db)):
    jobs = db.scalars(select(TrainingJob).where(TrainingJob.status == "completed").order_by(TrainingJob.finished_at.desc())).all()
    output = []
    for job in jobs:
        model_type = (job.config or {}).get("model_type", "detector")
        if model_type in {"detector", "ocr"} and job.model_path and Path(job.model_path).is_file():
            output.append({"job_id": job.id, "model_type": model_type, "created_at": job.finished_at, "best_epoch": (job.metrics or {}).get("best_epoch")})
    return {"models": output}


@app.post("/api/recognition")
def recognize_image(payload: RecognitionRequest, db: Session = Depends(get_db)):
    import os
    from time import perf_counter

    image = db.get(ImageRecord, payload.image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    training_dir = Path(os.environ.get("TRAINING_DIR", "/training")).resolve()

    def selected_checkpoint(job_id: str | None, model_type: str) -> str | None:
        if not job_id:
            return None
        job = db.get(TrainingJob, job_id)
        if not job or job.status != "completed" or (job.config or {}).get("model_type", "detector") != model_type or not job.model_path:
            raise HTTPException(400, f"Select a completed {model_type} checkpoint.")
        path = Path(job.model_path).resolve()
        try:
            path.relative_to((training_dir / "jobs" / job.id).resolve())
        except ValueError:
            raise HTTPException(400, "Selected model path is invalid.") from None
        if not path.is_file():
            raise HTTPException(404, "Selected checkpoint file is missing.")
        return str(path)

    try:
        path = image_path(image.source, image.original_filename, image.storage_key)
    except FileNotFoundError:
        raise HTTPException(404, "Image file not found") from None
    detector = selected_checkpoint(payload.detector_job_id, "detector")
    ocr = selected_checkpoint(payload.ocr_job_id, "ocr")
    started = perf_counter()
    try:
        detections = run_detection(path, detector, ocr)
    except Exception as exc:
        raise HTTPException(422, f"Recognition failed: {str(exc)[:500]}") from None
    import base64
    import cv2
    frame = cv2.imread(str(path))
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection["bbox_xyxy_pixels"])
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            ok, encoded = cv2.imencode(".jpg", frame[y1:y2, x1:x2])
            if ok:
                detection["crop_data_url"] = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    return {"image_id": image.id, "filename": image.original_filename, "width": image.width, "height": image.height, "file_url": f"/api/images/{image.id}/file",
            "detections": detections, "elapsed_ms": round((perf_counter() - started) * 1000, 1),
            "detector_job_id": payload.detector_job_id, "ocr_job_id": payload.ocr_job_id}


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/api/metrics")
def dashboard_metrics(db: Session = Depends(get_db)):
    return metrics_payload(db)


def _snapshot_payload(snapshot: DatasetSnapshot) -> dict:
    return {"id": snapshot.id, "task": snapshot.task, "split_counts": snapshot.split_counts,
            "manifest_sha256": snapshot.manifest_sha256, "created_at": snapshot.created_at}


@app.get("/api/dataset-snapshots")
def list_dataset_snapshots(task: str | None = None, db: Session = Depends(get_db)):
    if task is not None and task not in {"detector", "ocr"}:
        raise HTTPException(400, "Unsupported dataset task")
    query = select(DatasetSnapshot).order_by(DatasetSnapshot.created_at.desc())
    if task:
        query = query.where(DatasetSnapshot.task == task)
    return {"snapshots": [_snapshot_payload(row) for row in db.scalars(query).all()]}


@app.get("/api/dataset-manifests")
def list_dataset_manifests(task: str | None = None, db: Session = Depends(get_db)):
    if task is not None and task not in {"detector", "ocr"}:
        raise HTTPException(400, "Unsupported dataset task")
    query = select(DatasetManifest).order_by(DatasetManifest.created_at.desc())
    if task:
        query = query.where(DatasetManifest.task == task)
    return {"manifests": [{"id": row.id, "task": row.task, "version": row.version, "path": row.path,
                           "manifest_sha256": row.manifest_sha256, "sample_count": row.sample_count, "created_at": row.created_at}
                          for row in db.scalars(query.limit(100)).all()]}


@app.post("/api/dataset-manifests")
def create_dataset_manifest(task: str):
    try:
        payload, path = create_manifest(task, DATASETS_DIR)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"task": task, "version": payload["version"], "path": str(path), "sample_count": len(payload["samples"]),
            "split_counts": payload["split_counts"], "split_percentages": payload["split_percentages"],
            "grouping_warning": payload["grouping_warning"]}


@app.get("/api/dataset-audit")
def get_dataset_audit(task: str = "detector"):
    try:
        report = audit_dataset(task)
        path = report_path(DATASETS_DIR, f"audit-{task}", report)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    report["report_path"] = str(path)
    return report


@app.post("/api/dataset-snapshots")
def create_dataset_snapshot(task: str, db: Session = Depends(get_db)):
    if task == "detector":
        try:
            snapshot = create_detection_snapshot()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return _snapshot_payload(snapshot)
    if task == "ocr":
        import os
        from .ocr_training import _collect_crops
        snapshot_id = uuid4().hex
        training_dir = Path(os.environ.get("TRAINING_DIR", "/training"))
        snapshot_path = DATASETS_DIR.resolve() / "snapshots" / "ocr" / snapshot_id
        try:
            try:
                create_manifest("ocr", DATASETS_DIR)
            except ValueError:
                # Keep supporting legacy OCR CSV rows that have not been registered yet.
                pass
            _collect_crops(snapshot_id, training_dir, output_dir=snapshot_path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        except Exception as exc:
            raise HTTPException(400, f"Could not prepare OCR snapshot: {exc}") from None
        manifest = __import__("json").loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
        snapshot = DatasetSnapshot(id=snapshot_id, task="ocr", path=str(snapshot_path), split_counts=manifest["split_counts"], manifest_sha256=tree_sha256(snapshot_path))
        db.add(snapshot)
        db.commit()
        db.refresh(snapshot)
        for item in snapshot_path.rglob("*"):
            if item.is_file():
                item.chmod(0o444)
        for item in sorted((path for path in snapshot_path.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
            item.chmod(0o555)
        snapshot_path.chmod(0o555)
        return _snapshot_payload(snapshot)
    raise HTTPException(400, "Task must be detector or ocr")


def _ocr_sample_rows():
    root = Path(os.environ.get("RAW_OCR_DATASET_DIR", "/raw-dataset-ocr")).resolve()
    manifest = root / "annotations.csv"
    if not manifest.is_file():
        return root, [], []
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if "image_path" not in fields:
            raise HTTPException(400, "OCR annotation CSV must contain an image_path column.")
        rows = list(reader)
    keyed = []
    for index, row in enumerate(rows):
        key = f"{index}\0{row.get('image_path', '')}\0{row.get('object_index', '')}"
        sample_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        keyed.append((sample_id, row))
    return root, fields, keyed


def _ocr_sample_file(root: Path, row: dict) -> Path:
    relative = Path(row.get("image_path", ""))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(400, "OCR sample image path is invalid.")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise HTTPException(400, "OCR sample image path is outside the configured dataset.") from None
    if not path.is_file():
        raise HTTPException(404, "OCR sample image is missing.")
    return path


@app.get("/api/ocr-samples")
def list_ocr_samples(limit: int = 100, offset: int = 0):
    root, _fields, rows = _ocr_sample_rows()
    limit, offset = min(max(limit, 1), 500), max(0, offset)
    result = []
    for sample_id, row in rows[offset:offset+limit]:
        try:
            source = _ocr_sample_file(root, row)
        except HTTPException as exc:
            source = None
            error = exc.detail
        else:
            error = None
        result.append({"id": sample_id, "filename": Path(row.get("image_path", "")).name,
                       "file_url": f"/api/ocr-samples/{sample_id}/image", "plate_text": row.get("plate_text", ""),
                       "verified": str(row.get("verified", "")).strip().lower() in {"1", "true", "yes", "verified"},
                       "status": row.get("review_status") or ("verified" if str(row.get("verified", "")).strip().lower() in {"1", "true", "yes", "verified"} else "pending"),
                       "is_crop": str(row.get("is_crop", "")).strip().lower() in {"1", "true", "yes"},
                       "error": error, "available": source is not None})
    return {"samples": result, "total": len(rows), "offset": offset, "limit": limit}


@app.get("/api/ocr-samples/{sample_id}/image")
def ocr_sample_image(sample_id: str, margin: float = Query(default=0.08, ge=0, le=0.3)):
    root, _fields, rows = _ocr_sample_rows()
    row = next((item for key, item in rows if key == sample_id), None)
    if row is None:
        raise HTTPException(404, "OCR sample not found")
    source = _ocr_sample_file(root, row)
    if str(row.get("is_crop", "")).strip().lower() in {"1", "true", "yes"}:
        return FileResponse(source)
    import cv2
    image = cv2.imread(str(source))
    if image is None:
        raise HTTPException(400, "OCR sample image cannot be decoded.")
    height, width = image.shape[:2]
    coords = [str(row.get(key, "")).strip() for key in ("x1", "y1", "x2", "y2")]
    box = None
    if all(coords):
        try:
            box = tuple(float(value) for value in coords)
        except ValueError:
            raise HTTPException(400, "OCR sample crop coordinates are malformed.") from None
    elif any(coords):
        raise HTTPException(400, "OCR sample must supply all four crop coordinates.")
    else:
        try:
            parts = list(source.relative_to(root).parts)
            image_index = next(index for index, part in enumerate(parts) if part.lower() == "images")
            parts[image_index] = "labels"
            label_path = root.joinpath(*parts).with_suffix(".txt")
            lines = label_path.read_text(encoding="utf-8").splitlines()
            object_index = int(row.get("object_index") or 0)
        except (OSError, ValueError):
            lines, object_index = [], 0
        boxes = []
        for line in lines:
            try:
                values = [float(value) for value in line.split()]
            except ValueError:
                continue
            points = values[1:]
            if len(points) == 4:
                cx, cy, bw, bh = points
                boxes.append(((cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height))
            elif len(points) >= 6 and len(points) % 2 == 0:
                boxes.append((min(points[0::2])*width, min(points[1::2])*height, max(points[0::2])*width, max(points[1::2])*height))
        if 0 <= object_index < len(boxes):
            box = boxes[object_index]
    if box is None:
        raise HTTPException(400, "No crop coordinates or matching YOLO box were found for this sample.")
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2-x1)*margin, (y2-y1)*margin
    x1, x2 = max(0, int(x1-pad_x)), min(width, int(x2+pad_x))
    y1, y2 = max(0, int(y1-pad_y)), min(height, int(y2+pad_y))
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(400, "OCR sample crop is empty or outside its image.")
    ok, encoded = cv2.imencode(".jpg", image[y1:y2, x1:x2])
    if not ok:
        raise HTTPException(400, "Could not encode OCR sample crop.")
    return Response(encoded.tobytes(), media_type="image/jpeg")


@app.put("/api/ocr-samples/{sample_id}")
def update_ocr_sample(sample_id: str, update: UpdateOcrSample):
    root, fields, rows = _ocr_sample_rows()
    selected_index = next((index for index, (key, _) in enumerate(rows) if key == sample_id), None)
    if selected_index is None:
        raise HTTPException(404, "OCR sample not found")
    if update.status == "verified" and not update.plate_text:
        raise HTTPException(400, "Enter the ground-truth plate text before verifying this OCR sample.")
    with _ocr_csv_lock:
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "annotations.csv"
        names = list(fields)
        for name in ("plate_text", "verified", "review_status"):
            if name not in names:
                names.append(name)
        row_values = [dict(row) for _, row in rows]
        row_values[selected_index]["plate_text"] = update.plate_text
        row_values[selected_index]["verified"] = "true" if update.status == "verified" else "false"
        row_values[selected_index]["review_status"] = update.status
        with NamedTemporaryFile("w", newline="", encoding="utf-8", dir=root, delete=False, suffix=".csv.tmp") as temp:
            temp_path = Path(temp.name)
            writer = csv.DictWriter(temp, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(row_values)
        os.replace(temp_path, manifest)
    return {"id": sample_id, "plate_text": update.plate_text, "status": update.status}


def evaluation_payload(job: EvaluationJob) -> dict:
    return {"id": job.id, "status": job.status, "config": job.config, "processed": job.processed,
            "total": job.total, "results": job.results, "error": job.error, "created_at": job.created_at,
            "started_at": job.started_at, "finished_at": job.finished_at}


@app.post("/api/evaluations")
def start_evaluation(payload: StartEvaluation, db: Session = Depends(get_db)):
    snapshot = db.get(DatasetSnapshot, payload.dataset_snapshot_id)
    if not snapshot or snapshot.task != "detector":
        raise HTTPException(400, "Select a detector snapshot with a fixed test split.")
    active = db.scalar(select(EvaluationJob).where(EvaluationJob.status.in_(["queued", "running"])).limit(1))
    if active:
        return evaluation_payload(active)
    for job_id, model_type in ((payload.detector_job_id, "detector"), (payload.ocr_job_id, "ocr")):
        if job_id:
            job = db.get(TrainingJob, job_id)
            if not job or job.status != "completed" or (job.config or {}).get("model_type", "detector") != model_type:
                raise HTTPException(400, f"Select a completed {model_type} checkpoint.")
    job = EvaluationJob(id=uuid4().hex, status="queued", config=payload.model_dump())
    db.add(job)
    db.commit()
    run_evaluation_job.delay(job.id)
    return evaluation_payload(job)


@app.get("/api/evaluations/{job_id}")
def get_evaluation(job_id: str, db: Session = Depends(get_db)):
    job = db.get(EvaluationJob, job_id)
    if not job:
        raise HTTPException(404, "Evaluation job not found")
    return evaluation_payload(job)


@app.get("/api/images")
def list_images(
    source: str = "registered",
    limit: int = 500,
    offset: int = 0,
    review: str = "all",
    threshold: float = 0.7,
    review_status: str = "all",
    db: Session = Depends(get_db),
):
    if source not in {"registered", "upload", "datasets_folder"}:
        raise HTTPException(400, "Unsupported image source")
    if review not in {"all", "detector", "ocr", "both"}:
        raise HTTPException(400, "Unsupported confidence review filter")
    if review_status not in {"all", "needs_review", "reviewed", "verified", "rejected", "no_plate"}:
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


@app.post("/api/images/upload")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = Path(file.filename or "image").name[:255]
    image_id = uuid4().hex
    suffix = Path(filename).suffix.lower()
    storage_key = f"{image_id}{suffix}"
    path, digest, width, height = await save_upload(file, storage_key)
    duplicate = db.scalar(select(ImageRecord).where(ImageRecord.source == "upload", ImageRecord.sha256 == digest))
    if duplicate:
        path.unlink(missing_ok=True)
        return image_payload(duplicate)
    image = db.scalar(
        select(ImageRecord).where(ImageRecord.source == "upload", ImageRecord.original_filename == filename)
    )
    if image:
        stem, extension = Path(filename).stem, Path(filename).suffix
        filename = f"{stem[:220]}-{image_id[:8]}{extension}"[:255]
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


@app.post("/api/images/sync-datasets-folder")
def sync_datasets_folder_endpoint(db: Session = Depends(get_db)):
    """Scan datasets/imports/ for images and register them for review.

    Images are registered with source='datasets_folder'. YOLO sidecar label
    files are imported as unverified annotations. Existing records are updated
    only when the file hash changes. Returns the count of new/updated images.
    """
    try:
        count = sync_datasets_folder(db)
    except Exception as exc:
        raise HTTPException(500, f"Sync failed: {str(exc)[:300]}") from None
    total = db.scalar(select(func.count()).select_from(ImageRecord).where(ImageRecord.source == "datasets_folder")) or 0
    return {"synced": count, "total": total}


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
        annotation_payload(item) for item in image.annotations
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
            .where(ImageRecord.source.in_(["registered", "datasets_folder"]))
            .order_by(ImageRecord.original_filename.collate("C"))
        ).all()
        if not images:
            raise HTTPException(400, "No images are available. Sync the datasets folder or import images first.")
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
    train_images = int((job.split_counts or {}).get("train", 0))
    batch_size = int((job.config or {}).get("batch_size", 1))
    total_batches = max(1, math.ceil(train_images / max(1, batch_size))) if train_images else job.total_batches
    saved_metrics = dict(job.metrics or {})
    epoch_history = saved_metrics.pop("epoch_history", [])
    best_epoch = saved_metrics.pop("best_epoch", None)
    best_score = saved_metrics.pop("best_score", None)
    best_metrics = saved_metrics.pop("best_metrics", {})
    return {
        "id": job.id,
        "status": job.status,
        "model_type": (job.config or {}).get("model_type", "detector"),
        "config": job.config,
        "split_counts": job.split_counts,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "current_batch": job.current_batch,
        "total_batches": total_batches,
        "current_val": job.current_val,
        "total_val": job.total_val or int((job.split_counts or {}).get("val", 0)),
        "current_test": job.current_test,
        "total_test": job.total_test or int((job.split_counts or {}).get("test", 0)),
        "cancel_requested": job.cancel_requested,
        "metrics": saved_metrics,
        "epoch_history": epoch_history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "model_ready": bool(job.model_path and Path(job.model_path).is_file()),
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


@app.post("/api/training-jobs")
def start_training(payload: StartTraining, db: Session = Depends(get_db)):
    snapshot = db.get(DatasetSnapshot, payload.dataset_snapshot_id)
    if not snapshot or snapshot.task != payload.model_type:
        raise HTTPException(400, f"Select an existing {payload.model_type} dataset snapshot before training.")
    existing = db.scalar(
        select(TrainingJob)
        .where(TrainingJob.status.in_(["queued", "running"]))
        .order_by(TrainingJob.created_at.desc())
    )
    if existing:
        if (existing.config or {}).get("model_type", "detector") != payload.model_type:
            raise HTTPException(409, f"{(existing.config or {}).get('model_type', 'detector').title()} training is already active.")
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
def latest_training_job(model_type: str | None = None, db: Session = Depends(get_db)):
    if model_type is not None and model_type not in {"detector", "ocr"}:
        raise HTTPException(400, "Unsupported training model type")
    query = select(TrainingJob).order_by(TrainingJob.created_at.desc())
    if model_type == "ocr":
        query = query.where(TrainingJob.config["model_type"].astext == "ocr")
    elif model_type == "detector":
        model_field = TrainingJob.config["model_type"].astext
        query = query.where(or_(model_field == "detector", model_field.is_(None)))
    job = db.scalar(query.limit(1))
    return {"job": training_payload(job) if job else None}


@app.get("/api/training-jobs/{job_id}")
def get_training_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    return training_payload(job)


@app.post("/api/training-jobs/{job_id}/cancel")
def cancel_training_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    if job.status not in {"queued", "running"}:
        raise HTTPException(409, f"Cannot stop a {job.status} training job")

    job.cancel_requested = True
    if job.status == "queued":
        job.status = "cancelled"
        job.finished_at = utcnow()
    else:
        try:
            inspector = celery_app.control.inspect(timeout=1)
            worker_tasks = (inspector.active() or {}) | (inspector.reserved() or {})
            task_is_alive = any(
                any((task.get("args") or [None])[0] == job_id for task in tasks)
                for tasks in worker_tasks.values()
            )
        except Exception:
            task_is_alive = True
        if not task_is_alive:
            job.status = "cancelled"
            job.finished_at = utcnow()
    db.commit()
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
    model_type = (job.config or {}).get("model_type", "detector")
    extension = ".ocr.pt" if model_type == "ocr" else ".pt"
    active_path = training_dir / ("active_ocr.pt" if model_type == "ocr" else "active.pt")
    temporary_path = training_dir / f"active{extension}.tmp"
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, temporary_path)
    os.replace(temporary_path, active_path)
    return {"status": "active", "model": f"trained {model_type}", "job_id": job.id}

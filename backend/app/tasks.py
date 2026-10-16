from datetime import datetime, timezone

from sqlalchemy import func, select

from .alpr import run_detection
from .celery_app import celery_app
from .database import SessionLocal
from .logic import iou
from .models import Annotation, DetectionJob, ImageRecord
from .storage import image_path


@celery_app.task(name="platelens.run_detection_job")
def run_detection_job(job_id: str):
    with SessionLocal() as db:
        job = db.get(DetectionJob, job_id)
        if not job or job.status not in {"queued", "running"}:
            return
        image_ids = list(job.image_ids)
        start_index = min(max(job.completed, 0), min(job.total, len(image_ids)))
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        job.completed = start_index
        db.commit()

    if start_index >= min(job.total, len(image_ids)):
        with SessionLocal() as db:
            job = db.get(DetectionJob, job_id)
            if job and job.status == "running":
                detected_query = select(func.count(func.distinct(Annotation.image_id))).where(
                    Annotation.ai_detection.is_(True)
                )
                if job.kind == "batch":
                    detected_query = detected_query.join(ImageRecord).where(ImageRecord.source == "registered")
                elif image_ids:
                    detected_query = detected_query.where(Annotation.image_id == image_ids[0])
                job.detected = db.scalar(detected_query) or 0
                job.failed = min(job.failed, job.total)
                job.status = "partial" if job.failed else "completed"
                job.finished_at = datetime.now(timezone.utc)
                db.commit()
        return

    for image_id in image_ids[start_index:job.total]:
        with SessionLocal() as db:
            image = db.get(ImageRecord, image_id)
            job = db.get(DetectionJob, job_id)
            if not job or job.status not in {"queued", "running"}:
                return
            if not image:
                job.failed += 1
                job.completed = min(job.completed + 1, job.total)
                db.commit()
                continue
            try:
                detections = run_detection(image_path(image.source, image.original_filename, image.storage_key))
                old = list(db.scalars(select(Annotation).where(Annotation.image_id == image_id)).all())
                matched: set[int] = set()
                new_annotations = []
                for detection in detections:
                    bounds = detection["bbox_xyxy_pixels"]
                    best_index, best_overlap = None, 0.0
                    for index, previous in enumerate(old):
                        if index in matched:
                            continue
                        overlap = iou(bounds, previous.bbox_xyxy_pixels)
                        if overlap > best_overlap:
                            best_index, best_overlap = index, overlap
                    previous = old[best_index] if best_index is not None and best_overlap >= 0.15 else None
                    if previous:
                        matched.add(best_index)
                    verified = previous.verified_result if previous else None
                    width, height = image.width, image.height
                    x1, y1, x2, y2 = bounds
                    new_annotations.append(
                        Annotation(
                            image_id=image_id,
                            bbox_xyxy_pixels=[round(value, 2) for value in bounds],
                            bbox_normalized_xywh=[
                                round((x1 + x2) / (2 * width), 6),
                                round((y1 + y2) / (2 * height), 6),
                                round((x2 - x1) / width, 6),
                                round((y2 - y1) / height, 6),
                            ],
                            ai_detection=True,
                            ai_result=detection["text"] or None,
                            verified_result=verified,
                            detection_confidence=detection["detection_confidence"],
                            ocr_confidence=detection["ocr_confidence"],
                            country=detection["country"],
                            country_confidence=detection["country_confidence"],
                        )
                    )
                for index, previous in enumerate(old):
                    if index not in matched:
                        new_annotations.append(
                            Annotation(
                                image_id=image_id,
                                bbox_xyxy_pixels=previous.bbox_xyxy_pixels,
                                bbox_normalized_xywh=previous.bbox_normalized_xywh,
                                ai_detection=previous.ai_detection,
                                ai_result=previous.ai_result,
                                verified_result=previous.verified_result,
                                detection_confidence=previous.detection_confidence,
                                ocr_confidence=previous.ocr_confidence,
                                country=previous.country,
                                country_confidence=previous.country_confidence,
                            )
                        )
                for previous in old:
                    db.delete(previous)
                db.flush()
                db.add_all(new_annotations)
                image.detection_complete = True
                image.verified_at = None
                image.review_status = "needs_review"
                job.detected += bool(detections)
                job.completed = min(job.completed + 1, job.total)
                db.commit()
            except Exception as exc:
                db.rollback()
                job = db.get(DetectionJob, job_id)
                if job:
                    job.failed += 1
                    job.completed = min(job.completed + 1, job.total)
                    job.error = str(exc)[:1000]
                    db.commit()

    with SessionLocal() as db:
        job = db.get(DetectionJob, job_id)
        if job:
            job.status = "partial" if job.failed and job.completed else "failed" if job.failed else "completed"
            job.finished_at = datetime.now(timezone.utc)
            db.commit()

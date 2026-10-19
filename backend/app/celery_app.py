import os

from celery import Celery
from celery.signals import worker_ready

celery_app = Celery(
    "platelens",
    broker=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
    backend=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
    include=["app.tasks", "app.training", "app.evaluation"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


@worker_ready.connect
def reap_orphaned_training_jobs(**_kwargs):
    """Fail training rows that a previous worker died on.

    Recreating the worker container (a deploy) kills any in-flight training
    task, but the database row keeps `status='running'` forever. That is worse
    than a plain failure: the UI shows the run as active indefinitely, and
    POST /api/detection-jobs refuses to start while a training job looks
    active, so the whole verification queue is blocked with no way out.

    `worker_ready` fires once this worker is up and consuming, before it has
    executed anything. With a single worker at concurrency=1, any row still
    marked `running` at that moment therefore belongs to a dead process and is
    safe to close out.

    Only `running` rows are touched. A row left `queued` is not orphaned: the
    broker still holds its task, and this same worker will pick it up.
    """
    from .database import SessionLocal
    from .models import TrainingJob, utcnow
    from sqlalchemy import select

    try:
        with SessionLocal() as db:
            orphans = db.scalars(
                select(TrainingJob).where(TrainingJob.status == "running")
            ).all()
            if not orphans:
                return
            for job in orphans:
                job.status = "failed"
                job.error = (
                    "Interrupted: the worker restarted while this run was in "
                    f"progress (stopped at epoch {job.current_epoch} of "
                    f"{job.total_epochs}, batch {job.current_batch} of "
                    f"{job.total_batches}). No checkpoint was produced."
                )
                job.finished_at = utcnow()
            db.commit()
            print(f"[platelens] marked {len(orphans)} orphaned training job(s) as failed")
    except Exception as exc:  # pragma: no cover - never block worker startup
        print(f"[platelens] could not reap orphaned training jobs: {exc}")

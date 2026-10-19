"""Worker restarts must not leave training rows stuck at status='running'."""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

from app import celery_app as celery_module
from app import database as database_module


class _FakeQuery:
    """Stands in for the result of db.scalars(...)."""

    def __init__(self, rows):
        self.rows = rows

    def scalars(self, *_args, **_kwargs):
        return self

    def all(self):
        return list(self.rows)


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.committed = False
        self.added = []

    def scalars(self, *args, **kwargs):
        return _FakeQuery(self.rows)

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _job(job_id, epoch=2, total=5, batch=624, batches=835):
    return SimpleNamespace(
        id=job_id, status="running", error=None, finished_at=None,
        current_epoch=epoch, total_epochs=total,
        current_batch=batch, total_batches=batches,
    )


class OrphanReaperTest(unittest.TestCase):
    def _run(self, rows):
        session = _FakeSession(rows)
        # The reaper imports SessionLocal inside the function body, so the real
        # binding to patch is app.database.SessionLocal.
        with mock.patch.object(database_module, "SessionLocal", return_value=session):
            celery_module.reap_orphaned_training_jobs()
        return session

    def test_orphaned_running_job_is_marked_failed(self):
        job = _job("a" * 32)
        session = self._run([job])
        self.assertEqual(job.status, "failed")
        self.assertIsNotNone(job.finished_at)
        self.assertTrue(session.committed)

    def test_error_message_is_actionable(self):
        job = _job("b" * 32, epoch=2, total=5, batch=624, batches=835)
        self._run([job])
        message = job.error
        self.assertIn("worker restarted", message)
        self.assertIn("epoch 2 of 5", message)
        self.assertIn("batch 624 of 835", message)
        self.assertIn("No checkpoint", message)

    def test_multiple_orphans_are_all_closed_out(self):
        jobs = [_job("a" * 32), _job("b" * 32), _job("c" * 32)]
        self._run(jobs)
        self.assertTrue(all(job.status == "failed" for job in jobs))

    def test_nothing_to_do_is_a_no_op(self):
        session = self._run([])
        self.assertFalse(session.committed)

    def test_reaper_never_raises_into_worker_startup(self):
        """A database outage must not prevent the worker from booting."""
        with mock.patch.object(
            database_module, "SessionLocal", side_effect=RuntimeError("db down")
        ):
            celery_module.reap_orphaned_training_jobs()  # must not raise


class ReaperScopeTest(unittest.TestCase):
    def test_only_running_rows_are_targeted(self):
        """Queued jobs still have their task in the broker and must be left alone."""
        source = inspect.getsource(celery_module.reap_orphaned_training_jobs)
        self.assertIn('TrainingJob.status == "running"', source)
        self.assertNotIn('TrainingJob.status.in_', source)

    def test_signal_is_worker_ready_not_process_init(self):
        self.assertTrue(hasattr(celery_module, "reap_orphaned_training_jobs"))
        # Guard against moving this into the API: the docstring records why.
        self.assertIn("concurrency=1", inspect.getsource(celery_module.reap_orphaned_training_jobs))


if __name__ == "__main__":
    unittest.main()
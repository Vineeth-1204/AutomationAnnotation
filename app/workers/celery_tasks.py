"""
Celery task definitions for all long-running pipeline operations.

Each task:
  1. Marks the DB job RUNNING and stores the Celery task_id.
  2. Calls the corresponding service method (which contains all the real logic).
  3. Reports progress checkpoints into ProcessingJob.result.
  4. Marks the DB job COMPLETED or FAILED.
  5. Retries on transient failures with exponential back-off.

Cancellation:
  POST /api/v1/jobs/{job_id}/cancel  →  celery_app.control.revoke(task_id, terminate=True)
  The endpoint also sets the DB status to CANCELLED directly.

Progress format written to ProcessingJob.result during execution:
  {"progress": 45, "progress_message": "Annotating image 9/20", ...final_fields}
"""

import asyncio
import logging
import os
import tempfile
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from app.celery_app import celery_app

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run_async(coro):
    """Run an async coroutine from a synchronous Celery task."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


async def _mark_job(job_id: int, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Update a ProcessingJob's status (and optional extra fields) inside its own DB session."""
    import app.db.session as db_session
    from app.repositories.processing_job import ProcessingJobRepository

    async with db_session.SessionLocal() as db:
        repo = ProcessingJobRepository(db)
        job = await repo.get(job_id)
        if not job:
            logger.warning(f"[celery] Job {job_id} not found when trying to set status={status}")
            return

        update = {"status": status}
        if status == "RUNNING":
            update["started_at"] = datetime.now(timezone.utc)
        elif status in ("COMPLETED", "FAILED", "CANCELLED"):
            update["completed_at"] = datetime.now(timezone.utc)

        if extra:
            update.update(extra)

        await repo.update(job, update)
        await db.commit()


async def _set_task_id(job_id: int, task_id: str) -> None:
    """Persist the Celery task UUID into the DB row so the cancel endpoint can use it."""
    import app.db.session as db_session
    from app.repositories.processing_job import ProcessingJobRepository

    async with db_session.SessionLocal() as db:
        repo = ProcessingJobRepository(db)
        job = await repo.get(job_id)
        if job:
            await repo.update(job, {"task_id": task_id})
            await db.commit()


async def _set_progress(job_id: int, progress: int, message: str, extra: Optional[Dict] = None) -> None:
    """Write incremental progress to ProcessingJob.result without changing status."""
    import app.db.session as db_session
    from app.repositories.processing_job import ProcessingJobRepository

    async with db_session.SessionLocal() as db:
        repo = ProcessingJobRepository(db)
        job = await repo.get(job_id)
        if job:
            merged = {**(job.result or {}), "progress": progress, "progress_message": message}
            if extra:
                merged.update(extra)
            await repo.update(job, {"result": merged})
            await db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Base Task class with structured error handling
# ──────────────────────────────────────────────────────────────────────────────

class ManagedTask(Task):
    """
    Custom base Task that automatically catches unhandled exceptions, writes
    them to the DB job's error_message, and marks the job as FAILED.
    """
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job_id = args[0] if args else kwargs.get("job_id")
        if job_id:
            _run_async(_mark_job(
                job_id,
                "FAILED",
                {"error_message": f"{type(exc).__name__}: {exc}"}
            ))
        super().on_failure(exc, task_id, args, kwargs, einfo)


# ──────────────────────────────────────────────────────────────────────────────
# Task 1 — Image Collection + Pipeline Filtering
# ──────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    base=ManagedTask,
    name="app.workers.celery_tasks.run_image_collection",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    soft_time_limit=600,
)
def run_image_collection(
    self,
    job_id: int,
    dataset_id: int,
    queries: List[str],
    limit_per_query: int,
) -> Dict[str, Any]:
    """Scrape images, run preprocessing pipeline (blur, dedup, CLIP), store results."""

    async def _run():
        from app.services.image import ImageService
        from app.repositories.image import ImageRepository
        from app.repositories.processing_job import ProcessingJobRepository

        await _set_task_id(job_id, self.request.id)
        await _mark_job(job_id, "RUNNING")
        await _set_progress(job_id, 5, "Initialising image collection …")

        try:
            # ImageService needs a repository — build a minimal one with its own session
            import app.db.session as db_session
            async with db_session.SessionLocal() as db:
                from app.repositories.image import ImageRepository
                repo = ImageRepository(db)
                service = ImageService(repo)
                await service.run_collection_background_task(
                    job_id=job_id,
                    dataset_id=dataset_id,
                    queries=queries,
                    limit_per_query=limit_per_query,
                )
        except (IOError, OSError, ConnectionError) as exc:
            logger.warning(f"[run_image_collection] Transient error, retrying: {exc}")
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)
        except SoftTimeLimitExceeded:
            await _mark_job(job_id, "FAILED", {"error_message": "Task exceeded time limit"})
            raise

    _run_async(_run())
    return {"job_id": job_id, "status": "COMPLETED"}


# ──────────────────────────────────────────────────────────────────────────────
# Task 2 — Auto-Annotation (Grounding DINO + SAM2 + Florence-2)
# ──────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    base=ManagedTask,
    name="app.workers.celery_tasks.run_annotation",
    bind=True,
    max_retries=3,
    default_retry_delay=15,
    soft_time_limit=1800,
)
def run_annotation(
    self,
    job_id: int,
    dataset_id: int,
) -> Dict[str, Any]:
    """Run Grounding DINO + SAM2 auto-annotation for every image in the dataset."""

    async def _run():
        from app.repositories.annotation import AnnotationRepository
        from app.services.annotation import AnnotationService

        await _set_task_id(job_id, self.request.id)
        await _mark_job(job_id, "RUNNING")
        await _set_progress(job_id, 5, "Loading annotation engine …")

        try:
            import app.db.session as db_session
            async with db_session.SessionLocal() as db:
                repo = AnnotationRepository(db)
                service = AnnotationService(repo)
                # Progress hook: the background task method updates result inline
                await service.run_annotation_background_task(
                    job_id=job_id,
                    dataset_id=dataset_id,
                )
        except (IOError, RuntimeError) as exc:
            logger.warning(f"[run_annotation] Transient error, retrying: {exc}")
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)
        except SoftTimeLimitExceeded:
            await _mark_job(job_id, "FAILED", {"error_message": "Task exceeded time limit"})
            raise

    _run_async(_run())
    return {"job_id": job_id, "status": "COMPLETED"}


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — Augmentation (Albumentations / MixUp / CutMix / Mosaic)
# ──────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    base=ManagedTask,
    name="app.workers.celery_tasks.run_augmentation",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    soft_time_limit=1200,
)
def run_augmentation(
    self,
    job_id: int,
    dataset_id: int,
    method: str,
    version_tag: str,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply an augmentation pipeline and persist augmented images + annotations."""

    async def _run():
        from app.services.augmentation import AugmentationService

        await _set_task_id(job_id, self.request.id)
        await _mark_job(job_id, "RUNNING")
        await _set_progress(job_id, 5, f"Starting '{method}' augmentation …")

        try:
            import app.db.session as db_session
            async with db_session.SessionLocal() as db:
                service = AugmentationService()
                await _set_progress(job_id, 20, "Running augmentation pipeline …")
                result = await service.augment_dataset(
                    db=db,
                    dataset_id=dataset_id,
                    method=method,
                    version_tag=version_tag,
                    description=description,
                )
                await _mark_job(
                    job_id,
                    "COMPLETED",
                    {"result": {**result, "progress": 100, "progress_message": "Augmentation complete"}},
                )
        except (IOError, ValueError) as exc:
            logger.warning(f"[run_augmentation] Transient error, retrying: {exc}")
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)
        except SoftTimeLimitExceeded:
            await _mark_job(job_id, "FAILED", {"error_message": "Task exceeded time limit"})
            raise

    _run_async(_run())
    return {"job_id": job_id, "status": "COMPLETED"}


# ──────────────────────────────────────────────────────────────────────────────
# Task 4 — Class Balancing + Stratified Split
# ──────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    base=ManagedTask,
    name="app.workers.celery_tasks.run_balance_and_split",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    soft_time_limit=900,
)
def run_balance_and_split(
    self,
    job_id: int,
    dataset_id: int,
    oversample: bool,
    imbalance_ratio: float,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    version_tag: str,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyse class distribution, oversample minorities, produce stratified splits."""

    async def _run():
        from app.services.dataset_balancer import DatasetBalancerService

        await _set_task_id(job_id, self.request.id)
        await _mark_job(job_id, "RUNNING")
        await _set_progress(job_id, 5, "Analysing class distribution …")

        try:
            import app.db.session as db_session
            async with db_session.SessionLocal() as db:
                service = DatasetBalancerService()
                await _set_progress(job_id, 30, "Oversampling minority classes …")
                result = await service.balance_and_split(
                    db=db,
                    dataset_id=dataset_id,
                    oversample=oversample,
                    imbalance_ratio=imbalance_ratio,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    version_tag=version_tag,
                    description=description,
                )
                await _mark_job(
                    job_id,
                    "COMPLETED",
                    {"result": {"progress": 100, "progress_message": "Split complete", **result}},
                )
        except (IOError, ValueError) as exc:
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)
        except SoftTimeLimitExceeded:
            await _mark_job(job_id, "FAILED", {"error_message": "Task exceeded time limit"})
            raise

    _run_async(_run())
    return {"job_id": job_id, "status": "COMPLETED"}


# ──────────────────────────────────────────────────────────────────────────────
# Task 5 — Dataset Export (YOLO / COCO / VOC / Classification / Segmentation)
# ──────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    base=ManagedTask,
    name="app.workers.celery_tasks.run_export",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    soft_time_limit=1200,
)
def run_export(
    self,
    job_id: int,
    dataset_id: int,
    export_format: str,
    version_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Package the dataset as a ZIP archive in the requested format.

    The resulting file is written to EXPORT_TEMP_DIR/{job_id}.zip.
    The download endpoint reads it back and streams it to the client.
    """

    async def _run():
        from app.services.export import DatasetExportService
        from app.core.config import settings

        await _set_task_id(job_id, self.request.id)
        await _mark_job(job_id, "RUNNING")
        await _set_progress(job_id, 5, f"Preparing '{export_format}' export …")

        try:
            import app.db.session as db_session
            async with db_session.SessionLocal() as db:
                service = DatasetExportService()
                await _set_progress(job_id, 20, "Compiling images and annotations …")
                zip_bytes = await service.export_dataset(
                    db=db,
                    dataset_id=dataset_id,
                    export_format=export_format,
                    version_tag=version_tag,
                )

            # Write ZIP to temp file (avoids storing large blobs in Redis)
            export_dir = settings.EXPORT_TEMP_DIR
            os.makedirs(export_dir, exist_ok=True)
            out_path = os.path.join(export_dir, f"{job_id}.zip")
            with open(out_path, "wb") as f:
                f.write(zip_bytes)

            await _set_progress(job_id, 90, "Finalising …")
            await _mark_job(
                job_id,
                "COMPLETED",
                {
                    "result": {
                        "progress": 100,
                        "progress_message": "Export complete",
                        "export_format": export_format,
                        "file_path": out_path,
                        "file_size_bytes": len(zip_bytes),
                    }
                },
            )
        except (IOError, ValueError) as exc:
            logger.warning(f"[run_export] Transient error, retrying: {exc}")
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)
        except SoftTimeLimitExceeded:
            await _mark_job(job_id, "FAILED", {"error_message": "Task exceeded time limit"})
            raise

    _run_async(_run())
    return {"job_id": job_id, "status": "COMPLETED"}

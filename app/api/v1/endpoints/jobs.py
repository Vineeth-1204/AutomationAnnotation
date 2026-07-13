"""
Dedicated job management endpoints.

GET  /api/v1/jobs/{job_id}              — poll status + progress
POST /api/v1/jobs/{job_id}/cancel       — revoke Celery task and mark CANCELLED
GET  /api/v1/jobs/{job_id}/download     — stream the export ZIP file (export jobs only)
"""
import os
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.core.deps import (
    get_current_user,
    get_dataset_service,
    get_processing_job_repository,
)
from app.core.exceptions import NotFoundError, ForbiddenError, ValidationError
from app.models.user import User
from app.schemas.processing_job import ProcessingJobResponse
from app.services.dataset import DatasetService

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# GET /jobs/{job_id}  — poll status + live progress
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{job_id}", response_model=ProcessingJobResponse)
async def get_job_status(
    job_id: int,
    current_user: User = Depends(get_current_user),
    job_repo=Depends(get_processing_job_repository),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> ProcessingJobResponse:
    """
    Poll the status of any long-running job.

    The `result` field carries live progress while the job is RUNNING:
        {"progress": 45, "progress_message": "Annotating image 9/20", ...}

    Once COMPLETED, `result` contains the final statistics.
    If FAILED, `error_message` carries the exception details.
    """
    job = await job_repo.get(job_id)
    if not job:
        raise NotFoundError("Job not found")

    dataset = await dataset_service.get_by_id(job.dataset_id)
    if not dataset:
        raise NotFoundError("Associated dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    return job


# ──────────────────────────────────────────────────────────────────────────────
# POST /jobs/{job_id}/cancel  — revoke Celery task + mark CANCELLED
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/{job_id}/cancel", response_model=ProcessingJobResponse)
async def cancel_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    job_repo=Depends(get_processing_job_repository),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> ProcessingJobResponse:
    """
    Cancel a running job.

    Sends a SIGTERM to the Celery worker executing the task (via broker control
    channel) and immediately updates the DB status to CANCELLED.

    Jobs in COMPLETED / FAILED / CANCELLED state cannot be cancelled again.
    """
    job = await job_repo.get(job_id)
    if not job:
        raise NotFoundError("Job not found")

    dataset = await dataset_service.get_by_id(job.dataset_id)
    if not dataset:
        raise NotFoundError("Associated dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    terminal_states = {"COMPLETED", "FAILED", "CANCELLED"}
    if job.status in terminal_states:
        raise ValidationError(
            f"Job is already in terminal state '{job.status}' and cannot be cancelled."
        )

    # Revoke the Celery task — terminate=True sends SIGTERM to the worker process
    if job.task_id:
        try:
            from app.celery_app import celery_app
            celery_app.control.revoke(job.task_id, terminate=True, signal="SIGTERM")
        except Exception:
            # If broker is unreachable, still mark the DB job as cancelled
            pass

    from datetime import datetime, timezone
    updated_job = await job_repo.update(job, {
        "status": "CANCELLED",
        "completed_at": datetime.now(timezone.utc),
        "error_message": "Cancelled by user request",
    })
    return updated_job


# ──────────────────────────────────────────────────────────────────────────────
# GET /jobs/{job_id}/download  — stream export ZIP (export jobs only)
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{job_id}/download")
async def download_export(
    job_id: int,
    current_user: User = Depends(get_current_user),
    job_repo=Depends(get_processing_job_repository),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> FileResponse:
    """
    Stream the ZIP archive produced by a completed export job.

    Only available when job_type == 'export' and status == 'COMPLETED'.
    The file is served directly from the filesystem (EXPORT_TEMP_DIR).
    """
    job = await job_repo.get(job_id)
    if not job:
        raise NotFoundError("Job not found")

    dataset = await dataset_service.get_by_id(job.dataset_id)
    if not dataset:
        raise NotFoundError("Associated dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    if job.job_type != "export":
        raise ValidationError("This endpoint is only available for export jobs.")
    if job.status != "COMPLETED":
        raise ValidationError(f"Export job is not yet complete (status: {job.status}).")

    file_path = (job.result or {}).get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise NotFoundError("Export file not found on disk. It may have expired.")

    export_format = (job.result or {}).get("export_format", "dataset")
    filename = f"dataset_{job.dataset_id}_{export_format}.zip"

    return FileResponse(
        path=file_path,
        media_type="application/zip",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

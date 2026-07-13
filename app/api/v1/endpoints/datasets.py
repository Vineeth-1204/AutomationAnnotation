"""
Dataset-scoped API endpoints.

Long-running operations are dispatched to Celery workers via .delay().
The endpoint returns a ProcessingJobResponse immediately (HTTP 202).
Clients poll GET /api/v1/jobs/{job_id} for live progress and final result.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, status, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import (
    get_db,
    get_current_user,
    get_dataset_service,
    get_image_service,
    get_processing_job_repository,
    get_analytics_service,
    get_export_service,
)
from app.core.exceptions import NotFoundError, ForbiddenError
from app.models.user import User
from app.schemas.processing_job import ProcessingJobResponse
from app.schemas.dataset_split import BalanceAndSplitRequest, BalanceAndSplitResponse
from app.schemas.dataset_augment import AugmentationRequest, AugmentationResponse
from app.schemas.dataset_analytics import DatasetAnalyticsResponse
from app.services.analytics import DatasetAnalyticsService
from app.services.export import DatasetExportService
from app.services.dataset import DatasetService
from app.services.image import ImageService

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _create_job(db, job_type: str, dataset_id: int, creator_id: int, params: dict) -> "ProcessingJob":
    """Persist a new ProcessingJob with PENDING status and return it."""
    from app.repositories.processing_job import ProcessingJobRepository
    repo = ProcessingJobRepository(db)
    job = await repo.create({
        "job_type": job_type,
        "status": "PENDING",
        "parameters": params,
        "dataset_id": dataset_id,
        "creator_id": creator_id,
    })
    return job


# ──────────────────────────────────────────────────────────────────────────────
# POST /{dataset_id}/collect  — image scraping + pipeline filtering
# ──────────────────────────────────────────────────────────────────────────────

class ImageCollectionRequest(BaseModel):
    queries: List[str]
    limit_per_query: int = 5


@router.post("/{dataset_id}/collect", response_model=ProcessingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def collect_images(
    dataset_id: int,
    request: ImageCollectionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> ProcessingJobResponse:
    """Start an asynchronous image collection + pipeline filtering job for a dataset."""
    dataset = await dataset_service.get_by_id(dataset_id)
    if not dataset:
        raise NotFoundError("Dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    job = await _create_job(
        db,
        job_type="image_collection",
        dataset_id=dataset_id,
        creator_id=current_user.id,
        params={"queries": request.queries, "limit_per_query": request.limit_per_query},
    )
    await db.commit()
    await db.refresh(job)

    # Dispatch to Celery worker
    from app.workers.celery_tasks import run_image_collection
    run_image_collection.delay(
        job_id=job.id,
        dataset_id=dataset_id,
        queries=request.queries,
        limit_per_query=request.limit_per_query,
    )

    return job


# ──────────────────────────────────────────────────────────────────────────────
# POST /{dataset_id}/balance-and-split  — class balancing + stratified splits
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/{dataset_id}/balance-and-split", response_model=ProcessingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def balance_and_split_dataset(
    dataset_id: int,
    request: BalanceAndSplitRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> ProcessingJobResponse:
    """Dispatch a class-balancing + stratified-split job to the Celery worker."""
    dataset = await dataset_service.get_by_id(dataset_id)
    if not dataset:
        raise NotFoundError("Dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    job = await _create_job(
        db,
        job_type="balance_and_split",
        dataset_id=dataset_id,
        creator_id=current_user.id,
        params={
            "oversample": request.oversample,
            "imbalance_ratio": request.imbalance_ratio,
            "train_ratio": request.train_ratio,
            "val_ratio": request.val_ratio,
            "test_ratio": request.test_ratio,
            "version_tag": request.version_tag,
            "description": request.description,
        },
    )
    await db.commit()
    await db.refresh(job)

    from app.workers.celery_tasks import run_balance_and_split
    run_balance_and_split.delay(
        job_id=job.id,
        dataset_id=dataset_id,
        oversample=request.oversample,
        imbalance_ratio=request.imbalance_ratio,
        train_ratio=request.train_ratio,
        val_ratio=request.val_ratio,
        test_ratio=request.test_ratio,
        version_tag=request.version_tag,
        description=request.description,
    )

    return job


# ──────────────────────────────────────────────────────────────────────────────
# POST /{dataset_id}/augment  — augmentation pipeline
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/{dataset_id}/augment", response_model=ProcessingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def augment_dataset(
    dataset_id: int,
    request: AugmentationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> ProcessingJobResponse:
    """Dispatch an augmentation pipeline job (albumentations / mixup / cutmix / mosaic) to Celery."""
    dataset = await dataset_service.get_by_id(dataset_id)
    if not dataset:
        raise NotFoundError("Dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    job = await _create_job(
        db,
        job_type="augmentation",
        dataset_id=dataset_id,
        creator_id=current_user.id,
        params={"method": request.method, "version_tag": request.version_tag},
    )
    await db.commit()
    await db.refresh(job)

    from app.workers.celery_tasks import run_augmentation
    run_augmentation.delay(
        job_id=job.id,
        dataset_id=dataset_id,
        method=request.method,
        version_tag=request.version_tag,
        description=request.description,
    )

    return job


# ──────────────────────────────────────────────────────────────────────────────
# POST /{dataset_id}/export  — dataset export + ZIP packaging
# ──────────────────────────────────────────────────────────────────────────────

class ExportRequest(BaseModel):
    export_format: str  # yolo | coco | pascal_voc | classification | segmentation
    version_tag: Optional[str] = None


@router.post("/{dataset_id}/export", response_model=ProcessingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def export_dataset(
    dataset_id: int,
    request: ExportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> ProcessingJobResponse:
    """
    Dispatch an export job to Celery.

    Poll GET /api/v1/jobs/{job_id} until status == COMPLETED, then
    download the ZIP from GET /api/v1/jobs/{job_id}/download.
    """
    valid_formats = {"yolo", "coco", "pascal_voc", "classification", "segmentation"}
    if request.export_format not in valid_formats:
        from app.core.exceptions import ValidationError
        raise ValidationError(f"export_format must be one of: {', '.join(sorted(valid_formats))}")

    dataset = await dataset_service.get_by_id(dataset_id)
    if not dataset:
        raise NotFoundError("Dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    job = await _create_job(
        db,
        job_type="export",
        dataset_id=dataset_id,
        creator_id=current_user.id,
        params={"export_format": request.export_format, "version_tag": request.version_tag},
    )
    await db.commit()
    await db.refresh(job)

    from app.workers.celery_tasks import run_export
    run_export.delay(
        job_id=job.id,
        dataset_id=dataset_id,
        export_format=request.export_format,
        version_tag=request.version_tag,
    )

    return job


# ──────────────────────────────────────────────────────────────────────────────
# GET /{dataset_id}/analytics  — JSON analytics
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{dataset_id}/analytics", response_model=DatasetAnalyticsResponse)
async def get_dataset_analytics(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    analytics_service: DatasetAnalyticsService = Depends(get_analytics_service),
    db: AsyncSession = Depends(get_db),
) -> DatasetAnalyticsResponse:
    """Retrieve structured metrics and class distributions for a dataset."""
    dataset = await dataset_service.get_by_id(dataset_id)
    if not dataset:
        raise NotFoundError("Dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    return await analytics_service.get_dataset_analytics(db, dataset_id)


# ──────────────────────────────────────────────────────────────────────────────
# GET /{dataset_id}/analytics/pdf  — PDF report (synchronous, fast)
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{dataset_id}/analytics/pdf")
async def get_dataset_analytics_pdf(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    analytics_service: DatasetAnalyticsService = Depends(get_analytics_service),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Retrieve a premium PDF document report of the dataset analytics."""
    dataset = await dataset_service.get_by_id(dataset_id)
    if not dataset:
        raise NotFoundError("Dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    analytics_data = await analytics_service.get_dataset_analytics(db, dataset_id)
    pdf_bytes = analytics_service.generate_pdf_report(analytics_data)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=dataset_{dataset_id}_analytics.pdf"
        },
    )

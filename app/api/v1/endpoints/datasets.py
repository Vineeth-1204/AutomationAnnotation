from typing import List
from fastapi import APIRouter, Depends, BackgroundTasks, status, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import (
    get_db,
    get_current_user,
    get_dataset_service,
    get_image_service,
    get_processing_job_repository,
    get_balancer_service,
    get_augmentation_service,
    get_analytics_service,
    get_export_service
)
from app.core.exceptions import NotFoundError, ForbiddenError
from app.models.user import User
from app.schemas.processing_job import ProcessingJobResponse
from app.schemas.dataset_split import BalanceAndSplitRequest, BalanceAndSplitResponse
from app.schemas.dataset_augment import AugmentationRequest, AugmentationResponse
from app.schemas.dataset_analytics import DatasetAnalyticsResponse
from app.services.dataset_balancer import DatasetBalancerService
from app.services.augmentation import AugmentationService
from app.services.analytics import DatasetAnalyticsService
from app.services.export import DatasetExportService
from app.services.dataset import DatasetService
from app.services.image import ImageService

router = APIRouter()

class ImageCollectionRequest(BaseModel):
    queries: List[str]
    limit_per_query: int = 5

@router.post("/{dataset_id}/collect", response_model=ProcessingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def collect_images(
    dataset_id: int,
    request: ImageCollectionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    image_service: ImageService = Depends(get_image_service),
) -> ProcessingJobResponse:
    """Start an asynchronous image collection job for a dataset."""
    dataset = await dataset_service.get_by_id(dataset_id)
    if not dataset:
        raise NotFoundError("Dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    # Start the job DB entry (status: PENDING)
    job = await image_service.start_collection_job(
        db=db,
        dataset_id=dataset_id,
        queries=request.queries,
        limit_per_query=request.limit_per_query,
        creator_id=current_user.id
    )
    await db.commit()
    await db.refresh(job)

    # Dispatch to background task worker
    background_tasks.add_task(
        image_service.run_collection_background_task,
        job_id=job.id,
        dataset_id=dataset_id,
        queries=request.queries,
        limit_per_query=request.limit_per_query
    )

    return job

@router.get("/jobs/{job_id}", response_model=ProcessingJobResponse)
async def get_job_status(
    job_id: int,
    current_user: User = Depends(get_current_user),
    job_repo = Depends(get_processing_job_repository),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> ProcessingJobResponse:
    """Get status of an image collection/processing job."""
    job = await job_repo.get(job_id)
    if not job:
        raise NotFoundError("Job not found")

    # Verify ownership of the associated dataset
    dataset = await dataset_service.get_by_id(job.dataset_id)
    if not dataset:
        raise NotFoundError("Associated dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    return job

@router.post("/{dataset_id}/balance-and-split", response_model=BalanceAndSplitResponse)
async def balance_and_split_dataset(
    dataset_id: int,
    request: BalanceAndSplitRequest,
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    balancer_service: DatasetBalancerService = Depends(get_balancer_service),
    db: AsyncSession = Depends(get_db)
) -> BalanceAndSplitResponse:
    """Analyze class distribution, oversample minority classes using horizontal flips, and generate stratified splits."""
    dataset = await dataset_service.get_by_id(dataset_id)
    if not dataset:
        raise NotFoundError("Dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    result = await balancer_service.balance_and_split(
        db=db,
        dataset_id=dataset_id,
        oversample=request.oversample,
        imbalance_ratio=request.imbalance_ratio,
        train_ratio=request.train_ratio,
        val_ratio=request.val_ratio,
        test_ratio=request.test_ratio,
        version_tag=request.version_tag,
        description=request.description
    )
    return result

@router.post("/{dataset_id}/augment", response_model=AugmentationResponse)
async def augment_dataset(
    dataset_id: int,
    request: AugmentationRequest,
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    augmentation_service: AugmentationService = Depends(get_augmentation_service),
    db: AsyncSession = Depends(get_db)
) -> AugmentationResponse:
    """Run an augmentation pipeline (albumentations, mixup, cutmix, or mosaic) on the dataset."""
    dataset = await dataset_service.get_by_id(dataset_id)
    if not dataset:
        raise NotFoundError("Dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    result = await augmentation_service.augment_dataset(
        db=db,
        dataset_id=dataset_id,
        method=request.method,
        version_tag=request.version_tag,
        description=request.description
    )
    return result

@router.get("/{dataset_id}/analytics", response_model=DatasetAnalyticsResponse)
async def get_dataset_analytics(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    analytics_service: DatasetAnalyticsService = Depends(get_analytics_service),
    db: AsyncSession = Depends(get_db)
) -> DatasetAnalyticsResponse:
    """Retrieve structured metrics and class distributions for a dataset."""
    dataset = await dataset_service.get_by_id(dataset_id)
    if not dataset:
        raise NotFoundError("Dataset not found")
    if dataset.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    result = await analytics_service.get_dataset_analytics(db, dataset_id)
    return result

@router.get("/{dataset_id}/analytics/pdf")
async def get_dataset_analytics_pdf(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    analytics_service: DatasetAnalyticsService = Depends(get_analytics_service),
    db: AsyncSession = Depends(get_db)
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
        }
    )

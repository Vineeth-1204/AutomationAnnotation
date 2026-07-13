from typing import List
from fastapi import APIRouter, Depends, status, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_current_user, get_annotation_service, get_dataset_service
from app.core.exceptions import ForbiddenError, NotFoundError
from app.models.user import User
from app.models.image import Image
from app.models.annotation import Annotation
from app.schemas.processing_job import ProcessingJobResponse
from app.schemas.annotation import AnnotationResponse
from app.services.annotation import AnnotationService
from app.services.dataset import DatasetService

router = APIRouter()

@router.post("/{dataset_id}/annotate", response_model=ProcessingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def annotate_dataset(
    dataset_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    annotation_service: AnnotationService = Depends(get_annotation_service),
    db: AsyncSession = Depends(get_db)
) -> ProcessingJobResponse:
    """Trigger background auto-annotation for all images in the dataset."""
    dataset = await dataset_service.get_by_id(dataset_id)
    if not dataset:
        raise NotFoundError("Dataset not found")
    
    # Check project ownership
    project = dataset.project
    if project.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    job = await annotation_service.start_annotation_job(
        db=db,
        dataset_id=dataset_id,
        creator_id=current_user.id
    )
    await db.commit()
    await db.refresh(job)

    background_tasks.add_task(
        annotation_service.run_annotation_background_task,
        job_id=job.id,
        dataset_id=dataset_id
    )

    return job

@router.get("/{dataset_id}/annotations", response_model=List[AnnotationResponse])
async def get_dataset_annotations(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    db: AsyncSession = Depends(get_db)
) -> List[AnnotationResponse]:
    """Retrieve all annotations generated for a dataset."""
    dataset = await dataset_service.get_by_id(dataset_id)
    if not dataset:
        raise NotFoundError("Dataset not found")
    
    # Check ownership
    project = dataset.project
    if project.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")

    res = await db.execute(
        select(Annotation)
        .join(Image)
        .where(Image.dataset_id == dataset_id)
    )
    annotations = res.scalars().all()
    return annotations

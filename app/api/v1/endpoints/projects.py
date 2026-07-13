from typing import List, Any, Union
from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.deps import get_project_service, get_current_user, get_db
from app.core.exceptions import ForbiddenError, NotFoundError, BadRequestError
from app.models.user import User
from app.models.dataset import Dataset
from app.models.processing_job import ProcessingJob
from app.schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse
from app.schemas.annotation_class import AnnotationClassesSubmission, AnnotationClassConfig
from app.services.project import ProjectService

router = APIRouter()

@router.get("/", response_model=List[ProjectResponse])
async def read_projects(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service),
) -> List[ProjectResponse]:
    """Retrieve projects owned by the current user."""
    if current_user.is_superuser:
        return await project_service.get_all(skip=skip, limit=limit)
    return await project_service.get_by_owner(owner_id=current_user.id, skip=skip, limit=limit)

@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_in: ProjectCreate,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    """Create a new project owned by the current user."""
    return await project_service.create_with_owner(project_in=project_in, owner_id=current_user.id)

@router.get("/{id}", response_model=ProjectResponse)
async def read_project(
    id: int,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    """Get project by ID."""
    project = await project_service.get_by_id(id)
    if not project:
        raise NotFoundError("Project not found")
    if project.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")
    return project

@router.put("/{id}", response_model=ProjectResponse)
async def update_project(
    id: int,
    project_in: ProjectUpdate,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    """Update a project by ID."""
    project = await project_service.get_by_id(id)
    if not project:
        raise NotFoundError("Project not found")
    if project.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")
    return await project_service.update(project, project_in.model_dump(exclude_unset=True))

@router.delete("/{id}", response_model=ProjectResponse)
async def delete_project(
    id: int,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    """Delete a project by ID."""
    project = await project_service.get_by_id(id)
    if not project:
        raise NotFoundError("Project not found")
    if project.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")
    return await project_service.delete(id)

@router.post("/{project_id}/annotation-classes", response_model=ProjectResponse)
async def submit_annotation_classes(
    project_id: int,
    submission: AnnotationClassesSubmission,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service),
    db: AsyncSession = Depends(get_db),
) -> ProjectResponse:
    """Submit annotation classes configuration for a project after image collection is complete."""
    project = await project_service.get_by_id(project_id)
    if not project:
        raise NotFoundError("Project not found")
    if project.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")
        
    # Check if all collection jobs are completed
    datasets_result = await db.execute(
        select(Dataset).where(Dataset.project_id == project.id)
    )
    datasets = datasets_result.scalars().all()
    if not datasets:
        raise BadRequestError("No datasets found for this project. Please create a dataset and trigger image collection first.")
        
    dataset_ids = [d.id for d in datasets]
    jobs_result = await db.execute(
        select(ProcessingJob).where(
            ProcessingJob.dataset_id.in_(dataset_ids),
            ProcessingJob.job_type == "image_collection"
        )
    )
    jobs = jobs_result.scalars().all()
    if not jobs:
        raise BadRequestError("No collection jobs found for this project. Please trigger image collection first.")
        
    incomplete_jobs = [j for j in jobs if j.status != "COMPLETED"]
    if incomplete_jobs:
        raise BadRequestError(f"Image collection is not complete. Incomplete jobs: {len(incomplete_jobs)}")
        
    classes_data = [c.model_dump() for c in submission.classes]
    
    updated_project = await project_service.update(project, {"annotation_classes": classes_data})
    await db.commit()
    return updated_project

@router.get("/{project_id}/annotation-classes", response_model=List[Union[str, AnnotationClassConfig]])
async def get_annotation_classes(
    project_id: int,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service),
) -> List[Any]:
    """Retrieve annotation classes configuration for a project."""
    project = await project_service.get_by_id(project_id)
    if not project:
        raise NotFoundError("Project not found")
    if project.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")
    return project.annotation_classes

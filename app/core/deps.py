from typing import AsyncGenerator
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from app.core.security import decode_token
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import AuthenticationError, ForbiddenError
from app.db.session import SessionLocal

# To be defined in subsequent files
from app.models.user import User
from app.repositories.user import UserRepository
from app.services.user import UserService
from app.repositories.item import ItemRepository
from app.services.item import ItemService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

async def get_user_repository(db: AsyncSession = Depends(get_db)) -> UserRepository:
    return UserRepository(db)

async def get_user_service(
    user_repo: UserRepository = Depends(get_user_repository)
) -> UserService:
    return UserService(user_repo)

from app.services.auth import AuthService

async def get_auth_service(
    user_service: UserService = Depends(get_user_service)
) -> AuthService:
    return AuthService(user_service)

async def get_item_repository(db: AsyncSession = Depends(get_db)) -> ItemRepository:
    return ItemRepository(db)

from app.repositories.project import ProjectRepository
from app.services.project import ProjectService

async def get_project_repository(db: AsyncSession = Depends(get_db)) -> ProjectRepository:
    return ProjectRepository(db)

async def get_project_service(
    project_repo: ProjectRepository = Depends(get_project_repository)
) -> ProjectService:
    return ProjectService(project_repo)

from app.services.prompt import PromptService

async def get_prompt_service() -> PromptService:
    return PromptService()

from app.repositories.dataset import DatasetRepository
from app.services.dataset import DatasetService
from app.repositories.image import ImageRepository
from app.services.image import ImageService
from app.repositories.processing_job import ProcessingJobRepository

async def get_dataset_repository(db: AsyncSession = Depends(get_db)) -> DatasetRepository:
    return DatasetRepository(db)

async def get_dataset_service(
    dataset_repo: DatasetRepository = Depends(get_dataset_repository)
) -> DatasetService:
    return DatasetService(dataset_repo)

async def get_image_repository(db: AsyncSession = Depends(get_db)) -> ImageRepository:
    return ImageRepository(db)

async def get_image_service(
    image_repo: ImageRepository = Depends(get_image_repository)
) -> ImageService:
    return ImageService(image_repo)

from app.services.pipeline import ImagePipelineService

async def get_pipeline_service() -> ImagePipelineService:
    return ImagePipelineService()

from app.repositories.annotation import AnnotationRepository
from app.services.annotation import AnnotationService

async def get_annotation_repository(db: AsyncSession = Depends(get_db)) -> AnnotationRepository:
    return AnnotationRepository(db)

async def get_annotation_service(
    annotation_repo: AnnotationRepository = Depends(get_annotation_repository)
) -> AnnotationService:
    return AnnotationService(annotation_repo)

from app.services.dataset_balancer import DatasetBalancerService

async def get_balancer_service() -> DatasetBalancerService:
    return DatasetBalancerService()

from app.services.augmentation import AugmentationService

async def get_augmentation_service() -> AugmentationService:
    return AugmentationService()

from app.services.analytics import DatasetAnalyticsService

async def get_analytics_service() -> DatasetAnalyticsService:
    return DatasetAnalyticsService()

from app.services.export import DatasetExportService

async def get_export_service() -> DatasetExportService:
    return DatasetExportService()

async def get_processing_job_repository(db: AsyncSession = Depends(get_db)) -> ProcessingJobRepository:
    return ProcessingJobRepository(db)

async def get_item_service(
    item_repo: ItemRepository = Depends(get_item_repository)
) -> ItemService:
    return ItemService(item_repo)

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    user_service: UserService = Depends(get_user_service)
) -> User:
    try:
        payload = decode_token(token, "access")
        user_id = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Invalid token: missing subject field")
    except Exception:
        raise AuthenticationError("Could not validate credentials")
        
    user = await user_service.get_by_id(int(user_id))
    if not user:
        raise AuthenticationError("User not found")
    if not user.is_active:
        raise AuthenticationError("Inactive user")
        
    return user

from typing import List

class RoleChecker:
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        if current_user.is_superuser:
            return current_user
        if current_user.role not in self.allowed_roles:
            raise ForbiddenError("Not enough permissions (role not allowed)")
        return current_user

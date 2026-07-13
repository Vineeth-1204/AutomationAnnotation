from typing import List
from app.models.project import Project
from app.repositories.project import ProjectRepository
from app.schemas.project import ProjectCreate
from app.services.base import BaseService

class ProjectService(BaseService[Project]):
    def __init__(self, repository: ProjectRepository):
        super().__init__(repository)
        self.repository = repository

    async def get_by_owner(
        self, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[Project]:
        return await self.repository.get_multi_by_owner(
            owner_id=owner_id, skip=skip, limit=limit
        )

    async def create_with_owner(
        self, *, project_in: ProjectCreate, owner_id: int
    ) -> Project:
        project_data = project_in.model_dump()
        project_data["owner_id"] = owner_id
        return await self.repository.create(project_data)

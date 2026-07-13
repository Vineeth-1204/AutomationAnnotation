from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.project import Project
from app.repositories.base import BaseRepository

class ProjectRepository(BaseRepository[Project]):
    def __init__(self, db: AsyncSession):
        super().__init__(Project, db)

    async def get_multi_by_owner(
        self, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[Project]:
        query = (
            select(self.model)
            .where(self.model.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

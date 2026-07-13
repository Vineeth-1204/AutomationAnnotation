from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.item import Item
from app.repositories.base import BaseRepository

class ItemRepository(BaseRepository[Item]):
    def __init__(self, db: AsyncSession):
        super().__init__(Item, db)

    async def get_multi_by_owner(
        self, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[Item]:
        query = (
            select(self.model)
            .where(self.model.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

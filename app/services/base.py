from typing import Generic, TypeVar, List, Optional, Any
from app.repositories.base import BaseRepository, ModelType

class BaseService(Generic[ModelType]):
    def __init__(self, repository: BaseRepository[ModelType]):
        self.repository = repository

    async def get_by_id(self, id: Any) -> Optional[ModelType]:
        return await self.repository.get(id)

    async def get_all(self, *, skip: int = 0, limit: int = 100) -> List[ModelType]:
        return await self.repository.get_multi(skip=skip, limit=limit)

    async def create(self, obj_in: dict) -> ModelType:
        return await self.repository.create(obj_in)

    async def update(self, db_obj: ModelType, obj_in: dict) -> ModelType:
        return await self.repository.update(db_obj, obj_in)

    async def delete(self, id: Any) -> Optional[ModelType]:
        return await self.repository.delete(id)

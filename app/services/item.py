from typing import List
from app.models.item import Item
from app.repositories.item import ItemRepository
from app.schemas.item import ItemCreate, ItemUpdate
from app.services.base import BaseService

class ItemService(BaseService[Item]):
    def __init__(self, repository: ItemRepository):
        super().__init__(repository)
        self.repository = repository

    async def get_by_owner(
        self, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[Item]:
        return await self.repository.get_multi_by_owner(
            owner_id=owner_id, skip=skip, limit=limit
        )

    async def create_with_owner(
        self, *, item_in: ItemCreate, owner_id: int
    ) -> Item:
        item_data = item_in.model_dump()
        item_data["owner_id"] = owner_id
        return await self.repository.create(item_data)

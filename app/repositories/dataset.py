from sqlalchemy.ext.asyncio import AsyncSession
from app.models.dataset import Dataset
from app.repositories.base import BaseRepository

class DatasetRepository(BaseRepository[Dataset]):
    def __init__(self, db: AsyncSession):
        super().__init__(Dataset, db)

from sqlalchemy.ext.asyncio import AsyncSession
from app.models.image import Image
from app.repositories.base import BaseRepository

class ImageRepository(BaseRepository[Image]):
    def __init__(self, db: AsyncSession):
        super().__init__(Image, db)

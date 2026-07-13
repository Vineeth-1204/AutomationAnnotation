from sqlalchemy.ext.asyncio import AsyncSession
from app.models.annotation import Annotation
from app.repositories.base import BaseRepository

class AnnotationRepository(BaseRepository[Annotation]):
    def __init__(self, db: AsyncSession):
        super().__init__(Annotation, db)

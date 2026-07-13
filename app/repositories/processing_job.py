from sqlalchemy.ext.asyncio import AsyncSession
from app.models.processing_job import ProcessingJob
from app.repositories.base import BaseRepository

class ProcessingJobRepository(BaseRepository[ProcessingJob]):
    def __init__(self, db: AsyncSession):
        super().__init__(ProcessingJob, db)

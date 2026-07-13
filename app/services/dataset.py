from app.models.dataset import Dataset
from app.repositories.dataset import DatasetRepository
from app.services.base import BaseService

class DatasetService(BaseService[Dataset]):
    def __init__(self, repository: DatasetRepository):
        super().__init__(repository)
        self.repository = repository

from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import String, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.base import Base

if TYPE_CHECKING:
    from app.models.dataset import Dataset
    from app.models.dataset_statistics import DatasetStatistics

class DatasetVersion(Base):
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_tag: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    version_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    dataset: Mapped["Dataset"] = relationship(back_populates="versions")
    statistics: Mapped[List["DatasetStatistics"]] = relationship(back_populates="version", cascade="all, delete-orphan")

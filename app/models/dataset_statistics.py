from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import Integer, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.base import Base

if TYPE_CHECKING:
    from app.models.dataset import Dataset
    from app.models.dataset_version import DatasetVersion

class DatasetStatistics(Base):
    __tablename__ = "dataset_statistics"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id: Mapped[Optional[int]] = mapped_column(ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=True, index=True)
    num_images: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_annotations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    class_distribution: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    dataset: Mapped["Dataset"] = relationship(back_populates="statistics")
    version: Mapped[Optional["DatasetVersion"]] = relationship(back_populates="statistics")

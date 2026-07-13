from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import String, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.base import Base

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.project import Project
    from app.models.image import Image
    from app.models.dataset_version import DatasetVersion
    from app.models.dataset_statistics import DatasetStatistics
    from app.models.processing_job import ProcessingJob

class Dataset(Base):
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    project: Mapped["Project"] = relationship(back_populates="datasets")
    owner: Mapped["User"] = relationship(back_populates="datasets")
    images: Mapped[List["Image"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    versions: Mapped[List["DatasetVersion"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    statistics: Mapped[List["DatasetStatistics"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    processing_jobs: Mapped[List["ProcessingJob"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")

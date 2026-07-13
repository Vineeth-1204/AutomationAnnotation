import enum
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import String, Boolean, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.base import Base

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"
    ANNOTATOR = "annotator"

if TYPE_CHECKING:
    from app.models.item import Item
    from app.models.project import Project
    from app.models.dataset import Dataset
    from app.models.annotation import Annotation
    from app.models.processing_job import ProcessingJob

class User(Base):
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(50), default=UserRole.USER.value, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    items: Mapped[List["Item"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    projects: Mapped[List["Project"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    datasets: Mapped[List["Dataset"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    annotations: Mapped[List["Annotation"]] = relationship(back_populates="creator")
    processing_jobs: Mapped[List["ProcessingJob"]] = relationship(back_populates="creator")

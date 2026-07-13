from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import String, Integer, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.base import Base

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.dataset import Dataset

class Project(Base):
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    dataset_description: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    user_prompt: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    annotation_classes: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    desired_image_count: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    dataset_type: Mapped[str] = mapped_column(String(100), default="object_detection", nullable=False)
    output_format: Mapped[str] = mapped_column(String(100), default="coco", nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    owner: Mapped["User"] = relationship(back_populates="projects")
    datasets: Mapped[List["Dataset"]] = relationship(back_populates="project", cascade="all, delete-orphan")

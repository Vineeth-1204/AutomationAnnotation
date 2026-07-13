from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import String, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.base import Base

if TYPE_CHECKING:
    from app.models.image import Image
    from app.models.user import User

class Annotation(Base):
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    annotation_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    creator_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    image: Mapped["Image"] = relationship(back_populates="annotations")
    creator: Mapped["User"] = relationship(back_populates="annotations")

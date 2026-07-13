from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict

class AnnotationBase(BaseModel):
    image_id: int
    label: str
    annotation_data: dict
    creator_id: Optional[int] = None

class AnnotationCreate(AnnotationBase):
    pass

class AnnotationUpdate(BaseModel):
    label: Optional[str] = None
    annotation_data: Optional[dict] = None
    creator_id: Optional[int] = None

class AnnotationResponse(AnnotationBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

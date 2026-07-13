from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, ConfigDict
from app.schemas.annotation_class import AnnotationClassConfig

class ProjectBase(BaseModel):
    name: str
    dataset_description: Optional[str] = None
    user_prompt: Optional[str] = None
    annotation_classes: List[Union[str, AnnotationClassConfig]] = []
    desired_image_count: int = 100
    dataset_type: str = "object_detection"
    output_format: str = "coco"
    status: str = "pending"

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    dataset_description: Optional[str] = None
    user_prompt: Optional[str] = None
    annotation_classes: Optional[List[Union[str, AnnotationClassConfig]]] = None
    desired_image_count: Optional[int] = None
    dataset_type: Optional[str] = None
    output_format: Optional[str] = None
    status: Optional[str] = None

class ProjectResponse(ProjectBase):
    id: int
    owner_id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

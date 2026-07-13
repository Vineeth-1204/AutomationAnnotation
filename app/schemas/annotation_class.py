from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class AnnotationClassConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    aliases: List[str] = Field(default_factory=list)
    color: Optional[str] = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("Name cannot be empty")
        if not all(c.isalnum() or c == "_" for c in cleaned):
            raise ValueError("Name must be alphanumeric with underscores only")
        return cleaned

class AnnotationClassesSubmission(BaseModel):
    classes: List[AnnotationClassConfig]

    @field_validator("classes")
    def validate_classes(cls, v: List[AnnotationClassConfig]) -> List[AnnotationClassConfig]:
        names = [c.name for c in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate class names are not allowed")
        return v

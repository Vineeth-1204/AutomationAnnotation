from typing import Optional
from pydantic import BaseModel, Field, field_validator

class AugmentationRequest(BaseModel):
    method: str = Field(..., description="Augmentation method: 'albumentations', 'mixup', 'cutmix', 'mosaic'")
    version_tag: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = None

    @field_validator("method")
    def validate_method(cls, v: str) -> str:
        cleaned = v.strip().lower()
        valid = ["albumentations", "mixup", "cutmix", "mosaic"]
        if cleaned not in valid:
            raise ValueError(f"Invalid method. Must be one of: {valid}")
        return cleaned

    @field_validator("version_tag")
    def validate_tag(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("version_tag cannot be empty")
        return cleaned

class AugmentationResponse(BaseModel):
    augmented_images_count: int
    version_id: int
    version_tag: str

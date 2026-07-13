from typing import Dict
from pydantic import BaseModel, Field

class ConfidenceStats(BaseModel):
    average: float = Field(0.0, description="Average confidence score of annotations")
    min_val: float = Field(0.0, description="Minimum confidence score")
    max_val: float = Field(0.0, description="Maximum confidence score")
    median: float = Field(0.0, description="Median confidence score")

class ClassDistributionDetail(BaseModel):
    count: int = Field(..., description="Number of annotations for this class")
    percentage: float = Field(..., description="Percentage of annotations for this class")

class AugmentationStats(BaseModel):
    horizontal_flip: int = Field(0, description="Horizontal flip augmentations count")
    single_image: int = Field(0, description="Albumentations single-image augmentations count")
    mixup: int = Field(0, description="MixUp augmentations count")
    cutmix: int = Field(0, description="CutMix augmentations count")
    mosaic: int = Field(0, description="2x2 Mosaic augmentations count")
    total: int = Field(0, description="Total augmented images count")

class DuplicateRemovalSummary(BaseModel):
    duplicates_removed: int = Field(0, description="Number of duplicate files filtered out")
    blurry_images_removed: int = Field(0, description="Number of blurry files filtered out")

class DatasetAnalyticsResponse(BaseModel):
    dataset_id: int
    dataset_name: str
    image_count: int
    annotation_count: int
    dataset_size_bytes: int
    class_distribution: Dict[str, ClassDistributionDetail]
    confidence_stats: ConfidenceStats
    duplicate_removal_summary: DuplicateRemovalSummary
    augmentation_summary: AugmentationStats

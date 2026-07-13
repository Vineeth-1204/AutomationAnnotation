from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator

class BalanceAndSplitRequest(BaseModel):
    oversample: bool = True
    imbalance_ratio: float = Field(0.5, ge=0.01, le=1.0)
    train_ratio: float = Field(0.70, ge=0.1, le=0.9)
    val_ratio: float = Field(0.15, ge=0.0, le=0.45)
    test_ratio: float = Field(0.15, ge=0.0, le=0.45)
    version_tag: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = None

    @field_validator("train_ratio", "val_ratio", "test_ratio")
    def validate_ratios_sum(cls, v: float, info) -> float:
        # Check sum when test_ratio is validated
        # Note: field_validator order is execution order.
        # We can validate all ratios sum inside a model validator.
        return v

    @field_validator("version_tag")
    def validate_tag(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("version_tag cannot be empty")
        return cleaned

    # Model validator to check that sum of ratios equals 1.0 (with small tolerance)
    @field_validator("test_ratio")
    def check_ratios_total(cls, v: float, info) -> float:
        values = info.data
        train = values.get("train_ratio", 0.70)
        val = values.get("val_ratio", 0.15)
        total = train + val + v
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Ratios must sum to 1.0. Got: {total}")
        return v

class BalanceAndSplitResponse(BaseModel):
    initial_distribution: Dict[str, int]
    imbalance_identified: bool
    minority_classes: List[str]
    augmented_images_count: int
    final_distribution: Dict[str, int]
    splits: Dict[str, List[int]]
    version_id: int
    version_tag: str

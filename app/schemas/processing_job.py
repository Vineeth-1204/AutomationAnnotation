from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict

class ProcessingJobResponse(BaseModel):
    id: int
    job_type: str
    status: str
    parameters: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    dataset_id: int
    creator_id: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

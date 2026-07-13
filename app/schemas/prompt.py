from typing import List
from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str

class PromptAnalysisResponse(BaseModel):
    original_prompt: str
    optimized_search_queries: List[str]
    synonyms: List[str]
    related_keywords: List[str]
    search_variations: List[str]
    detected_classes: List[str]

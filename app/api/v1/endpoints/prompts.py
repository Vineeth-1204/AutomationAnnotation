from fastapi import APIRouter, Depends
from app.core.deps import get_prompt_service, get_current_user
from app.models.user import User
from app.schemas.prompt import PromptRequest, PromptAnalysisResponse
from app.services.prompt import PromptService

router = APIRouter()

@router.post("/understand", response_model=PromptAnalysisResponse)
async def understand_prompt(
    prompt_in: PromptRequest,
    current_user: User = Depends(get_current_user),
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptAnalysisResponse:
    """Analyze a natural language prompt and return optimized queries, synonyms, and variations."""
    return await prompt_service.analyze_prompt(prompt_in.prompt)

from fastapi import APIRouter
from app.api.v1.endpoints import auth, users, items, projects, prompts, datasets, annotations

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(prompts.router, prefix="/prompts", tags=["prompts"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(annotations.router, prefix="/datasets", tags=["datasets"])

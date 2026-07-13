from fastapi import APIRouter, Depends, BackgroundTasks
from app.core.deps import get_user_service, get_current_user
from app.core.exceptions import ForbiddenError, NotFoundError
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.services.user import UserService
from app.workers.tasks import send_welcome_email

router = APIRouter()

@router.post("/", response_model=UserResponse, status_code=201)
async def create_user(
    user_in: UserCreate,
    background_tasks: BackgroundTasks,
    user_service: UserService = Depends(get_user_service),
) -> User:
    """Create a new user (registration)."""
    user = await user_service.register(user_in)
    background_tasks.add_task(send_welcome_email, email=user.email, name=user.full_name)
    return user

@router.get("/me", response_model=UserResponse)
async def read_user_me(current_user: User = Depends(get_current_user)) -> User:
    """Get profile of the current authenticated user."""
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_user_me(
    user_in: UserUpdate,
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> User:
    """Update profile of the current authenticated user."""
    return await user_service.update_user(current_user, user_in)

@router.get("/{user_id}", response_model=UserResponse)
async def read_user_by_id(
    user_id: int,
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> User:
    """Get a specific user by ID. Requires superuser privileges or matching self."""
    if current_user.id != user_id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")
    user = await user_service.get_by_id(user_id)
    if not user:
        raise NotFoundError("User not found")
    return user

from app.core.deps import RoleChecker
from app.models.user import UserRole

@router.delete("/{user_id}", response_model=UserResponse)
async def delete_user(
    user_id: int,
    current_user: User = Depends(RoleChecker([UserRole.ADMIN.value])),
    user_service: UserService = Depends(get_user_service),
) -> User:
    """Delete a user by ID. Only accessible to Admins."""
    user = await user_service.get_by_id(user_id)
    if not user:
        raise NotFoundError("User not found")
    return await user_service.delete(user_id)

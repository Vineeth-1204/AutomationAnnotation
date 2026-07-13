from typing import List
from fastapi import APIRouter, Depends
from app.core.deps import get_item_service, get_current_user
from app.core.exceptions import ForbiddenError, NotFoundError
from app.models.user import User
from app.schemas.item import ItemCreate, ItemUpdate, ItemResponse
from app.services.item import ItemService

router = APIRouter()

@router.get("/", response_model=List[ItemResponse])
async def read_items(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    item_service: ItemService = Depends(get_item_service),
) -> List[ItemResponse]:
    """Retrieve items owned by current user."""
    if current_user.is_superuser:
        return await item_service.get_all(skip=skip, limit=limit)
    return await item_service.get_by_owner(owner_id=current_user.id, skip=skip, limit=limit)

@router.post("/", response_model=ItemResponse, status_code=201)
async def create_item(
    item_in: ItemCreate,
    current_user: User = Depends(get_current_user),
    item_service: ItemService = Depends(get_item_service),
) -> ItemResponse:
    """Create a new item owned by current user."""
    return await item_service.create_with_owner(item_in=item_in, owner_id=current_user.id)

@router.get("/{id}", response_model=ItemResponse)
async def read_item(
    id: int,
    current_user: User = Depends(get_current_user),
    item_service: ItemService = Depends(get_item_service),
) -> ItemResponse:
    """Get item by ID."""
    item = await item_service.get_by_id(id)
    if not item:
        raise NotFoundError("Item not found")
    if item.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")
    return item

@router.put("/{id}", response_model=ItemResponse)
async def update_item(
    id: int,
    item_in: ItemUpdate,
    current_user: User = Depends(get_current_user),
    item_service: ItemService = Depends(get_item_service),
) -> ItemResponse:
    """Update an item by ID."""
    item = await item_service.get_by_id(id)
    if not item:
        raise NotFoundError("Item not found")
    if item.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")
    return await item_service.update(item, item_in.model_dump(exclude_unset=True))

@router.delete("/{id}", response_model=ItemResponse)
async def delete_item(
    id: int,
    current_user: User = Depends(get_current_user),
    item_service: ItemService = Depends(get_item_service),
) -> ItemResponse:
    """Delete an item by ID."""
    item = await item_service.get_by_id(id)
    if not item:
        raise NotFoundError("Item not found")
    if item.owner_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Not enough permissions")
    return await item_service.delete(id)

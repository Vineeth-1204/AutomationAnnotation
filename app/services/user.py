from typing import Optional
from app.core.security import get_password_hash, verify_password
from app.core.exceptions import ConflictError, AuthenticationError
from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.user import UserCreate, UserUpdate
from app.services.base import BaseService

class UserService(BaseService[User]):
    def __init__(self, repository: UserRepository):
        super().__init__(repository)
        self.repository = repository

    async def get_by_email(self, email: str) -> Optional[User]:
        return await self.repository.get_by_email(email)

    async def register(self, user_in: UserCreate) -> User:
        existing_user = await self.get_by_email(user_in.email)
        if existing_user:
            raise ConflictError("A user with this email already exists.")
            
        hashed_password = get_password_hash(user_in.password)
        
        user_data = user_in.model_dump(exclude={"password"})
        user_data["hashed_password"] = hashed_password
        
        return await self.repository.create(user_data)

    async def authenticate(self, email: str, password: str) -> User:
        user = await self.get_by_email(email)
        if not user:
            raise AuthenticationError("Incorrect email or password")
        if not verify_password(password, user.hashed_password):
            raise AuthenticationError("Incorrect email or password")
        if not user.is_active:
            raise AuthenticationError("User is inactive")
        return user

    async def update_user(self, user: User, user_in: UserUpdate) -> User:
        update_data = user_in.model_dump(exclude_unset=True)
        if "password" in update_data and update_data["password"]:
            update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
        return await self.repository.update(user, update_data)

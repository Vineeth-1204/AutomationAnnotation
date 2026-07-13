import logging
from typing import Tuple
from app.core.exceptions import AuthenticationError, NotFoundError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    create_verification_token,
    create_password_reset_token,
    decode_token,
)
from app.models.user import User
from app.services.user import UserService

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self, user_service: UserService):
        self.user_service = user_service

    async def get_tokens(self, user: User) -> Tuple[str, str]:
        access_token = create_access_token(subject=user.id)
        refresh_token = create_refresh_token(subject=user.id)
        return access_token, refresh_token

    async def refresh_access_token(self, refresh_token: str) -> str:
        try:
            payload = decode_token(refresh_token, "refresh")
            user_id = payload.get("sub")
            if not user_id:
                raise AuthenticationError("Invalid refresh token")
        except Exception:
            raise AuthenticationError("Invalid or expired refresh token")

        user = await self.user_service.get_by_id(int(user_id))
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")

        return create_access_token(subject=user.id)

    async def request_verification(self, email: str) -> None:
        user = await self.user_service.get_by_email(email)
        if not user:
            raise NotFoundError("User not found")
        if user.is_verified:
            return

        token = create_verification_token(email)
        logger.info("--- MOCK EMAIL SENDER ---")
        logger.info(f"To: {email}")
        logger.info(f"Verification Link: http://localhost:8000/api/v1/auth/verify-email?token={token}")
        logger.info("-------------------------")

    async def verify_email(self, token: str) -> User:
        try:
            payload = decode_token(token, "verification")
            email = payload.get("sub")
            if not email:
                raise AuthenticationError("Invalid verification token")
        except Exception:
            raise AuthenticationError("Invalid or expired verification token")

        user = await self.user_service.get_by_email(email)
        if not user:
            raise NotFoundError("User not found")

        from app.schemas.user import UserUpdate
        await self.user_service.update_user(user, UserUpdate(is_verified=True))
        return user

    async def request_password_recovery(self, email: str) -> None:
        user = await self.user_service.get_by_email(email)
        if not user:
            raise NotFoundError("User not found")

        token = create_password_reset_token(email)
        logger.info("--- MOCK EMAIL SENDER ---")
        logger.info(f"To: {email}")
        logger.info(f"Password Reset Link: http://localhost:8000/api/v1/auth/reset-password?token={token}")
        logger.info("-------------------------")

    async def reset_password(self, token: str, new_password: str) -> User:
        try:
            payload = decode_token(token, "password_reset")
            email = payload.get("sub")
            if not email:
                raise AuthenticationError("Invalid password reset token")
        except Exception:
            raise AuthenticationError("Invalid or expired password reset token")

        user = await self.user_service.get_by_email(email)
        if not user:
            raise NotFoundError("User not found")

        from app.schemas.user import UserUpdate
        await self.user_service.update_user(user, UserUpdate(password=new_password))
        return user

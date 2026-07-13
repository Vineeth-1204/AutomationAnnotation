from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from app.core.deps import get_user_service, get_current_user, get_auth_service
from app.models.user import User
from app.schemas.token import (
    Token,
    TokenRefreshRequest,
    EmailVerificationRequest,
    PasswordRecoveryRequest,
    PasswordResetRequest,
)
from app.schemas.user import UserResponse
from app.services.user import UserService
from app.services.auth import AuthService

router = APIRouter()

@router.post("/login", response_model=Token)
async def login_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    user_service: UserService = Depends(get_user_service),
    auth_service: AuthService = Depends(get_auth_service),
) -> Token:
    """OAuth2 compatible token login, get access and refresh tokens."""
    user = await user_service.authenticate(
        email=form_data.username, password=form_data.password
    )
    access_token, refresh_token = await auth_service.get_tokens(user)
    return Token(access_token=access_token, refresh_token=refresh_token, token_type="bearer")

@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_in: TokenRefreshRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> Token:
    """Exchange a refresh token for a new access token."""
    new_access_token = await auth_service.refresh_access_token(refresh_in.refresh_token)
    return Token(
        access_token=new_access_token,
        refresh_token=refresh_in.refresh_token,
        token_type="bearer",
    )

@router.post("/request-verification", status_code=status.HTTP_202_ACCEPTED)
async def request_verification(
    verification_in: PasswordRecoveryRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Request email verification token (sends simulated email)."""
    await auth_service.request_verification(verification_in.email)
    return {"message": "Verification email sent"}

@router.post("/verify-email", response_model=UserResponse)
async def verify_email(
    token_in: EmailVerificationRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    """Confirm user email verification using the token."""
    return await auth_service.verify_email(token_in.token)

@router.post("/password-recovery", status_code=status.HTTP_202_ACCEPTED)
async def password_recovery(
    recovery_in: PasswordRecoveryRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Request password recovery link (sends simulated email)."""
    await auth_service.request_password_recovery(recovery_in.email)
    return {"message": "Password recovery email sent"}

@router.post("/reset-password", response_model=UserResponse)
async def reset_password(
    reset_in: PasswordResetRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    """Reset user password using token."""
    return await auth_service.reset_password(reset_in.token, reset_in.new_password)

@router.post("/test-token", response_model=UserResponse)
async def test_token(current_user: User = Depends(get_current_user)) -> User:
    """Test access token validity."""
    return current_user

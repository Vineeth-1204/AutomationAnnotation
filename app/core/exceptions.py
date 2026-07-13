from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class AppException(Exception):
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class NotFoundError(AppException):
    def __init__(self, message: str = "Resource not found", details: dict = None):
        super().__init__(message, status_code=404, details=details)

class AuthenticationError(AppException):
    def __init__(self, message: str = "Could not authenticate user", details: dict = None):
        super().__init__(message, status_code=401, details=details)

class ForbiddenError(AppException):
    def __init__(self, message: str = "Permission denied", details: dict = None):
        super().__init__(message, status_code=403, details=details)

class ConflictError(AppException):
    def __init__(self, message: str = "Resource already exists", details: dict = None):
        super().__init__(message, status_code=409, details=details)

class BadRequestError(AppException):
    def __init__(self, message: str = "Bad request", details: dict = None):
        super().__init__(message, status_code=400, details=details)

class ValidationError(AppException):
    def __init__(self, message: str = "Validation error", details: dict = None):
        super().__init__(message, status_code=422, details=details)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        logger.warning(
            f"AppException raised: {exc.message} | URL: {request.url} | Status: {exc.status_code}"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "message": exc.message,
                    "code": exc.__class__.__name__,
                    "details": exc.details
                }
            }
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            f"Unhandled exception: {str(exc)} | URL: {request.url}",
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "message": "Internal server error",
                    "code": "InternalServerError",
                    "details": {}
                }
            }
        )

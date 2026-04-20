import logging
from fastapi import Request
from fastapi.responses import JSONResponse

from app.exceptions.auth_exceptions import AppBaseException, RateLimitException

logger = logging.getLogger(__name__)


def register_handlers(app):
    @app.exception_handler(AppBaseException)
    async def app_base_exc_handler(request: Request, exc: AppBaseException):
        logger.warning("AppBaseException: %s", exc)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(RateLimitException)
    async def rate_limit_handler(request: Request, exc: RateLimitException):
        headers = {"Retry-After": "60"}
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict(), headers=headers)

    @app.exception_handler(Exception)
    async def generic_exc_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

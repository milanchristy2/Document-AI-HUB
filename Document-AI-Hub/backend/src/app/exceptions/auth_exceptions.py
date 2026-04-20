from typing import Dict


class AppBaseException(Exception):
    status_code: int = 400
    message: str = "An error occurred"

    def to_dict(self) -> Dict[str, str]:
        return {"error": self.message}


class TokenExpiredException(AppBaseException):
    status_code = 401
    message = "Token expired"


class TokenInvalidException(AppBaseException):
    status_code = 401
    message = "Token invalid"


class UnauthorizedException(AppBaseException):
    status_code = 403
    message = "Unauthorized"


class NotFoundException(AppBaseException):
    status_code = 404
    message = "Not found"


class RateLimitException(AppBaseException):
    status_code = 429
    message = "Too Many Requests"


class ValidationException(AppBaseException):
    status_code = 422
    message = "Validation failed"

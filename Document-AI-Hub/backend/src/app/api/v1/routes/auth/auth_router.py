from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.api.v1.deps.deps import get_db, get_token_payload
from app.services.auth_service import AuthService

router = APIRouter()
logger = logging.getLogger(__name__)

class SignupRequest(BaseModel):
    email: str
    password: str
    role: str = "user"

@router.post("/signup")
async def signup(req: SignupRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user.
    
    Returns:
        - 200: User created successfully with id and email
        - 400: Email already in use or invalid request
        - 500: Database error
    """
    service = AuthService(db)
    try:
        user = await service.signup(req.email, req.password, req.role)
        logger.info(f"User registered successfully: {req.email}")
        return {"id": user.id, "email": user.email}
    except ValueError as e:
        # Handle duplicate email or validation errors
        logger.warning(f"Signup validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle database or other errors
        logger.exception(f"Signup error: {e}")
        error_msg = str(e)
        # Return more specific status codes based on error type
        if "already in use" in error_msg.lower() or "unique" in error_msg.lower():
            raise HTTPException(status_code=409, detail=str(e))
        elif "table" in error_msg.lower():
            raise HTTPException(status_code=503, detail="Database service unavailable")
        else:
            raise HTTPException(status_code=500, detail=str(e))

class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return JWT token.
    
    Returns:
        - 200: Authentication successful with access_token, token_type, and role
        - 401: Invalid credentials
        - 503: Database service unavailable
    """
    service = AuthService(db)
    try:
        result = await service.login(req.email, req.password)
        logger.info(f"User logged in: {req.email}")
        return result
    except ValueError as e:
        # Invalid credentials
        logger.debug(f"Login failed for {req.email}: {e}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        # Database errors
        logger.exception(f"Login error: {e}")
        error_msg = str(e)
        if "table" in error_msg.lower() or "database" in error_msg.lower():
            raise HTTPException(status_code=503, detail="Database service unavailable. Please try again later.")
        else:
            raise HTTPException(status_code=401, detail="Authentication failed")


@router.get("/me")
async def me(payload: dict = Depends(get_token_payload)):
    """Get current authenticated user info."""
    return payload


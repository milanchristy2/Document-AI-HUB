"""Unified rate limiting module combining IP-based and user-role-based rate limiting.

This module provides:
1. RateLimitMiddleware: Global IP-based rate limiting for all endpoints
2. User role-based rate limiting: Per-user, per-action limits by role
3. Utilities: Status checks and limit configuration
"""

from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi.responses import JSONResponse as FastAPIJSONResponse

# Conditional Redis import (use mock if unavailable for local development)
try:
    from app.infra.cache.redis_client import get_redis_client
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from app.config.config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION: Rate limits by role (requests per minute)
# ============================================================================
RATE_LIMITS = {
    "lawyer": {"upload": 10, "query": 30, "search": 50},
    "doctor": {"upload": 10, "query": 30, "search": 50},
    "researcher": {"upload": 5, "query": 20, "search": 40},
    "analyst": {"upload": 10, "query": 25, "search": 45},
    "user": {"upload": 3, "query": 10, "search": 20},
    "admin": {"upload": 1000, "query": 1000, "search": 1000},
}

# In-memory store for rate limit tracking (in production, use Redis)
_rate_limit_store: Dict[str, list] = {}

# ============================================================================
# SECTION 1: Middleware for Global IP-Based Rate Limiting
# ============================================================================


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Global rate limiting middleware based on IP address and endpoint.
    
    Uses Redis for distributed rate limiting if available, falls back to in-memory.
    Checks against settings.RATE_LIMIT_CHAT and settings.RATE_LIMIT_UPLOAD.
    """

    async def dispatch(self, request: Request, call_next):
        try:
            path = request.url.path
            ip = request.client.host if request.client else "anonymous"
            
            # Create rate limit key: rl:{ip}:{path}:{time_window}
            window_key = int(request.scope.get('time', 0)) // settings.RATE_LIMIT_WINDOW_S
            key = f"rl:{ip}:{path}:{window_key}"
            
            # Determine limit based on endpoint
            limit = (
                settings.RATE_LIMIT_CHAT 
                if path.startswith("/api/v1/chat") 
                else settings.RATE_LIMIT_UPLOAD
            )
            
            # Increment counter using Redis if available, else in-memory
            if REDIS_AVAILABLE:
                rc = get_redis_client()
                cnt = await rc.incr(key, ttl=settings.RATE_LIMIT_WINDOW_S)
            else:
                # In-memory fallback for local development
                cnt = _increment_in_memory(key)
            
            # Check if limit exceeded
            if limit and cnt > limit:
                logger.warning(
                    f"Rate limit exceeded for IP {ip} on {path}. "
                    f"Limit: {limit}, Requests: {cnt}"
                )
                return FastAPIJSONResponse(
                    status_code=429,
                    content={"error": "rate_limited", "message": "Too many requests"}
                )
        
        except Exception as e:
            logger.debug(f"RateLimit middleware failed, failing open: {e}")
        
        return await call_next(request)


def _increment_in_memory(key: str) -> int:
    """In-memory counter for rate limiting (local development only)."""
    if key not in _rate_limit_store:
        _rate_limit_store[key] = []
    
    now = time.time()
    window_start = now - settings.RATE_LIMIT_WINDOW_S
    
    # Remove old requests outside window
    _rate_limit_store[key] = [t for t in _rate_limit_store[key] if t > window_start]
    
    # Add current request
    _rate_limit_store[key].append(now)
    return len(_rate_limit_store[key])


# ============================================================================
# SECTION 2: User Role-Based Rate Limiting
# ============================================================================


def check_rate_limit(
    user_id: str,
    action: str,
    user_role: str = "user"
) -> Tuple[bool, str, int]:
    """Check if user is within rate limits for a specific action.
    
    Implements per-user, per-action rate limiting with a 1-minute rolling window.
    
    Args:
        user_id: Unique user identifier
        action: Action type - "upload", "query", or "search"
        user_role: User's role for limit lookup (lawyer, doctor, researcher, analyst, user, admin)
    
    Returns:
        Tuple of:
        - allowed (bool): True if request is allowed, False if rate limited
        - message (str): Status message
        - remaining (int): Remaining requests in current window
    
    Example:
        >>> allowed, msg, remaining = check_rate_limit("user123", "upload", "lawyer")
        >>> if not allowed:
        ...     return {"error": msg}, 429
        >>> return {"status": "uploaded", "remaining": remaining}
    """
    # Normalize role
    user_role = user_role.lower().strip()
    if user_role not in RATE_LIMITS:
        user_role = "user"
        logger.debug(f"Unknown role, defaulting to 'user'")
    
    limit = RATE_LIMITS[user_role].get(action, 10)
    
    # Create storage key: {user_id}:{action}
    key = f"{user_id}:{action}"
    now = time.time()
    window_start = now - 60  # 1-minute rolling window
    
    # Initialize or clean old requests
    if key not in _rate_limit_store:
        _rate_limit_store[key] = []
    
    # Remove requests outside the window
    _rate_limit_store[key] = [
        t for t in _rate_limit_store[key] 
        if t > window_start
    ]
    
    # Check if limit exceeded
    if len(_rate_limit_store[key]) >= limit:
        remaining = 0
        reset_time = int(_rate_limit_store[key][0] + 60)
        time_until_reset = reset_time - int(now)
        
        message = (
            f"Rate limit exceeded for {action}. "
            f"Limit: {limit}/min. "
            f"Reset in {time_until_reset}s"
        )
        logger.warning(f"Rate limit exceeded: {user_id} action={action} role={user_role}")
        return False, message, remaining
    
    # Record this request
    _rate_limit_store[key].append(now)
    remaining = limit - len(_rate_limit_store[key])
    
    return True, f"OK ({remaining} remaining)", remaining


def get_rate_limit_status(
    user_id: str,
    user_role: str = "user"
) -> Dict[str, dict]:
    """Get current rate limit status for a user across all actions.
    
    Args:
        user_id: Unique user identifier
        user_role: User's role for limit lookup
    
    Returns:
        Dictionary with status for each action:
        {
            "upload": {"limit": 10, "used": 2, "remaining": 8, "reset_in_seconds": 45},
            "query": {"limit": 30, "used": 5, "remaining": 25, "reset_in_seconds": 45},
            "search": {"limit": 50, "used": 0, "remaining": 50, "reset_in_seconds": 60}
        }
    """
    # Normalize role
    user_role = user_role.lower().strip()
    if user_role not in RATE_LIMITS:
        user_role = "user"
    
    status = {}
    now = time.time()
    window_start = now - 60
    
    for action in ["upload", "query", "search"]:
        key = f"{user_id}:{action}"
        limit = RATE_LIMITS[user_role].get(action, 10)
        
        # Count requests in current window
        if key in _rate_limit_store:
            requests_in_window = len([
                t for t in _rate_limit_store[key]
                if t > window_start
            ])
        else:
            requests_in_window = 0
        
        # Calculate reset time
        if key in _rate_limit_store and _rate_limit_store[key]:
            oldest_request = min(_rate_limit_store[key])
            reset_in = int(max(0, 60 - (now - oldest_request)))
        else:
            reset_in = 60
        
        status[action] = {
            "limit": limit,
            "used": requests_in_window,
            "remaining": limit - requests_in_window,
            "reset_in_seconds": reset_in
        }
    
    return status


# ============================================================================
# SECTION 3: Utility Functions
# ============================================================================


def reset_user_limits(user_id: str, action: Optional[str] = None) -> None:
    """Reset rate limits for a user (useful for testing or admin operations).
    
    Args:
        user_id: User identifier
        action: Specific action to reset, or None to reset all actions
    """
    if action:
        key = f"{user_id}:{action}"
        if key in _rate_limit_store:
            _rate_limit_store[key] = []
            logger.info(f"Reset rate limit for {user_id}:{action}")
    else:
        # Reset all actions for user
        keys_to_reset = [k for k in _rate_limit_store if k.startswith(f"{user_id}:")]
        for key in keys_to_reset:
            _rate_limit_store[key] = []
        logger.info(f"Reset all rate limits for {user_id}")


def get_all_rate_limits() -> Dict[str, dict]:
    """Get all configured rate limits by role.
    
    Useful for API endpoints that expose rate limit info.
    """
    return RATE_LIMITS.copy()

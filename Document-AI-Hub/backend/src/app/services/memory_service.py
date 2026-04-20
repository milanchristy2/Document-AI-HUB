import asyncio
import logging
from typing import Any, List, Dict, Optional

from app.infra.cache.redis_client import get_redis_client
from app.config.config import settings

logger = logging.getLogger(__name__)

# Keep memory light: limit to last N turns per (user, session)
MAX_TURNS = 10
TTL_SECONDS = 60 * 60 * 24  # keep session history for 24h


class MemoryService:
    def __init__(self):
        self._redis = get_redis_client()

    async def load(self, user_id: Optional[str], session_id: Optional[str]) -> List[Dict[str, Any]]:
        key = self._key(user_id, session_id)
        try:
            data = await self._redis.get_json(key)
            if not data:
                return []
            return data
        except Exception as e:
            logger.debug("Memory load failed for %s: %s", key, e)
            return []

    async def append(self, user_id: Optional[str], session_id: Optional[str], role: str, text: str) -> bool:
        key = self._key(user_id, session_id)
        try:
            hist = await self._redis.get_json(key) or []
            hist.append({"role": role, "text": text})
            # trim to last MAX_TURNS
            if len(hist) > MAX_TURNS:
                hist = hist[-MAX_TURNS:]
            ok = await self._redis.set_json(key, hist, ttl=TTL_SECONDS)
            return ok
        except Exception as e:
            logger.debug("Memory append failed for %s: %s", key, e)
            return False

    def format_for_prompt(self, hist: List[Dict[str, Any]]) -> str:
        if not hist:
            return ""
        parts: List[str] = []
        for t in hist:
            role = t.get("role", "user")
            text = t.get("text", "")
            if role == "assistant":
                parts.append(f"Assistant: {text}")
            else:
                parts.append(f"User: {text}")
        return "\n".join(parts)

    def _key(self, user_id: Optional[str], session_id: Optional[str]) -> str:
        uid = user_id or "anon"
        sid = session_id or "default"
        return f"conv:{uid}:{sid}"


memory_service = MemoryService()

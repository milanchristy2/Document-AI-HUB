import asyncio
import json
import logging
from typing import Any, Optional

from app.config.config import settings

logger = logging.getLogger(__name__)

_redis = None


def _get_redis():
    global _redis
    if _redis is None:
        try:
            # prefer redis-py's asyncio client (redis>=4.2) and fall back to aioredis
            try:
                import redis.asyncio as _aioredis
                _redis = _aioredis.from_url(settings.REDIS_URL)
            except Exception:
                import aioredis as _aioredis
                _redis = _aioredis.from_url(settings.REDIS_URL)
        except Exception as e:
            # log full exception to help debug duplicate-base-class or version conflicts
            logger.warning("Redis init failed: %s", e)
            logger.debug("Redis import traceback:", exc_info=True)
            _redis = None
    return _redis


class RedisClient:
    def __init__(self):
        self._r = _get_redis()

    async def get(self, key: str) -> Optional[str]:
        try:
            if not self._r:
                return None
            return await self._r.get(key)
        except Exception as e:
            logger.debug("Redis get failed: %s", e)
            return None

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        try:
            if not self._r:
                return False
            await self._r.set(key, value, ex=ttl)
            return True
        except Exception as e:
            logger.debug("Redis set failed: %s", e)
            return False

    async def delete(self, key: str) -> bool:
        try:
            if not self._r:
                return False
            await self._r.delete(key)
            return True
        except Exception as e:
            logger.debug("Redis delete failed: %s", e)
            return False

    async def incr(self, key: str, ttl: int | None = None) -> int:
        try:
            if not self._r:
                return 0
            val = await self._r.incr(key)
            if ttl:
                await self._r.expire(key, ttl)
            return int(val)
        except Exception as e:
            logger.debug("Redis incr failed: %s", e)
            return 0

    async def get_json(self, key: str) -> Any | None:
        raw = await self.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    async def set_json(self, key: str, value: Any, ttl: int | None = None) -> bool:
        try:
            return await self.set(key, json.dumps(value), ttl)
        except Exception:
            return False


_redis_client = None

def get_redis_client() -> RedisClient:
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client

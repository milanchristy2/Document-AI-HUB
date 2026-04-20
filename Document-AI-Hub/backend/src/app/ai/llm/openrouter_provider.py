import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class OpenRouterProvider:
    """Minimal OpenRouter provider stub.

    Keep this adapter lightweight so the rest of the app can call
    `generate(prompt)` uniformly regardless of provider.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    async def generate(self, prompt: str, timeout: int = 30) -> Optional[str]:
        try:
            await asyncio.sleep(0)
            return f"[openrouter] simulated response for prompt length {len(prompt)}"
        except Exception as e:
            logger.exception("OpenRouter generation failed: %s", e)
            return None

__all__ = ["OpenRouterProvider"]

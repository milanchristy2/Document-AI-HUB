import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class GroqProvider:
    """Minimal Groq provider stub.

    Replace the internal call with the real Groq HTTP client when ready.
    """

    def __init__(self, api_key: str | None = None, endpoint: str | None = None):
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.groq.ai"

    async def generate(self, prompt: str, timeout: int = 30) -> Optional[str]:
        try:
            await asyncio.sleep(0)
            return f"[groq] simulated response for prompt length {len(prompt)}"
        except Exception as e:
            logger.exception("Groq generation failed: %s", e)
            return None

__all__ = ["GroqProvider"]

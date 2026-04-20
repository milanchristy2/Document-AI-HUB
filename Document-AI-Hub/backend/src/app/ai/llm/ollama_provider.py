import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from app.infra.llm.ollama_client import call_ollama
    _HAS_CALL = True
except Exception:
    call_ollama = None
    _HAS_CALL = False


class OllamaProvider:
    """Ollama provider that delegates to `app.infra.llm.ollama_client.call_ollama` when available.

    Falls back to a lightweight simulated response when the infra client or requests
    is not available. Uses `asyncio.to_thread` to keep compatibility with sync client.
    """

    def __init__(self, model: str = "phi3:mini"):
        self.model = model

    async def generate(self, prompt: str, timeout: int = 30) -> Optional[str]:
        try:
            if _HAS_CALL and call_ollama is not None:
                # call the synchronous helper in a thread
                return await asyncio.to_thread(call_ollama, prompt, self.model, timeout)
            # fallback simulated response
            await asyncio.sleep(0)
            return f"[ollama:{self.model}] simulated response for prompt length {len(prompt)}"
        except Exception as e:
            logger.exception("Ollama generation failed: %s", e)
            return None


__all__ = ["OllamaProvider"]

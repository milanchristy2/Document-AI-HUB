"""Minimal Ollama client wrapper.

Attempts to call a local Ollama API at the default port. This is intentionally
lightweight and resilient: it falls back silently if the service is not reachable.
"""
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False


def call_ollama(prompt: str, model: str = "phi3:mini", timeout: int = 120) -> Optional[str]:
    """Call Ollama HTTP API synchronously. Returns text or None on failure."""
    if not REQUESTS_AVAILABLE:
        return None
    url = "http://localhost:11434/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, dict):
            # Try 'response' (standard Ollama generate format)
            if "response" in j:
                return j["response"]
            # Try 'output' (some variations)
            if "output" in j:
                return j["output"]
            # Try 'results' (as previously used)
            if "results" in j and isinstance(j["results"], list):
                parts = []
                for item in j["results"]:
                    if isinstance(item, dict):
                        c = item.get("content") or item.get("text")
                        if c:
                            parts.append(c)
                return "".join(parts) if parts else None
        return None
    except Exception as e:
        logger.debug("Ollama call failed: %s", e)
        return None

"""Lightweight wrapper for text embeddings used by RAG multimodal steps.

Delegates to `app.multimodal.embeddings.get_text_embeddings` when available,
otherwise falls back to the deterministic fallback in `app.utils.embeddings`.
"""
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from app.multimodal.embeddings import get_text_embeddings as _get
except Exception:
    _get = None


def get_text_embeddings(texts: List[str], model_name: Optional[str] = None):
    if _get:
        try:
            return _get(texts, model_name=model_name)
        except Exception:
            logger.exception("app.multimodal.embeddings failed; falling back")

    # fallback to utils embedding implementation
    from app.utils.embeddings import embed_texts
    return embed_texts(texts, model_name or "all-MiniLM-L6-v2")

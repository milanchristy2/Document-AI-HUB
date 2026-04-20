"""Multimodal embeddings helpers.

This module provides convenience wrappers for text embeddings used by the RAG pipeline.
It currently delegates to `app.utils.embeddings.embed_texts`.
"""
from typing import List, Optional
from app.utils.embeddings import embed_texts


def get_text_embeddings(texts: List[str], model_name: Optional[str] = None):
    model = model_name or "all-MiniLM-L6-v2"
    return embed_texts(texts, model_name=model)

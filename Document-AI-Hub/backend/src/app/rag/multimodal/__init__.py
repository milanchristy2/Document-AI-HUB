"""Wafer-thin multimodal helpers for the RAG pipeline.

These wrappers provide safe, optional hooks for image/audio embeddings and extraction.
They avoid hard dependencies and return empty/placeholder values when native libs are missing.
"""

from .embeddings import get_text_embeddings
from ...processors.extractors.image_extractor import extract_image_text, extract_image_caption
from ...processors.extractors.audio_extractor import transcribe_audio
from .visual_caption import caption_image

__all__ = [
    "get_text_embeddings",
    "extract_image_text",
    "extract_image_caption",
    "transcribe_audio",
    "caption_image",
]

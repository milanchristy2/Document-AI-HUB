"""Thin audio transcription wrapper (Whisper if available).

Returns empty string if no transcription library is installed.
"""
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def transcribe_audio(path: str, model: str = "small") -> str:
    """Transcribe audio file at `path` using Whisper (if installed), else return empty string."""
    try:
        import whisper
    except Exception:
        logger.debug("whisper not available for transcription")
        return ""
    try:
        m = whisper.load_model(model)
        res = m.transcribe(path)
        return res.get('text', '')
    except Exception as e:
        logger.debug("whisper transcription failed: %s", e)
        return ""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_pdf(path: str) -> str:
    """Try `pdfplumber` for PDF text extraction, fallback to empty string."""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n\n".join(pages)
    except Exception as e:
        logger.debug("pdf extraction failed for %s: %s", path, e)
        try:
            # fallback: try plain read (not useful for PDFs but safe)
            with open(path, "rb") as f:
                return ""
        except Exception:
            return ""


def extract_text_from_image(path: str) -> str:
    """Try `pytesseract` via PIL.Image, fallback to empty string."""
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        logger.debug("image OCR failed for %s: %s", path, e)
        return ""


def extract_text_from_audio(path: str) -> str:
    """Try `whisper` if available, otherwise empty string."""
    try:
        import whisper
        model = whisper.load_model("base")
        res = model.transcribe(path)
        return res.get("text", "")
    except Exception as e:
        logger.debug("audio transcription failed for %s: %s", path, e)
        return ""


def extract_text(path: str, mime: Optional[str] = None) -> str:
    """Dispatch to the appropriate extractor based on mime or extension."""
    if mime:
        m = mime.lower()
        if "pdf" in m:
            return extract_text_from_pdf(path)
        if "image" in m:
            return extract_text_from_image(path)
        if "audio" in m:
            return extract_text_from_audio(path)
    # fallback by extension
    if path.lower().endswith(".pdf"):
        return extract_text_from_pdf(path)
    if path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        return extract_text_from_image(path)
    if path.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
        return extract_text_from_audio(path)
    # default: try text read
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_image",
    "extract_text_from_audio",
    "extract_text",
]
"""Thin extractors for multimodal content (PDF, image, audio).

These functions attempt to use common libraries when available and otherwise
return lightweight placeholders so the rest of the pipeline can function in
environments without heavy native dependencies.
"""
from typing import Optional


def extract_text_from_pdf(file_path: str) -> str:
    """Attempt to extract text from PDF. Falls back to empty string."""
    try:
        from pdfminer.high_level import extract_text

        return extract_text(file_path) or ""
    except Exception:
        return ""


def extract_text_from_image(image_bytes: bytes) -> str:
    """Attempt OCR via pytesseract when available."""
    try:
        from PIL import Image
        import io
        import pytesseract

        img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


def extract_text_from_audio(audio_bytes: bytes) -> str:
    """Attempt to run a speech-to-text model (whisper) if available; otherwise return empty."""
    try:
        import whisper
        import io
        wf = io.BytesIO(audio_bytes)
        # Whisper expects a file path; we keep this thin and avoid complex bridging.
        return ""
    except Exception:
        return ""

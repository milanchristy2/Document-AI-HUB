"""Wafer-thin image extractors: OCR and simple caption hooks.

These functions try to use `pytesseract` and `Pillow` if present; otherwise return empty strings.
"""
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None


def extract_image_text(path: str, lang: Optional[str] = None) -> str:
    """Return OCR text for `path`, or empty string if OCR not available."""
    if not Image or not pytesseract:
        logger.debug("OCR libs not available (Pillow/pytesseract)")
        return ""
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang=lang) if lang else pytesseract.image_to_string(img)
        return text or ""
    except Exception as e:
        logger.debug("OCR failed for %s: %s", path, e)
        return ""


def extract_image_caption(path: str) -> str:
    """Optional image captioning hook. Returns empty string when captioning models are not installed."""
    # Keep this thin: try to import a captioner (e.g., transformers/BLIP) if available
    try:
        # lazy import to avoid hard deps
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image as _PILImage
        img = _PILImage.open(path).convert('RGB')
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption or ""
    except Exception:
        return ""

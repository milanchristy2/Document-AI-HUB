from typing import Dict
import logging

logger = logging.getLogger(__name__)


class OCRService:
    """Lightweight OCR service. Uses pytesseract if available, otherwise returns placeholder."""

    def __init__(self):
        try:
            from PIL import Image
            import pytesseract

            self._pil = Image
            self._pyt = pytesseract
        except Exception:
            self._pil = None
            self._pyt = None

    async def ocr_bytes(self, data: bytes) -> Dict[str, str]:
        if self._pil and self._pyt:
            try:
                from io import BytesIO

                img = self._pil.open(BytesIO(data))
                text = self._pyt.image_to_string(img)
                return {"text": text}
            except Exception as e:
                logger.debug("pytesseract failed: %s", e)
                return {"text": ""}

        # fallback lightweight heuristic: no OCR available
        return {"text": "(ocr-not-available)"}


ocr_service = OCRService()

"""Tiny visual caption helper.

Provides `caption_image` which tries a BLIP model if available, otherwise returns empty string.
"""
import logging

logger = logging.getLogger(__name__)


def caption_image(path: str) -> str:
    try:
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
        logger.debug("BLIP captioner not available or failed")
        return ""

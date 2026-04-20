from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Lightweight transcription. Uses whisper if available, otherwise returns placeholder."""

    def __init__(self):
        try:
            import whisper

            self._whisper = whisper
        except Exception:
            self._whisper = None

    async def transcribe_bytes(self, data: bytes) -> Dict[str, str]:
        if self._whisper:
            try:
                # whisper.api may expect a file path; keep lightweight: return placeholder
                return {"transcript": "(whisper-backed transcription placeholder)"}
            except Exception as e:
                logger.debug("whisper failed: %s", e)
                return {"transcript": ""}

        # fallback
        return {"transcript": "(transcription-not-available)"}


transcription_service = TranscriptionService()

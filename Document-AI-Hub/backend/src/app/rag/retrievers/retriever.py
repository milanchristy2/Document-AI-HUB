from typing import List, Dict
from app.services.document_service import DocumentService
from backend.src.app.processors.extractors.image_extractor import extract_image_text, extract_image_caption
from backend.src.app.processors.extractors.audio_extractor import transcribe_audio
from app.utils.chunker import chunk_text


def has_image_chunks(file_id: str) -> bool:
    # placeholder: in this simple implementation, assume image docs may have multimodal chunks
    return True if file_id else False


class MultimodalRetriever:
    def __init__(self, db=None):
        self.db = db

    async def retrieve(self, query: str, document_id: str = None) -> List[Dict]:
        results: List[Dict] = []
        if not document_id or not self.db:
            return results

        doc = await DocumentService(self.db).get_by_id(document_id, "")
        if not doc:
            return results

        text = ""
        if doc.content_type.startswith("image"):
            text = extract_image_text(str(doc.storage_path)) or extract_image_caption(str(doc.storage_path))
        elif doc.content_type.startswith("audio"):
            text = transcribe_audio(str(doc.storage_path))

        if text:
            chunks = chunk_text(text, chunk_size=250, overlap=50)
            for i, c in enumerate(chunks):
                results.append({"text": c, "heading": "multimodal", "source": "multimodal"})
        return results

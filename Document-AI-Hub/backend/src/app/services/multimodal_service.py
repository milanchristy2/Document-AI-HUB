import logging
from typing import Dict, Any

from app.infra.storage.local_storage import local_storage
from app.models.document_model import Document, DocumentStatus

logger = logging.getLogger(__name__)


class MultimodalService:
    """Orchestrates multimodal ingestion: extract, chunk, embed, store."""

    def __init__(self, db):
        self.db = db

    async def ingest(self, file_bytes: bytes, content_type: str, document_id: str, filename: str) -> Dict[str, Any]:
        # Minimal implementation: mark processing, store placeholder chunks
        doc = await self.db.get(Document, document_id)
        if not doc:
            raise ValueError("Document not found")

        try:
            doc.mark_processing()
            self.db.add(doc)
            await self.db.flush()

            # Placeholder: in real implementation call extractors, chunker, embedder
            # For now, store original file in storage (already stored) and mark ready with 0 chunks
            doc.mark_ready(0)
            self.db.add(doc)
            await self.db.flush()
            return {"status": "ok"}
        except Exception as e:
            logger.exception("Multimodal ingest failed: %s", e)
            doc.mark_failed(str(e))
            self.db.add(doc)
            await self.db.flush()
            return {"status": "failed", "error": str(e)}


multimodal_service = None

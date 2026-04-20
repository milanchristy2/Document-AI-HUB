import asyncio
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infra.storage.local_storage import local_storage
from app.models.document_model import Document, DocumentStatus

logger = logging.getLogger(__name__)


class DocumentService:
    ALLOWED_TYPES = {"application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "image/png", "image/jpeg", "audio/mpeg", "video/mp4"}
    MAX_SIZE = 500 * 1024 * 1024  # 500MB

    def __init__(self, db: AsyncSession):
        self.db = db

    async def upload(self, data: bytes, filename: str, content_type: str, user_id: str) -> Document:
        if content_type not in self.ALLOWED_TYPES:
            raise ValueError("Unsupported content type")
        if len(data) > self.MAX_SIZE:
            raise ValueError("File too large")

        ext = filename.split('.')[-1] if '.' in filename else ''
        try:
            storage_name = await local_storage.save(data, ext)
        except IOError as e:
            logger.exception("Storage save failed for user=%s filename=%s: %s", user_id, filename, e)
            # Propagate a more informative message so API returns useful details for debugging
            raise ValueError(f"Failed to save to MinIO: {e}")
        except Exception as e:
            logger.exception("Unexpected storage error for user=%s filename=%s: %s", user_id, filename, e)
            raise

        doc = Document(user_id=user_id, filename=filename, storage_path=storage_name, content_type=content_type, file_size=len(data), status=DocumentStatus.queued)
        self.db.add(doc)
        await self.db.flush()

        # Trigger lightweight ingestion asynchronously using IngestPipeline (uses its own DB/session)
        try:
            from app.ingestion.pipeline import IngestPipeline

            asyncio.create_task(IngestPipeline(None).run(doc.id, storage_name, content_type, filename))
        except Exception as e:
            logger.debug("Failed to start ingest pipeline async: %s", e)

        return doc

    async def get_by_id(self, doc_id: str, user_id: str) -> Optional[Document]:
        q = await self.db.get(Document, doc_id)
        if not q or q.user_id != user_id:
            return None
        return q

    async def list_documents(self, user_id: str):
        res = await self.db.execute("SELECT * FROM documents WHERE user_id = :uid", {"uid": user_id})
        # Use SQLAlchemy Core/ORM select to avoid textual SQL coercion issues
        q = select(Document).where(Document.user_id == user_id)
        res = await self.db.execute(q)
        return res.scalars().all()

    async def delete(self, doc_id: str, user_id: str):
        doc = await self.get_by_id(doc_id, user_id)
        if not doc:
            return False
        await local_storage.delete(doc.storage_path)
        await self.db.delete(doc)
        await self.db.flush()
        return True


document_service = None

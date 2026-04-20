import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Enum, BigInteger
from sqlalchemy import Boolean

from app.infra.db.session import Base
import enum


class DocumentStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    file_size = Column(BigInteger, default=0)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.queued)
    chunk_count = Column(Integer, default=0)
    error_msg = Column(String, nullable=True)
    modality = Column(String, default="text")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def mark_processing(self):
        self.status = DocumentStatus.processing

    def mark_ready(self, chunk_count: int):
        self.status = DocumentStatus.ready
        self.chunk_count = chunk_count

    def mark_failed(self, error: str):
        self.status = DocumentStatus.failed
        self.error_msg = error

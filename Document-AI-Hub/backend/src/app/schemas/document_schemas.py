from pydantic import BaseModel
from typing import Optional


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int
    content_type: str
    error_msg: Optional[str]


class JobStatusResponse(BaseModel):
    id: str
    status: str
    message: Optional[str]

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class EvidenceMeta(BaseModel):
    page: Optional[int]
    chunk_index: Optional[int]
    score: Optional[float]
    extra: Optional[Dict[str, Any]] = None


class EvidenceBlock(BaseModel):
    id: str
    heading: Optional[str]
    snippet: Optional[str]
    text: Optional[str]
    source: Optional[str]
    meta: Optional[Dict[str, Any]]


class RAGMetadata(BaseModel):
    document_ids: List[str]
    retriever_scores: Optional[List[float]] = None


class RAGResponse(BaseModel):
    answer: str
    evidence: List[EvidenceBlock]
    metadata: Optional[RAGMetadata]
    raw: Optional[Dict[str, Any]] = None

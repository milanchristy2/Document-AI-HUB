from typing import Dict, Optional
from app.nlp.cleaning import normalize_text
from app.processors.processors import chunk_text
from app.processors.extractors.extractors import extract_text


def extract_from_path(path: str, mime: Optional[str] = None, max_chunks: int = 10) -> Dict:
    """Extract raw text from path and return normalized text + chunks + metadata.

    This is a thin wrapper that uses `app.rag.extractors.extract_text` then
    normalizes and chunks the text for ingestion.
    """
    raw = extract_text(path, mime)
    norm = normalize_text(raw)
    chunks = chunk_text(norm, chunk_size=1000, overlap=200)[:max_chunks]
    return {
        "path": path,
        "mime_type": mime,
        "text": norm,
        "chunks": chunks,
        "num_chunks": len(chunks),
    }


__all__ = ["extract_from_path"]

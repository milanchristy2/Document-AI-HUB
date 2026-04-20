import re
from typing import List, Dict, Optional
from app.nlp.cleaning import normalize_text, split_sentences


def preprocess_text(text: str, max_len: int = 2000) -> str:
    """Normalize whitespace, remove control chars, and truncate."""
    if not text:
        return ""
    t = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        return t[:max_len]
    return t


def _sentence_split(text: str) -> List[str]:
    # prefer nlp.split_sentences from the nlp utilities
    parts = split_sentences(text)
    if parts:
        return parts
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Sentence-aware chunker using NLTK when available.

    Joins sentences until `chunk_size` is reached, then rolls forward with `overlap`.
    """
    text = normalize_text(text)
    if not text:
        return []
    sentences = _sentence_split(text)
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for s in sentences:
        slen = len(s)
        if buf_len + slen + (1 if buf else 0) <= chunk_size:
            buf.append(s)
            buf_len += slen + (1 if buf_len else 0)
        else:
            if buf:
                chunks.append(" ".join(buf))
            # start new buffer with overlap sentences
            if overlap > 0 and chunks:
                # try to keep last sentence(s) as overlap
                overlap_sentences = []
                acc = 0
                for sent in reversed(_sentence_split(chunks[-1])):
                    if acc + len(sent) + 1 > overlap:
                        break
                    overlap_sentences.insert(0, sent)
                    acc += len(sent) + 1
                buf = overlap_sentences + [s]
                buf_len = sum(len(x) + 1 for x in buf)
            else:
                buf = [s]
                buf_len = slen
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def build_multimodal_context(ocr_text: Optional[str] = None, caption: Optional[str] = None) -> Dict:
    """Create a minimal multimodal context dict used by `chains` pre-retrieval hooks."""
    ctx: Dict = {}
    if ocr_text:
        ctx["ocr_text"] = preprocess_text(ocr_text, max_len=1200)
    if caption:
        ctx["caption"] = preprocess_text(caption, max_len=300)
    return ctx


__all__ = ["preprocess_text", "chunk_text", "build_multimodal_context"]

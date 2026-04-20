import re
from typing import List


def remove_control_chars(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", text)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str, max_len: int = 20000) -> str:
    """Normalize document text: control chars removal, whitespace collapse, minimal canonicalization.

    This is intentionally thin and deterministic.
    """
    if not text:
        return ""
    t = remove_control_chars(text)
    t = collapse_whitespace(t)
    if len(t) > max_len:
        return t[:max_len]
    return t


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter using punctuation heuristics.

    Prefer NLTK in production; this is a fallback.
    """
    # naive split keeping punctuation
    parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return parts


__all__ = ["normalize_text", "split_sentences", "remove_control_chars", "collapse_whitespace"]

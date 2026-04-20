from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Very small, fast chunker: split on whitespace to approx chunk_size."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        j = min(n, i + chunk_size)
        chunk = " ".join(words[i:j])
        chunks.append(chunk)
        i = j - overlap if j < n else j
    return chunks

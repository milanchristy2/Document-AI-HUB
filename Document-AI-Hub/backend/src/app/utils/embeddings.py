from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    S2_AVAILABLE = True
except Exception:
    S2_AVAILABLE = False


_MODEL = None


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    global _MODEL
    if _MODEL is None:
        if not S2_AVAILABLE:
            logger.warning("sentence-transformers not installed; falling back to deterministic embeddings")
            return None
        try:
            _MODEL = SentenceTransformer(model_name)
        except Exception as e:
            logger.exception("Failed to load SentenceTransformer %s: %s", model_name, e)
            _MODEL = None
    return _MODEL


from app.config.config import settings

def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2", dim: int | None = None) -> List[List[float]]:
    """Return embeddings for a list of texts.

    If `sentence-transformers` is available, uses the specified model (default: all-MiniLM-L6-v2).
    Otherwise falls back to a cheap deterministic vector for local testing.
    """
    model = _get_model(model_name)
    if model:
        try:
            embs = model.encode(texts, convert_to_numpy=True)
            # ensure list of primitive float lists
            return [[float(v) for v in e] for e in embs]
        except Exception:
            logger.exception("Embedding call failed; falling back to deterministic embeddings")

    # Fallback deterministic embeddings (simple stable hash->float mapping)
    import hashlib

    local_dim = dim if dim else (384 if S2_AVAILABLE else 64)
    vectors = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        vec = []
        for i in range(local_dim):
            b = h[i % len(h)]
            vec.append((b / 255.0))
        vectors.append(vec)
    return vectors


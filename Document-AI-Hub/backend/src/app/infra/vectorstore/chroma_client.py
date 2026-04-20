import logging
from functools import lru_cache

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:
    chromadb = None

from app.config.config import settings

logger = logging.getLogger(__name__)


@lru_cache()
def get_chroma_client():
    if chromadb is None:
        logger.warning("ChromaDB package not available")
        return None
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    return client


def get_collection(name: str):
    client = get_chroma_client()
    if client is None:
        return None
    try:
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    except Exception as e:
        logger.exception("Failed to get or create chroma collection: %s", e)
        return None


def get_text_collection():
    return get_collection(settings.CHROMA_COLLECTION)


def get_image_collection():
    return get_collection(settings.CHROMA_IMAGE_COLLECTION)

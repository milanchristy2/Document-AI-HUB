import logging
import os
import json
from typing import List, Dict, Any

import numpy as np

from app.config.config import settings
from app.infra.storage.minio_client import minio_client

logger = logging.getLogger(__name__)


def _ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _local_index_paths(document_id: str) -> Dict[str, str]:
    base = os.path.join(getattr(settings, "CHROMA_PATH", "./vector_store"), "faiss")
    _ensure_dir(base)
    return {
        "npy": os.path.join(base, f"{document_id}.npy"),
        "meta": os.path.join(base, f"{document_id}_meta.json"),
        "index": os.path.join(base, f"{document_id}.index"),
    }


def add_document_vectors(document_id: str, vectors: List[List[float]], ids: List[str], metadatas: List[Dict[str, Any]], persist_to_minio: bool = True) -> bool:
    """Store document vectors and metadata to local disk and optionally persist to MinIO.

    This implementation writes a numpy array and a metadata JSON. If FAISS is available
    it will also build and save a faiss index. The function is purposely simple and
    provides a robust fallback when native FAISS isn't installed.
    """
    paths = _local_index_paths(document_id)
    try:
        arr = np.asarray(vectors, dtype=np.float32)
        np.save(paths["npy"], arr)
        meta = {"ids": ids, "metadatas": metadatas, "dim": int(arr.shape[1])}
        with open(paths["meta"], "w", encoding="utf-8") as fh:
            json.dump(meta, fh)

        # Try to build a FAISS index if available (best-effort)
        try:
            import faiss  # type: ignore

            d = int(arr.shape[1])
            index = faiss.IndexFlatIP(d) if getattr(settings, "EMBED_DIM", None) else faiss.IndexFlatL2(d)
            # IndexFlat doesn't support ids; write simple index and persist array instead
            index.add(arr)
            faiss.write_index(index, paths["index"])
        except Exception:
            # FAISS not available or build failed; we continue with numpy files
            logger.debug("FAISS not available or index build failed; storing numpy and meta only")

        if persist_to_minio and getattr(minio_client, "_client", None):
            # Upload files to MinIO; name objects under faiss/{document_id}/
            bucket = getattr(settings, "MINIO_BUCKET", "rag-storage")
            base_obj = f"faiss/{document_id}/"
            # npy
            try:
                with open(paths["npy"], "rb") as fh:
                    data = fh.read()
                ok = minio_client.put_object(bucket, base_obj + os.path.basename(paths["npy"]), data, content_type="application/octet-stream")
                if not ok:
                    logger.warning("Failed to persist numpy vectors to MinIO for %s", document_id)
            except Exception as e:
                logger.debug("Persist numpy to MinIO failed: %s", e)

            # meta
            try:
                with open(paths["meta"], "rb") as fh:
                    mdata = fh.read()
                ok = minio_client.put_object(bucket, base_obj + os.path.basename(paths["meta"]), mdata, content_type="application/json")
                if not ok:
                    logger.warning("Failed to persist meta to MinIO for %s", document_id)
            except Exception as e:
                logger.debug("Persist meta to MinIO failed: %s", e)

            # index (optional)
            try:
                if os.path.exists(paths["index"]):
                    with open(paths["index"], "rb") as fh:
                        idata = fh.read()
                    ok = minio_client.put_object(bucket, base_obj + os.path.basename(paths["index"]), idata, content_type="application/octet-stream")
                    if not ok:
                        logger.warning("Failed to persist faiss index to MinIO for %s", document_id)
            except Exception as e:
                logger.debug("Persist faiss index to MinIO failed: %s", e)

        return True
    except Exception as e:
        logger.exception("Failed to write vectors/meta for %s: %s", document_id, e)
        return False


def load_document_vectors_from_minio(document_id: str) -> Dict[str, Any] | None:
    paths = _local_index_paths(document_id)
    bucket = getattr(settings, "MINIO_BUCKET", "rag-storage")
    base_obj = f"faiss/{document_id}/"
    try:
        # try npy
        data = minio_client.get_object(bucket, base_obj + os.path.basename(paths["npy"]))
        if data:
            # write locally
            with open(paths["npy"], "wb") as fh:
                fh.write(data)
        meta_b = minio_client.get_object(bucket, base_obj + os.path.basename(paths["meta"]))
        if meta_b:
            with open(paths["meta"], "wb") as fh:
                fh.write(meta_b)
        return {"npy": paths["npy"], "meta": paths["meta"]}
    except Exception as e:
        logger.debug("Failed to load vectors from MinIO for %s: %s", document_id, e)
        return None


__all__ = ["add_document_vectors", "load_document_vectors_from_minio"]

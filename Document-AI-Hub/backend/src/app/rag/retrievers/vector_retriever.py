import asyncio
import logging
import os
import json
from typing import Any, Dict, List, Optional, cast

import numpy as np

from app.config.config import settings
from app.infra.vectorstore.faiss_client import load_document_vectors_from_minio
from app.infra.vectorstore.faiss_client import _local_index_paths  # type: ignore
from app.infra.vectorstore.faiss_client import add_document_vectors  # noqa: F401
from app.infra.vectorstore.elasticsearch_client import get_es_client
from app.utils.embeddings import embed_texts
from app.infra.vectorstore.chroma_client import get_text_collection

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self, collection_name: str | None = None):
        # collection_name is unused in the new FAISS-based storage; retrieval is per-document
        self._collection_name = collection_name

    async def query(self, query: str, document_id: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        if document_id is None:
            return []

        def _compute_scores():
            try:
                paths = _local_index_paths(document_id)
                # load numpy vectors (either local or from MinIO)
                if not os.path.exists(paths["npy"]):
                    # attempt to fetch from MinIO
                    loaded = load_document_vectors_from_minio(document_id)
                    if not loaded:
                        return []

                arr = np.load(paths["npy"])  # shape (n, d)
                with open(paths["meta"], "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                ids = meta.get("ids", [])
                metadatas = meta.get("metadatas", [])

                if arr.size == 0:
                    return []

                # compute query embedding (synchronous call to embedding implementation)
                try:
                    qv = embed_texts([query], model_name=getattr(settings, "EMBED_MODEL", "all-MiniLM-L6-v2"), dim=getattr(settings, "EMBED_DIM", None))[0]
                except Exception:
                    return []

                qv = np.asarray(qv, dtype=np.float32)
                # cosine similarity: normalize
                def _norm(x):
                    n = np.linalg.norm(x, axis=1, keepdims=True)
                    n[n == 0] = 1.0
                    return x / n

                try:
                    arr_n = _norm(arr)
                    qn = qv / (np.linalg.norm(qv) or 1.0)
                    scores = (arr_n @ qn).astype(float)
                except Exception:
                    # fallback to dot
                    scores = (arr @ qv).astype(float)

                # select top_k
                idxs = np.argsort(-scores)[:top_k]

                candidates = []
                for i in idxs:
                    try:
                        score = float(scores[int(i)])
                    except Exception:
                        score = 0.0
                    mid = ids[int(i)] if i < len(ids) else None
                    mdata = metadatas[int(i)] if i < len(metadatas) else {}
                    candidates.append({"score": score, "meta": mdata, "id": mid})

                return candidates
            except Exception as e:
                logger.debug("Vector compute failed for %s: %s", document_id, e)
                return []

        # Try Chroma first (best-effort) to benefit from any existing semantic index
        def _query_chroma():
            try:
                col = get_text_collection()
                if not col:
                    return []
                where = {"document_id": document_id} if document_id else None
                try:
                    res = col.query(query_texts=[query], n_results=top_k, where=where)
                except Exception:
                    return []

                results = []
                if isinstance(res, dict):
                    docs = []
                    metadatas = []
                    distances = []
                    ids = []
                    if "results" in res:
                        first = res.get("results", [None])[0]
                        if first and isinstance(first, dict):
                            docs = first.get("documents", []) or []
                            metadatas = first.get("metadatas", []) or []
                            distances = first.get("distances", []) or []
                            ids = first.get("ids", []) or []
                    if not docs and "documents" in res:
                        doc_list = res.get("documents")
                        if isinstance(doc_list, list) and len(doc_list) > 0:
                            docs = doc_list[0] if isinstance(doc_list[0], list) else doc_list
                    if not metadatas and "metadatas" in res:
                        m_list = res.get("metadatas")
                        if isinstance(m_list, list) and len(m_list) > 0:
                            metadatas = m_list[0] if isinstance(m_list[0], list) else m_list
                    if not distances and "distances" in res:
                        d_list = res.get("distances")
                        if isinstance(d_list, list) and len(d_list) > 0:
                            distances = d_list[0] if isinstance(d_list[0], list) else d_list
                    if not ids and "ids" in res:
                        id_list = res.get("ids")
                        if isinstance(id_list, list) and len(id_list) > 0:
                            ids = id_list[0] if isinstance(id_list[0], list) else id_list

                    n = max(len(docs), len(metadatas), len(distances), len(ids))
                    for i in range(n):
                        try:
                            text = docs[i] if i < len(docs) else None
                        except Exception:
                            text = None
                        try:
                            meta = metadatas[i] if i < len(metadatas) else {}
                        except Exception:
                            meta = {}
                        try:
                            dist = distances[i] if i < len(distances) else None
                            if isinstance(dist, (list, tuple)) and len(dist) > 0:
                                dist = dist[0] if isinstance(dist[0], (int, float)) else dist
                        except Exception:
                            dist = None
                        try:
                            rid = ids[i] if i < len(ids) else None
                        except Exception:
                            rid = None

                        score = None
                        if isinstance(dist, (int, float)):
                            try:
                                score = float(1.0 - float(dist))
                            except Exception:
                                score = None

                        if text is None:
                            continue
                        r = {"text": text, "score": score if score is not None else 0.0, "meta": meta}
                        if rid:
                            r.setdefault("id", rid)
                        results.append(r)

                return results
            except Exception as e:
                logger.debug("Chroma query failed: %s", e)
                return []

        chroma_results = await asyncio.to_thread(_query_chroma)

        # FAISS candidates
        faiss_candidates = await asyncio.to_thread(_compute_scores)

        # Merge chroma + faiss: prefer chroma scores when present, dedupe by id or chunk_index
        merged: Dict[str, Dict[str, Any]] = {}

        def _key_from_meta(m):
            if not isinstance(m, dict):
                return None
            if "document_id" in m and "chunk_index" in m:
                return f"{m.get('document_id')}:{m.get('chunk_index')}"
            return None

        for r in chroma_results:
            key = r.get("id") or _key_from_meta(r.get("meta", {}))
            if not key:
                key = f"chroma:{len(merged)}"
            merged[key] = {"text": r.get("text", ""), "score": r.get("score", 0.0), "meta": r.get("meta", {}), "id": r.get("id")}

        for c in faiss_candidates:
            key = c.get("id") or _key_from_meta(c.get("meta", {}))
            if not key:
                key = f"faiss:{len(merged)}"
            existing = merged.get(key)
            if existing:
                # keep higher score
                if float(c.get("score", 0.0)) > float(existing.get("score", 0.0)):
                    merged[key]["score"] = float(c.get("score", 0.0))
                    merged[key]["meta"] = c.get("meta", merged[key]["meta"])
                    merged[key]["id"] = c.get("id", merged[key].get("id"))
            else:
                merged[key] = {"text": "", "score": float(c.get("score", 0.0)), "meta": c.get("meta", {}), "id": c.get("id")}

        # Create sorted list
        combined = sorted(list(merged.values()), key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]

        # Fill in text via Elasticsearch where missing
        es = get_es_client()
        results: List[Dict[str, Any]] = []
        for item in combined:
            text = item.get("text", "") or ""
            mdata = item.get("meta", {})
            if not text and es is not None and isinstance(mdata, dict) and "chunk_index" in mdata:
                doc_id = f"{document_id}:{mdata.get('chunk_index')}"
                try:
                    resp = await es.get(index="doc_chunks", id=doc_id)
                    # elastic_transport.ObjectApiResponse acts like a dict but isinstance(resp, dict) is False
                    # So we handle both dict and ObjectApiResponse types
                    if hasattr(resp, 'body'):
                        src_data = resp.body.get("_source") or resp.body.get("source") or {}
                    elif isinstance(resp, dict):
                        src_data = resp.get("_source") or resp.get("source") or {}
                    else:
                        # Try dict-like access (some response wrappers support this)
                        try:
                            src_data = resp.get("_source") or resp.get("source") or {}
                        except Exception:
                            src_data = {}
                    
                    text = src_data.get("text") or "" if isinstance(src_data, dict) else ""
                except Exception as e:
                    logger.debug("ES fetch failed: %s", e)
                    text = ""

            result = {"text": text, "score": float(item.get("score", 0.0)), "meta": mdata}
            if item.get("id"):
                result.setdefault("id", item.get("id"))
            if isinstance(mdata, dict) and "document_id" in mdata:
                result.setdefault("document_id", mdata.get("document_id"))
            results.append(result)

        return results

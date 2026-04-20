import logging
import io
import os
import asyncio
import tempfile
from typing import List, Dict, Any
from app.utils.chunker import chunk_text
from app.utils.embeddings import embed_texts
from app.infra.vectorstore.faiss_client import add_document_vectors
from app.infra.vectorstore.elasticsearch_client import get_es_client
from app.infra.vectorstore.chroma_client import get_text_collection

logger = logging.getLogger(__name__)


class IngestPipeline:
    """Thin ingestion pipeline: extract -> chunk -> embed -> index (best-effort).

    Designed to be lightweight and optional: will try to index into Chroma/Elasticsearch
    when available, otherwise it will store minimal metadata and return.
    """

    def __init__(self, db):
        self.db = db

    async def run(self, document_id: str, storage_name: str, content_type: str, filename: str) -> Dict[str, Any]:
        import time
        start_time = time.time()
        
        # Step 1: extract text (very thin)
        extract_start = time.time()
        text = await self._extract_text(storage_name, content_type)
        extract_time = time.time() - extract_start
        logger.info(f"[{document_id}] Extract: {extract_time:.2f}s")

        # Step 2: chunk
        chunks = chunk_text(text, chunk_size=300, overlap=50)

        # Step 3: embed (batched, with retries) - INCREASED batch size for better throughput
        embed_start = time.time()
        try:
            if len(chunks) > 0:
                # Ensure embedding vectors exist; embed_texts may be expensive so we batch
                # OPTIMIZATION: increased batch_size from 64 to 256 for better throughput
                batch_size = 256
                all_vectors = []
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    # simple retry for embed calls
                    attempts = 3
                    for attempt in range(attempts):
                        try:
                            from app.config.config import settings as config_settings

                            vs = embed_texts(batch, model_name=config_settings.EMBED_MODEL, dim=config_settings.EMBED_DIM)
                            all_vectors.extend(vs)
                            break
                        except Exception as e:
                            logger.debug("Embedding batch failed (attempt %d): %s", attempt + 1, e)
                            if attempt + 1 == attempts:
                                raise

                embed_time = time.time() - embed_start
                logger.info(f"[{document_id}] Embed {len(chunks)} chunks: {embed_time:.2f}s")

                ids = [f"{document_id}:{i}" for i in range(len(chunks))]
                metadatas = [{"document_id": document_id, "chunk_index": i, "filename": filename} for i in range(len(chunks))]

                # OPTIMIZATION: Run Chroma, FAISS, and Elasticsearch indexing in PARALLEL instead of sequentially
                async def _index_chroma():
                    try:
                        col = get_text_collection()
                        if col is not None:
                            try:
                                col.add(documents=chunks, embeddings=all_vectors, ids=ids, metadatas=metadatas)  # type:ignore
                                logger.debug(f"[{document_id}] Chroma indexing: OK")
                            except Exception as e:
                                logger.debug("Chroma add failed (non-fatal): %s", e)
                    except Exception:
                        logger.debug("Chroma client not available or add failed")

                def _persist_faiss_sync():
                    persist_attempts = 3
                    success = False
                    for pa in range(persist_attempts):
                        try:
                            ok = add_document_vectors(document_id, all_vectors, ids, metadatas, persist_to_minio=True)
                            success = bool(ok)
                            if success:
                                logger.debug(f"[{document_id}] FAISS+MinIO persistence: OK")
                                break
                        except Exception as e:
                            logger.debug("Persist vectors attempt %d failed: %s", pa + 1, e)
                    if not success:
                        logger.warning("Failed to persist vectors for document %s after %d attempts", document_id, persist_attempts)
                    return success

                async def _persist_faiss():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, _persist_faiss_sync)

                async def _index_elasticsearch():
                    try:
                        es = get_es_client()
                        if es is not None and len(chunks) > 0:
                            bulk = []
                            for i, c in enumerate(chunks):
                                doc = {"document_id": document_id, "chunk_index": i, "text": c}
                                bulk.append({"index": {"_index": "doc_chunks", "_id": f"{document_id}:{i}"}})
                                bulk.append(doc)
                            try:
                                # OPTIMIZATION: refresh=False during bulk, then refresh once at the end
                                await es.bulk(index="doc_chunks", body=bulk, refresh=False)
                                # Refresh index once after all documents are indexed
                                await es.indices.refresh(index="doc_chunks")
                                logger.debug(f"[{document_id}] Elasticsearch bulk indexing: OK ({len(chunks)} docs)")
                            except Exception as e:
                                logger.debug("Elasticsearch bulk failed: %s", e)
                    except Exception:
                        logger.debug("Elasticsearch client not available or bulk failed")

                # Run all three indexing operations in parallel
                index_start = time.time()
                await asyncio.gather(
                    _index_chroma(),
                    _persist_faiss(),
                    _index_elasticsearch(),
                    return_exceptions=True
                )
                index_time = time.time() - index_start
                logger.info(f"[{document_id}] Parallel indexing (Chroma+FAISS+ES): {index_time:.2f}s")
        except Exception as e:
            logger.debug("Vector persistence failed: %s", e)

        total_time = time.time() - start_time
        logger.info(f"[{document_id}] Total ingestion time: {total_time:.2f}s ({len(chunks)} chunks)")

        return {"document_id": document_id, "chunks": len(chunks)}

    async def _extract_text(self, storage_name: str, content_type: str) -> str:
        # Minimal content extraction: text files, PDF, Word, image OCR.
        try:
            from app.infra.storage.local_storage import local_storage

            data = await local_storage.load(storage_name)
            if data is None:
                return ""

            # Plain text
            if content_type.startswith("text") or content_type in ("application/json", "application/xml"):
                text = data.decode("utf-8", errors="ignore") if isinstance(data, (bytes, bytearray)) else str(data)
                return text

            # PDF extraction
            if content_type == "application/pdf" or storage_name.lower().endswith(".pdf"):
                try:
                    import pdfplumber
                except Exception:
                    return ""
                try: 
                    with pdfplumber.open(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data) as pdf: #type:ignore
                        pages = [p.extract_text() or "" for p in pdf.pages]
                        return "\n".join(pages).strip()
                except Exception as e:
                    logger.debug("PDF text extraction failed: %s", e)
                    return ""

            # Word extraction
            if content_type in ("application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document") or storage_name.lower().endswith(".docx"):
                try:
                    import docx
                except Exception:
                    return ""
                try:
                    doc = docx.Document(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else storage_name)
                    return "\n".join([p.text for p in doc.paragraphs if p.text]).strip()
                except Exception as e:
                    logger.debug("DOCX extraction failed: %s", e)
                    return ""

            # Image OCR fallback
            if content_type.startswith("image") or storage_name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                try:
                    from app.processors.extractors.image_extractor import extract_image_text
                    from PIL import Image
                except Exception:
                    return ""
                try:
                    # save to temp file just for OCR path convenience
                    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(storage_name)[1], delete=False) as tmp:
                        if isinstance(data, (bytes, bytearray)):
                            tmp.write(data)
                        tmp_path = tmp.name
                    text = extract_image_text(tmp_path)
                    os.remove(tmp_path)
                    return text
                except Exception as e:
                    logger.debug("Image OCR extraction failed: %s", e)
                    return ""

            return ""
        except Exception as e:
            logger.debug("Extract failed: %s", e)
            return ""


__all__ = ["IngestPipeline"]

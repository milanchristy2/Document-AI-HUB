from typing import Optional
from fastapi import APIRouter, Depends, BackgroundTasks, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import asyncio
import logging

from app.api.v1.deps.deps import get_db, get_token_payload
from app.pipelines.rag_pipeline import rag_pipeline
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.services.document_service import DocumentService
from app.processors.extractors.image_extractor import extract_image_text, extract_image_caption
from app.processors.extractors.audio_extractor import transcribe_audio
from app.utils.formatters import format_response
from app.utils.chains import build_evidence_blocks
from app.chains.rag_chain import call_llm_str

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: Optional[str] = None
    document_id: Optional[str] = None
    top_k: int = 5


class QueryRequest(BaseModel):
    query: Optional[str] = None
    document_id: Optional[str] = None
    session_id: Optional[str] = None  # NEW: Session tracking for conversation history
    stream: bool = False
    mode: Optional[str] = None
    provider: Optional[str] = None
    format: str = "json"
    top_k: int = 5
    explain: bool = False
    use_memory: bool = True  # NEW: Enable/disable conversation memory


router = APIRouter()


@router.post('/search')
async def search(
    payload: dict = Depends(get_token_payload),
    db=Depends(get_db),
    body: Optional[SearchRequest] = Body(None),
    query: Optional[str] = None,
    document_id: Optional[str] = None,
    top_k: Optional[int] = None,
):
    # prefer URL params first then JSON body
    q = query if query else (body.query if body and body.query else None)
    doc_id = document_id if document_id else (body.document_id if body and body.document_id else None)
    effective_top_k = top_k if top_k is not None else (body.top_k if body and body.top_k is not None else 6)

    if not q:
        return JSONResponse(status_code=400, content={"error": "query missing"})

    retriever = HybridRetriever('doc_chunks')
    results = await retriever.retrieve(q, document_id=doc_id, top_k=effective_top_k)
    return {"results": results}


@router.post('/ingest')
async def ingest(document_id: str, payload: dict = Depends(get_token_payload), db=Depends(get_db)):
    # Trigger re-ingestion for document if it exists
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        doc_service = DocumentService(db)
        doc = await doc_service.get_by_id(document_id, payload.get('sub') or '')
        if not doc:
            return JSONResponse(status_code=404, content={"error": "document not found"})

        from app.ingestion.pipeline import IngestPipeline
        storage_path = str(doc.storage_path)
        content_type = str(doc.content_type)
        filename = str(doc.filename)
        result = await IngestPipeline(db).run(document_id, storage_path, content_type, filename)
        return {"status": "ingested", "document_id": document_id, "result": result}
    except Exception as e:
        import traceback
        logger.exception(f"Ingest error for document {document_id}: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


@router.post('/query')
async def query(
    payload: dict = Depends(get_token_payload),
    db=Depends(get_db),
    body: QueryRequest = Body(...),
):
    from app.middleware.rate_limiter import check_rate_limit
    
    # Get user info from token
    user_id = payload.get('sub') or ''
    user_role = payload.get('role', 'user')
    
    # Check rate limit
    allowed, limit_msg, remaining = check_rate_limit(str(user_id), "query", str(user_role))
    if not allowed:
        return JSONResponse(status_code=429, content={"error": limit_msg, "remaining": remaining})
    
    # Only accept parameters from JSON body (QueryRequest). URL query params removed.
    q = body.query
    doc_id = body.document_id
    session_id = body.session_id  # NEW: Get session ID for conversation tracking
    stream_value = bool(body.stream)
    mode_value = body.mode
    provider_value = body.provider
    out_format = body.format or 'json'
    effective_top_k = body.top_k if body.top_k is not None else 6
    use_memory = body.use_memory if hasattr(body, 'use_memory') else True  # NEW: Check if memory is enabled

    if not q:
        return JSONResponse(status_code=400, content={"error": "query missing"})
    
    # NEW: Load conversation history if memory is enabled
    memory_service = None
    conversation_history = ""
    try:
        from app.services.memory_service import memory_service as _memory_service
        memory_service = _memory_service
        if use_memory and memory_service:
            hist = await memory_service.load(user_id, session_id)
            conversation_history = memory_service.format_for_prompt(hist)
    except Exception as e:
        logger.warning(f"Failed to load conversation memory: {e}")
        conversation_history = ""
    
    multimodal_context = {"user_role": payload.get('role'), "mode": mode_value}
    # NEW: Add conversation history to context
    if conversation_history:
        multimodal_context["conversation_history"] = conversation_history

    # Add explicit system instructions to improve LLM relevance and evidence citation
    multimodal_context.setdefault('system_instructions',
        "You are a precise assistant that answers questions based ONLY on provided documents. "
        "Be direct and factual. Cite sources when multiple documents are referenced. "
        "Do not speculate or infer beyond what's explicitly stated. "
        "If documents don't contain the answer, say: 'The provided documents do not contain information to answer this.'"
    )

    # if doc_id refers to image/audio, add OCR/transcript context
    if doc_id:
        doc_service = DocumentService(db)
        user_sub = str(payload.get('sub') or "")
        doc = await doc_service.get_by_id(doc_id, user_sub)
        if doc:
            content_type = str(doc.content_type).lower()
            storage_path = str(doc.storage_path)
            if content_type.startswith('image'):
                ocr_text = extract_image_text(storage_path)
                caption = extract_image_caption(storage_path)
                multimodal_context.update({'ocr_text': str(ocr_text), 'caption': str(caption)})
            elif content_type.startswith('audio'):
                transcript = transcribe_audio(storage_path)
                multimodal_context.update({'transcript': str(transcript)})

    if stream_value:
        async def event_generator():
            async for chunk in rag_pipeline.stream(q, file_id=doc_id, user_id=payload.get('sub'), multimodal_context=multimodal_context, provider=(provider_value or "ollama")):
                yield chunk
        return StreamingResponse(event_generator(), media_type='text/event-stream')

    # ========================================================================
    # NON-STREAMING: Step-by-step RAG process with quality checks
    # ========================================================================
    
    # Step 1: Initialize retriever and get evidence
    retriever = HybridRetriever('doc_chunks')
    
    # KEY IMPROVEMENT: Retrieve MORE chunks than displayed (2x for better LLM context)
    # This ensures LLM has richer context even if we only show top 5
    retrieval_top_k = max(effective_top_k * 2, 10)  # At least 10, preferably 2x requested
    logger.info(f"[RAG] Query: {q[:60]}... | Display Top-K: {effective_top_k} | Retrieve Top-K: {retrieval_top_k} | User: {user_id}")
    
    # Step 2: Retrieve chunks with quality metrics
    # IMPROVED: Retrieve more chunks to give LLM richer context
    evidence = await retriever.retrieve(q, document_id=doc_id, top_k=retrieval_top_k)
    
    retrieval_quality = {
        "total_chunks": len(evidence) if evidence else 0,
        "has_results": bool(evidence and len(evidence) > 0),
        "avg_score": 0.0,
        "query_decomposed": False,
    }
    
    if evidence:
        scores = [r.get('score', 0) or r.get('_score', 0) or 0 for r in evidence]
        retrieval_quality["avg_score"] = sum(scores) / len(scores) if scores else 0
        logger.info(f"[RAG] Retrieved: {len(evidence)} chunks, avg score: {retrieval_quality['avg_score']:.3f}")
    else:
        logger.warning(f"[RAG] No evidence retrieved for query: {q[:60]}")
    
    # Step 3: Generate answer using RAG chain with FULL evidence context
    # IMPROVED: Pass all retrieved chunks so LLM has maximum context
    answer = await rag_pipeline.rag_chain(
        q,
        retriever=retriever,
        top_k=len(evidence) if evidence else effective_top_k,  # Use all retrieved chunks
        provider=provider_value,
        multimodal_context=multimodal_context,
        reranker_obj=retriever.reranker,
        user_role=payload.get('role'),
    )
    
    # Step 4: Check if answer indicates insufficient information, but be smarter about it
    is_insufficient = (
        not answer or 
        len(answer.strip()) < 20 or
        "don't have enough information" in answer.lower() or 
        "insufficient" in answer.lower() or
        "[fallback" in answer.lower()
    )
    
    if is_insufficient and evidence and len(evidence) > 0:
        # Try one more time with a more targeted, directive prompt
        logger.info(f"[RAG] Initial answer too short/insufficient. Re-attempting synthesis with focused prompt.")
        
        # Re-combine evidence with all available context
        from app.chains.rag_chain import combine_evidence
        full_evidence = combine_evidence(evidence, max_chars=3000, deduplicate=False, prioritize_scores=True)
        
        # Use a more direct, imperative prompt to force synthesis
        synthesis_prompt = f"""Given the following documents, provide a DIRECT and COMPLETE answer to the question. 
Do not say "I don't have enough information". Instead, synthesize an answer from what IS available.
If some information is incomplete, state what you found and what is missing.

DOCUMENTS:
{full_evidence}

QUESTION: {q}

REQUIREMENTS:
- Start with a clear, direct answer
- Use information from the documents
- If asking for explanations or summaries, provide them from the documents
- Be specific and cite which documents you're referencing

ANSWER:"""
        
        try:
            retry_answer = await call_llm_str(synthesis_prompt, provider=provider_value)
            if retry_answer and len(retry_answer.strip()) > 30:
                answer = retry_answer
                is_insufficient = False  # Reset flag since we got a better answer
                logger.info(f"[RAG] Retry synthesis successful: {answer[:60]}...")
        except Exception as e:
            logger.warning(f"[RAG] Retry synthesis failed: {e}")
            pass  # Continue with original insufficient answer
    
    if is_insufficient and evidence and len(evidence) > 0:
        # IMPROVED: Instead of giving up, provide extracted summary from evidence
        logger.warning(f"[RAG] LLM couldn't synthesize. Extracting direct content from {len(evidence)} chunks.")
        
        # Extract FULL content directly from chunks with better field detection
        chunk_summaries = []
        for i, chunk in enumerate(evidence[:5], 1):  # Use top 5 chunks
            # Try multiple field names to find the actual text content
            text = (
                chunk.get('text') 
                or chunk.get('content') 
                or chunk.get('snippet') 
                or chunk.get('body')
                or chunk.get('data')
                or ""
            )
            
            # Also try to get heading/title for better formatting
            heading = (
                chunk.get('heading') 
                or chunk.get('title') 
                or chunk.get('section')
                or f"Section {i}"
            )
            
            # Use full text if available (much more helpful)
            if text and len(text.strip()) > 50:
                # Keep more content for better context (500+ chars instead of 300)
                chunk_text = text.strip()
                if len(chunk_text) > 800:
                    chunk_text = chunk_text[:800] + "..."
                
                chunk_summaries.append(f"**{heading}**\n\n{chunk_text}")
        
        if chunk_summaries:
            # Provide a concise, evidence-based summary when LLM can't synthesize
            # Format as a structured response instead of giving up
            answer = (
                "**Summary of Relevant Information**\n\n"
                + "\n\n---\n\n".join(chunk_summaries)
                + f"\n\n---\n\n**Your Question:** {q}\n\n"
                f"**Note:** The above represents the most relevant content from {len(evidence)} retrieved document(s). "
                f"The LLM could not generate a fully synthesized answer, so the raw document excerpts are provided above for your review."
            )
        else:
            # Fallback if extraction completely fails

            chunk_count = len(evidence)
            answer = (
                f"I found {chunk_count} relevant document{'s' if chunk_count > 1 else ''} "
                f"related to '{q[:40]}...'. Please review the evidence snippets below or try a more specific query."
            )
    elif is_insufficient and (not evidence or len(evidence) == 0):
        # No evidence at all
        answer = (
            f"No relevant documents were found for your query '{q}'. "
            f"Try:\n"
            f"  - Uploading documents related to your topic\n"
            f"  - Rephrasing your question with different keywords\n"
            f"  - Using broader search terms"
        )
    
    # Step 5: Build evidence blocks with enhanced metadata
    # IMPROVED: Show more evidence blocks (up to effective_top_k) for better context visibility
    from app.config.config import settings
    
    # Use the originally requested top_k for display, but evidence_blocks comes from all retrieved
    display_top_k = min(effective_top_k, len(evidence)) if evidence else 0
    
    _maybe_blocks = build_evidence_blocks(
        evidence,
        max_snippets=max(display_top_k, 5),  # Show at least 5 snippets even if user asks for fewer
        context_chars=settings.DEFAULT_CONTEXT_CHARS,
        include_scores=getattr(body, 'explain', False),
    )
    if asyncio.iscoroutine(_maybe_blocks):
        evidence_blocks = await _maybe_blocks
    elif hasattr(_maybe_blocks, "__aiter__"):
        evidence_blocks = [b async for b in _maybe_blocks]
    else:
        evidence_blocks = list(_maybe_blocks)  # type:ignore

    # Step 6: Extract document IDs and prepare metadata
    def _unique_preserve(seq):
        seen = set()
        out = []
        for s in seq:
            if s is None:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    doc_ids = [
        r.get('document_id') or r.get('doc_id') or r.get('source') 
        or r.get('index') or r.get('collection') 
        for r in (evidence or [])
    ]
    doc_ids = _unique_preserve(doc_ids)

    # Step 7: Extract retriever scores if explain mode is enabled
    retriever_scores = None
    if getattr(body, 'explain', False):
        retriever_scores = []
        for b in (evidence_blocks or []):
            score = None
            if isinstance(b, dict):
                score = b.get('meta', {}).get('score')
            else:
                meta = getattr(b, 'meta', None)
                if isinstance(meta, dict):
                    score = meta.get('score')
                else:
                    try:
                        score = getattr(meta, 'score')
                    except Exception:
                        score = None
            retriever_scores.append(score)

    logger.info(f"[RAG] Answer: {answer[:50]}... | Evidence blocks: {len(evidence_blocks or [])}")

    # Step 8: Format and return response
    if out_format == 'json':
        snippets = []
        for idx, b in enumerate((evidence_blocks or [])[:5], 1):
            snippet_obj = {
                "id": idx,
                "heading": b.get('heading', 'Unknown'),
                "snippet": b.get('snippet', b.get('text', '')),
                "source": b.get('source', 'unknown'),
            }
            # Add page/chunk metadata if available
            if b.get('meta'):
                meta = b['meta'] if isinstance(b['meta'], dict) else {}
                if 'page' in meta:
                    snippet_obj["page"] = meta['page']
                if 'chunk_index' in meta:
                    snippet_obj["chunk_index"] = meta['chunk_index']
                if 'score' in meta and getattr(body, 'explain', False):
                    snippet_obj["relevance_score"] = meta['score']
            snippets.append(snippet_obj)

        resp = {
            "answer": answer,
            "source_file": doc_ids[0] if doc_ids else doc_id,
            "snippets": snippets,
            "metadata": {
                "document_ids": doc_ids,
                "retrieval_quality": {
                    "chunks_found": retrieval_quality["total_chunks"],
                    "avg_relevance": round(retrieval_quality["avg_score"], 3),
                },
                "answer_quality": {
                    "has_evidence": len(evidence_blocks or []) > 0,
                    "evidence_count": len(evidence_blocks or []),
                },
            },
        }

        if retriever_scores is not None:
            resp["metadata"]["retriever_scores"] = retriever_scores

        return resp

    # Format as markdown/xml
    formatted = format_response(answer, evidence_blocks, style=out_format)  # type:ignore
    if out_format == 'json' and isinstance(formatted, dict):
        formatted['metadata'] = formatted.get('metadata', {}) or {}
        formatted['metadata'].update({
            'document_ids': doc_ids,
            'retrieval_quality': {
                'chunks_found': retrieval_quality["total_chunks"],
                'avg_relevance': round(retrieval_quality["avg_score"], 3),
            }
        })
        if retriever_scores is not None:
            formatted['metadata']['retriever_scores'] = retriever_scores
    return formatted

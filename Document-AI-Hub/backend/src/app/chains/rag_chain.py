import asyncio
import logging
import re
from dataclasses import dataclass
from typing import AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Any

from app.utils.embeddings import embed_texts
from app.config.config import settings
from app.processors.processors import build_multimodal_context, preprocess_text
from app.core import rbac
from app.infra.llm.ollama_client import call_ollama
from app.ai.llm.openrouter_provider import OpenRouterProvider
from app.ai.llm.groq_provider import GroqProvider
from langchain_core.utils.function_calling import convert_to_json_schema #type:ignore

logger = logging.getLogger(__name__)

# We use Ollama (`phi3:mini`) as the primary LLM backend. OpenAI support
# was intentionally removed per project policy to avoid external dependency.

@dataclass
class LLMResponse:
    text: str
    provider_requested: str
    provider_used: str
    fallback: bool = False


class ProviderUnavailable(Exception):
    """Raised when provider configuration is missing or incomplete."""


PROVIDER_PRIORITY = ["ollama", "openrouter", "groq"]

ProviderHandler = Callable[[str], Awaitable[Optional[str]]]

def _normalize_provider(provider: Optional[str]) -> str:
    candidate = (provider or settings.DEFAULT_PROVIDER or "ollama").lower()
    return candidate if candidate in PROVIDER_PRIORITY else "ollama"


def _provider_order(requested: str) -> List[str]:
    return [requested] + [p for p in PROVIDER_PRIORITY if p != requested]


async def _call_ollama(prompt: str) -> str:
    return await asyncio.to_thread(call_ollama, prompt, settings.OLLAMA_MODEL)


async def _call_openrouter(prompt: str) -> str:
    api_key = getattr(settings, "OPENROUTER_API_KEY", "")
    if not api_key:
        raise ProviderUnavailable("OpenRouter API key missing")
    orp = OpenRouterProvider(api_key)
    return await orp.generate(prompt)


async def _call_groq(prompt: str) -> str:
    api_key = getattr(settings, "GROQ_API_KEY", "")
    if not api_key:
        raise ProviderUnavailable("Groq API key missing")
    gp = GroqProvider(api_key)
    return await gp.generate(prompt)


PROVIDER_HANDLERS: Dict[str, ProviderHandler] = {
    "ollama": _call_ollama,
    "openrouter": _call_openrouter,
    "groq": _call_groq,
}


PROMPT_TEMPLATE = """You are a precise AI assistant that answers questions using ONLY information explicitly stated in the provided documents.

## CORE INSTRUCTIONS:
1. **Answer First**: Directly answer the question in your opening sentence(s). Be specific and factual.
2. **Use Evidence**: Ground every claim in the provided evidence. If citing multiple parts, reference them clearly.
3. **No Speculation**: DO NOT infer, assume, or add information not explicitly in the documents.
4. **If Insufficient**: If the evidence doesn't clearly address the question, respond with: "The provided documents do not contain information to answer this question."
5. **Format Well**: Use natural language with clear structure (bold for key terms, lists where helpful, paragraphs for narrative).

## EVIDENCE FROM DOCUMENTS:
{evidence}

## USER QUESTION:
{question}

## YOUR ANSWER:
(Start directly with the answer; do not repeat the question or use preamble.)"""

VERIFICATION_PROMPT_TEMPLATE = """You are a meticulous fact-checker. Your task is to verify if the provided "Answer" is FULLY supported by the "Evidence".

- The Answer MUST NOT contain any information, claims, or inferences that are not explicitly stated in the Evidence.
- Every single point in the Answer must be traceable back to the Evidence.

Is the Answer fully supported by the Evidence? Answer with a single word: "yes" or "no".

Evidence:
{evidence}

Answer:
{answer}

Verification (yes/no):"""


async def call_llm_with_metadata(prompt: str, provider: Optional[str] = None) -> LLMResponse:
    requested = _normalize_provider(provider)
    for candidate in _provider_order(requested):
        handler = PROVIDER_HANDLERS.get(candidate)
        if not handler:
            continue
        try:
            text = await handler(prompt)
        except ProviderUnavailable as exc:
            logger.debug("Skipping provider %s: %s", candidate, exc)
            continue
        except Exception:
            logger.warning("Provider %s failed", candidate, exc_info=True)
            continue
        if text:
            return LLMResponse(text=text.strip(), provider_requested=requested, provider_used=candidate, fallback=False)
    fallback_text = _create_fallback_from_evidence(prompt)
    return LLMResponse(text=fallback_text, provider_requested=requested, provider_used="fallback", fallback=True)


async def call_llm_str(prompt: str, provider: Optional[str] = None) -> str:
    return (await call_llm_with_metadata(prompt, provider=provider)).text


async def call_llm_stream(prompt: str, provider: Optional[str] = None) -> AsyncGenerator[str, None]:
    response = await call_llm_with_metadata(prompt, provider=provider)
    if response.fallback:
        logger.debug("Streaming fallback text (requested=%s)", response.provider_requested)
    tokens = response.text.split()
    if not tokens and response.text:
        tokens = [response.text]
    for tok in tokens:
        yield tok + " "
        await asyncio.sleep(0)


def combine_evidence(results: List[Dict], max_chars: int = 2500, deduplicate: bool = True, prioritize_scores: bool = True) -> str:
    """Combine retrieved evidence with deduplication, scoring prioritization, and formatting.
    
    Args:
        results: List of retrieval results (dicts with text, heading, score, etc.)
        max_chars: Maximum total character length (increased to 2500 for better LLM context)
        deduplicate: Remove near-duplicate snippets
        prioritize_scores: Sort by relevance score first to surface best evidence
    
    Returns:
        Formatted evidence string with highest-confidence snippets first, clearly numbered
    """
    # Sort by score/relevance if available to prioritize best evidence
    if prioritize_scores:
        scored_results = []
        for r in results:
            score = r.get('score') or r.get('_score') or r.get('similarity') or r.get('reranker_score') or 0
            scored_results.append((score, r))
        scored_results.sort(key=lambda x: x[0], reverse=True)
        results = [r for _, r in scored_results]
    
    parts: List[str] = []
    chars = 0
    seen_snippets = set()
    snippet_count = 0
    
    for idx, r in enumerate(results):
        heading = r.get("heading") or r.get("title") or r.get("section") or ""
        text = (r.get("text") or r.get("content") or "").strip()
        
        if not text or len(text) < 10:  # Skip very short snippets
            continue
        
        # Deduplication: skip if we've seen a similar snippet
        if deduplicate:
            snippet_key = text[:150].lower()  # Use first 150 chars as dedup key
            if snippet_key in seen_snippets:
                continue
            seen_snippets.add(snippet_key)
        
        snippet_count += 1
        # Format with clear numbering for LLM reference
        if heading:
            piece = f"[Document {snippet_count}: {heading}]\n{text}"
        else:
            piece = f"[Document {snippet_count}]\n{text}"
        
        # Truncate to max_chars
        if chars + len(piece) > max_chars:
            remaining = max_chars - chars
            if remaining > 150:
                piece = piece[:remaining].rsplit(" ", 1)[0] + "..."
            else:
                break
        
        parts.append(piece)
        chars += len(piece) + 3  # Account for newlines
    
    if not parts:
        return "(No relevant documents found)"
    
    return "\n\n".join(parts)


def build_evidence_blocks(results: List[Dict], max_snippets: int = 6, context_chars: int = 1000, include_scores: bool = False) -> List[Dict]:
    blocks: List[Dict] = []
    if not results:
        return blocks

    def _trim_to_sentences(text: str, limit: int) -> str:
        if not text:
            return ""
        if len(text) <= limit:
            return text
        sents = re.split(r'(?<=[.!?])\s+', text)
        out = []
        cur = 0
        for s in sents:
            if cur + len(s) > limit:
                break
            out.append(s)
            cur += len(s) + 1
        if not out:
            return text[:limit].rsplit(' ', 1)[0] + '...'
        return ' '.join(out).strip()

    for i, r in enumerate(results[:max_snippets]):
        text = (r.get('text') or r.get('content') or '').strip()
        heading = r.get('heading') or r.get('title') or f'Chunk {i+1}'
        source = r.get('source') or r.get('index') or r.get('document_id') or 'unknown'
        snippet = _trim_to_sentences(text, context_chars)
        page = r.get('page') or r.get('page_no') or None
        chunk_index = r.get('chunk_index') or r.get('index_in_doc') or None
        score = r.get('score') if 'score' in r else (r.get('_score') if '_score' in r else (r.get('similarity') if 'similarity' in r else (r.get('distance') if 'distance' in r else None)))
        meta = {k: v for k, v in r.items() if k not in ('text', 'content')}
        if page is not None:
            meta['page'] = page
        if chunk_index is not None:
            meta['chunk_index'] = chunk_index
        if include_scores and score is not None:
            meta['score'] = score

        block = {
            'id': r.get('id') or r.get('_id') or f'{i}',
            'heading': heading,
            'snippet': snippet,
            'text': text,
            'source': source,
            'meta': meta,
        }
        blocks.append(block)
    return blocks


def _parse_evidence_and_question(prompt: str) -> tuple[str, str]:
    evidence = ""
    question = ""
    try:
        if "EVIDENCE:" in prompt and "QUESTION:" in prompt:
            parts = prompt.split("EVIDENCE:", 1)[1]
            evidence, remainder = parts.split("QUESTION:", 1)
            question = remainder.split("ANSWER:", 1)[0].strip()
            evidence = evidence.strip()
    except Exception:
        pass
    return evidence, question


def _create_fallback_from_evidence(prompt: str) -> str:
    """Create a fallback response when no LLM is available, based on evidence.
    
    Only returns extractive snippets if evidence is substantial enough.
    """
    evidence, question = _parse_evidence_and_question(prompt)
    if not evidence or len(evidence) < 50:
        return "I don't have enough information to answer that question accurately."
    
    # Extract first few sentences as a minimal response
    sent = re.split(r'(?<=[.!?])\s+', evidence)
    top_snip = " ".join(sent[:3]).strip() if sent else evidence
    
    if len(top_snip) < 30:
        return "I don't have enough information to answer that question accurately."
    
    return f"[Fallback Answer] {top_snip}"


async def qa_chain(query: str, retriever, top_k: int = 6, reranker=None) -> str:
    # Decompose query into targeted sub-questions
    sub_questions = await decompose_query(query)
    all_results = []
    
    for sub_q in sub_questions:
        try:
            sub_results = await retriever.retrieve(sub_q, top_k=top_k)
            all_results.extend(sub_results)
        except Exception as e:
            logger.debug(f"Retrieval for sub-question '{sub_q}' failed: {e}")
            continue
    
    if not all_results:
        return "No relevant content found."
    
    # Deduplicate results by text content
    seen_texts = set()
    deduped_results = []
    for r in all_results:
        text_key = (r.get("text") or r.get("content") or "")[:200].lower()
        if text_key and text_key not in seen_texts:
            seen_texts.add(text_key)
            deduped_results.append(r)
        elif not text_key:
            deduped_results.append(r)
    
    results = deduped_results[:top_k * 2] if deduped_results else []
    
    texts = [r.get("text") or r.get("content") or "" for r in results]
    if reranker and texts:
        try:
            ranks = await reranker.rank(query, texts, top_n=min(len(texts), top_k))
            idxs = [i for i, _ in ranks]
            results = [results[i] for i in idxs]
        except Exception:
            logger.exception("Reranking failed; continuing with original order")

    evidence = combine_evidence(results)
    prompt = PROMPT_TEMPLATE.format(evidence=evidence, question=query)
    answer = await call_llm_str(prompt)
    if answer.startswith("[fallback-answer]") and evidence:
        distilled = " ".join(evidence.split()[:80])
        if distilled:
            return f"[fallback-answer] Extractive response: {distilled}"
    return answer


async def qa_chain_stream(query: str, retriever, top_k: int = 6, reranker=None) -> AsyncGenerator[str, None]:
    # Decompose query into targeted sub-questions
    sub_questions = await decompose_query(query)
    all_results = []
    
    for sub_q in sub_questions:
        try:
            sub_results = await retriever.retrieve(sub_q, top_k=top_k)
            all_results.extend(sub_results)
        except Exception as e:
            logger.debug(f"Retrieval for sub-question '{sub_q}' failed: {e}")
            continue
    
    if not all_results:
        yield "No relevant content found."
        return
    
    # Deduplicate results by text content
    seen_texts = set()
    deduped_results = []
    for r in all_results:
        text_key = (r.get("text") or r.get("content") or "")[:200].lower()
        if text_key and text_key not in seen_texts:
            seen_texts.add(text_key)
            deduped_results.append(r)
        elif not text_key:
            deduped_results.append(r)
    
    results = deduped_results[:top_k * 2] if deduped_results else []
    
    texts = [r.get("text") or r.get("content") or "" for r in results]
    if reranker and texts:
        try:
            ranks = await reranker.rank(query, texts, top_n=min(len(texts), top_k))
            idxs = [i for i, _ in ranks]
            results = [results[i] for i in idxs]
        except Exception:
            logger.exception("Reranking failed; continuing with original order")

    evidence = combine_evidence(results)
    prompt = PROMPT_TEMPLATE.format(evidence=evidence, question=query)

    async for token in call_llm_stream(prompt):
        yield token


async def pre_retrieval(query: str, multimodal_context: Optional[Dict] = None) -> str:
    """Enhance query with context and prepare for retrieval."""
    if not multimodal_context:
        return query
    
    parts = [query]
    
    # Add multimodal context if available
    caption = multimodal_context.get("caption")
    if caption:
        parts.append(f"[caption] {caption}")
    
    ocr_text = multimodal_context.get("ocr_text")
    if ocr_text:
        # Truncate long OCR text
        ocr_snippet = ocr_text[:200] if len(ocr_text) > 200 else ocr_text
        parts.append(f"[ocr] {ocr_snippet}")
    
    transcript = multimodal_context.get("transcript")
    if transcript:
        transcript_snippet = transcript[:200] if len(transcript) > 200 else transcript
        parts.append(f"[transcript] {transcript_snippet}")
    
    return " ".join(parts)


async def decompose_query(query: str, provider: Optional[str] = None) -> List[str]:
    """Decompose a complex query into simpler sub-questions for targeted retrieval.
    
    Returns a list of sub-questions that cover different aspects of the original query.
    """
    decomposition_prompt = f"""Break down the following query into 2-4 focused sub-questions that together would answer the original query comprehensively. Each sub-question should target a specific aspect.

Original Query: {query}

Return only the sub-questions as a numbered list, one per line. Example format:
1. What are the main responsibilities?
2. What skills are required?
3. What is the career path?
"""
    
    try:
        decomposition = await call_llm_str(decomposition_prompt, provider=provider)
        # Parse the decomposition response into individual questions
        lines = decomposition.split('\n')
        sub_questions = []
        for line in lines:
            # Extract text after number and period
            line = line.strip()
            if line and any(c.isdigit() for c in line[:3]):
                # Remove leading number, dots, and spaces
                q = re.sub(r'^\d+[\.\)\s]+', '', line).strip()
                if q and len(q) > 5:  # Ensure it's a meaningful question
                    sub_questions.append(q)
        
        # If decomposition failed or returned empty, fall back to simple expansion
        if not sub_questions:
            return await expand_query(query)
        
        return sub_questions
    except Exception as e:
        logger.debug(f"Query decomposition failed: {e}. Falling back to expansion.")
        return await expand_query(query)


async def generate_hyde(query: str, provider: Optional[str] = None) -> str:
    """Generate a HYDE-style synthetic document (short) to improve retrieval recall.

    Returns a short, factual-sounding one-sentence summary that can be used as
    an additional retrieval query. Falls back to the original query on error.
    """
    hyde_prompt = f"""Create a one-sentence factual summary of the information someone might write to answer: \"{query}\". Keep it concise (<= 30 words)."""
    try:
        hyde = await call_llm_str(hyde_prompt, provider=provider)
        hyde = (hyde or "").strip().split('\n')[0]
        if hyde:
            return hyde
    except Exception as e:
        logger.debug(f"HYDE generation failed: {e}")
    return query


async def expand_query(query: str) -> List[str]:
    """Generate query variations for better retrieval coverage.
    
    Returns multiple query forms to improve recall:
    - Original query
    - Simplified version (remove stopwords)
    - Synonym-based expansion (if available)
    """
    queries = [query]
    
    # Simple expansion: try removing common question words
    simplified = query.lower()
    for word in ["what is", "how do", "why is", "who are", "where is", "when did", "can you"]:
        simplified = simplified.replace(word, "").strip()
    
    if simplified and simplified != query.lower():
        queries.append(simplified)
    
    # Add a question-focused version if not already a question
    if not query.strip().endswith("?"):
        queries.append(f"Tell me about {query}")
    
    return list(dict.fromkeys(queries))  # Remove duplicates preserving order


async def pre_augmentation(results: List[Dict], multimodal_context: Optional[Dict] = None) -> List[Dict]:
    """Enhance retrieved results with multimodal context at the beginning.
    
    Prepends high-confidence multimodal evidence (OCR, captions, transcripts)
    to the retrieved results, giving them priority in the evidence chain.
    """
    if not multimodal_context:
        return results
    
    augmented = list(results)
    
    # Prepend OCR text if available
    ocr = multimodal_context.get("ocr_text")
    if ocr and isinstance(ocr, str) and ocr.strip():
        augmented.insert(0, {
            "text": ocr.strip(),
            "source": "ocr",
            "heading": "OCR Extracted Text",
            "score": 1.0  # High confidence
        })
    
    # Prepend caption if available
    caption = multimodal_context.get("caption")
    if caption and isinstance(caption, str) and caption.strip():
        augmented.insert(0, {
            "text": caption.strip(),
            "source": "image_caption",
            "heading": "Image Caption",
            "score": 0.95  # High confidence
        })
    
    # Prepend transcript if available
    transcript = multimodal_context.get("transcript")
    if transcript and isinstance(transcript, str) and transcript.strip():
        augmented.insert(0, {
            "text": transcript.strip(),
            "source": "audio_transcript",
            "heading": "Audio Transcript",
            "score": 0.95  # High confidence
        })
    
    return augmented


async def post_augmentation_rerank(results: List[Dict], reranker_obj, query: str, top_k: int = 6) -> List[Dict]:
    """Rerank results with confidence scoring and filtering.
    
    Args:
        results: Retrieved results to rerank
        reranker_obj: Reranker model (supports async rank method)
        query: Original query for relevance scoring
        top_k: Maximum results to return
    
    Returns:
        Reranked and filtered results
    """
    if not reranker_obj or not results:
        return results[:top_k] if len(results) > top_k else results
    
    texts = [r.get("text") or "" for r in results if r.get("text")]
    if not texts:
        return results[:top_k]
    
    try:
        # Call reranker with all texts
        ranks = await reranker_obj.rank(query, texts, top_n=min(len(texts), top_k))
        
        if not ranks:
            return results[:top_k]
        
        # Map back to original results preserving metadata
        idxs = [i for i, score in ranks]
        ordered = [results[i] for i in idxs if i < len(results)]
        
        # Store reranker scores in results
        for result, (idx, score) in zip(ordered, ranks[:len(ordered)]):
            if "reranker_score" not in result:
                result["reranker_score"] = score
        
        logger.debug(f"Reranked {len(ordered)} results for query: {query[:50]}...")
        return ordered
    except Exception as e:
        logger.warning(f"Reranking failed: {e}; returning original order")
        return results[:top_k]


async def verify_answer(answer: str, evidence: str, provider: Optional[str] = None) -> bool:
    """Verify if the answer is fully supported by the evidence."""
    if not answer or not evidence:
        return False
    prompt = VERIFICATION_PROMPT_TEMPLATE.format(evidence=evidence, answer=answer)
    try:
        verification_response = await call_llm_str(prompt, provider=provider)
        return "yes" in verification_response.lower()
    except Exception as e:
        logger.warning(f"Answer verification failed: {e}")
        return False  # Default to unverified if the check fails


async def rag_chain(
    query: str,
    retriever,
    top_k: int = 6,
    multimodal_context: Optional[Dict] = None,
    reranker_obj=None,
    user_role: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    if not user_role and multimodal_context and isinstance(multimodal_context, dict):
        user_role = multimodal_context.get("user_role")
    if user_role:
        rbac.require_permission(user_role, "query", resource="document")

    q2 = await pre_retrieval(query, multimodal_context)
    
    # Decompose query into sub-questions for comprehensive retrieval
    sub_questions = await decompose_query(q2, provider=provider)
    all_results = []
    
    for sub_q in sub_questions:
        try:
            sub_results = await retriever.retrieve(sub_q, top_k=top_k)
            all_results.extend(sub_results)
        except Exception as e:
            logger.debug(f"Retrieval for sub-question '{sub_q}' failed: {e}")
            continue
    
    if not all_results:
        # Fallback: try original query
        try:
            all_results = await retriever.retrieve(q2, top_k=top_k)
        except Exception as e:
            logger.warning(f"Fallback retrieval failed: {e}")
            return "I don't have enough information to answer that question accurately."
    
    # Deduplicate by text content
    seen_texts = set()
    deduped_results = []
    for r in all_results:
        text_key = (r.get("text") or r.get("content") or "")[:200].lower()
        if text_key and text_key not in seen_texts:
            seen_texts.add(text_key)
            deduped_results.append(r)
        elif not text_key:
            deduped_results.append(r)
    
    # Use top_k * 2 best results, prioritizing by score
    results = deduped_results[:top_k * 2] if deduped_results else []
    results = await pre_augmentation(results, multimodal_context)
    results = await post_augmentation_rerank(results, reranker_obj, q2, top_k=top_k)
    
    # Combine evidence with score prioritization for quality
    evidence = combine_evidence(results, prioritize_scores=True)
    
    # Validate evidence quality before sending to LLM
    if not evidence or len(evidence) < 50:
        return "I don't have enough information to answer that question accurately."
    
    prompt = PROMPT_TEMPLATE.format(evidence=evidence, question=query)
    answer = await call_llm_str(prompt, provider=provider)

    # Step 2: Verify the answer against the evidence
    is_verified = await verify_answer(answer, evidence, provider=provider)

    # Step 3: Handle unverified answers more gently
    if not is_verified:
        logger.warning("Answer verification flagged potential unsourced details, but returning it anyway.")
        # Optionally, you can prepend a disclaimer:
        # return f"**Disclaimer: The answer below could not be strictly verified against chunks.**\n\n{answer}"
    
    # Clean up any fallback or evidence markers from the answer
    cleaned = answer
    if cleaned.startswith("[Fallback Answer]"):
        cleaned = cleaned.replace("[Fallback Answer] ", "").strip()
    
    # Remove any [Evidence N] markers that the LLM might have copied
    cleaned = re.sub(r'\[Evidence \d+\]\s*', '', cleaned).strip()
    
    return cleaned


async def summarization_chain(results: List[Dict], summarizer_prompt: Optional[str] = None) -> str:
    evidence = combine_evidence(results, max_chars=3000)
    prompt = (summarizer_prompt or "Summarize the following evidence briefly:\n{evidence}").format(evidence=evidence)
    return await call_llm_str(prompt)


# LangChain helpers
def convert_to_json_schema(pydantic_model: Any) -> Dict:
    """Return a JSON schema for a Pydantic model. If LangChain offers extra conversion utilities, use them."""
    try:
        import langchain
        if hasattr(langchain, "convert_json_schema"):
            return convert_to_json_schema(pydantic_model)
    except Exception:
        pass
    try:
        return pydantic_model.schema()
    except Exception:
        return {}


def format_response_to_json_markdown(answer: str, evidence_blocks: List[Dict], format_type: str = "json") -> Dict[str, Any]:
    """Produce response in multiple formats: JSON, Markdown, or XML.
    
    Args:
        answer: The LLM-generated answer
        evidence_blocks: List of evidence/snippet dictionaries
        format_type: "json", "markdown", or "xml"
    
    Returns:
        Dictionary with formatted response
    """
    payload = {"answer": answer, "evidence": evidence_blocks, "format": format_type}
    
    # Build markdown table
    rows = ["| # | Heading | Source | Snippet |", "|---|---|---|---|"]
    for i, b in enumerate(evidence_blocks, 1):
        heading = (b.get("heading") or "").replace("|", "\\|")
        src = (b.get("source") or "").replace("|", "\\|")
        snip = (b.get("snippet") or "").replace("|", "\\|")[:100]
        rows.append(f"| {i} | {heading} | {src} | {snip} |")
    
    markdown_output = f"# Answer\n\n{answer}\n\n## Evidence\n\n" + "\n".join(rows)
    payload["evidence_md_table"] = markdown_output
    
    # Build XML format
    if format_type == "xml" or format_type == "all":
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<response>", f"<answer>{answer}</answer>", "<evidence>"]
        for i, b in enumerate(evidence_blocks, 1):
            xml_parts.append(f'<snippet id="{i}">')
            xml_parts.append(f'<heading>{b.get("heading", "")}</heading>')
            xml_parts.append(f'<source>{b.get("source", "")}</source>')
            xml_parts.append(f'<text>{b.get("snippet", "")}</text>')
            xml_parts.append('</snippet>')
        xml_parts.append("</evidence>")
        xml_parts.append("</response>")
        payload["xml_output"] = "\n".join(xml_parts)
    
    return payload
    
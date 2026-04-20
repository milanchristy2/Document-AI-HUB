import asyncio
import logging
import re
from typing import AsyncGenerator, List, Tuple, Dict, Optional
from uuid import uuid4

from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.utils.chains import qa_chain_stream
from app.ai.guardrails.input_guard import input_guardrail
from app.ai.guardrails.output_guard import output_guardrail
from app.ai.system_prompts.init import get_system_prompt
from app.config.config import settings
from app.chains.rag_chain import (
    combine_evidence,
    expand_query,
    decompose_query,
    generate_hyde,
    pre_retrieval,
    pre_augmentation,
    post_augmentation_rerank,
    call_llm_stream,
    call_llm_str,
    call_llm_with_metadata,
    verify_answer,
)

# Agent integration
try:
    from app.agents.rag_agent import RAGAgent
    from app.agents.router_agent import RouterAgent
    from app.agents.base_agent import AgentConfig, AgentInput, ExecutionContext, AgentType
    AGENTS_AVAILABLE = True
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.debug(f"Agents not available: {e}")
    AGENTS_AVAILABLE = False

try:
    from app.common_utils.intent_detector import detect_intent
except Exception:
    # fallback stub when intent detector is not present
    class _Det:
        def __init__(self, intent: str = "answer"):
            self.intent = intent

    def detect_intent(_q: str) -> _Det:
        return _Det("answer")

try:
    from app.services.cache_service import cache_service
except Exception:
    cache_service = None

try:
    from app.services.memory_service import memory_service
except Exception:
    memory_service = None

try:
    from app.tasks.conversation_tasks import persist_turn
except Exception:
    persist_turn = None

logger = logging.getLogger(__name__)


class RagPipeline:
    def __init__(self, elastic_index: str = "doc_chunks"):
        self.retriever = HybridRetriever(elastic_index)
        
        # Initialize agents if available
        self.agents_enabled = AGENTS_AVAILABLE
        self.rag_agent = None
        self.router_agent = None
        
        if self.agents_enabled:
            try:
                # Initialize RAG Agent
                rag_config = AgentConfig(
                    name="RAGAgent",
                    agent_type=AgentType.SEARCH,
                    description="RAG agent for document retrieval and synthesis",
                    version="1.0.0"
                )
                self.rag_agent = RAGAgent(rag_config, retriever=self.retriever)
                
                # Initialize Router Agent
                router_config = AgentConfig(
                    name="RouterAgent",
                    agent_type=AgentType.ROUTE,
                    description="Routes queries to appropriate agents",
                    version="1.0.0"
                )
                self.router_agent = RouterAgent(router_config)
                self.router_agent.set_rag_agent(self.rag_agent)
                
                logger.info("✓ Agents initialized successfully in RagPipeline")
            except Exception as e:
                logger.warning(f"Failed to initialize agents: {e}")
                self.agents_enabled = False

    async def rag_chain(self, query: str, retriever=None, top_k: int = 6, multimodal_context: dict | None = None, reranker_obj=None, user_role: str | None = None, provider: str | None = None) -> str:
        """Thin adapter to the utils.rag_chain implementation.

        Allows callers to use a pipeline-style object while the heavy lifting remains in `app.utils.chains`.
        """
        try:
            from app.utils.chains import rag_chain as _rag_chain
        except Exception:
            # if chains unavailable, provide a simple fallback answer
            async def _rag_chain(q, r, top_k=6, multimodal_context=None, reranker_obj=None, user_role=None):
                return ""
        r = retriever or self.retriever
        # pass provider through to the underlying implementation if it supports it
        try:
            return await _rag_chain(query, r, top_k=top_k, multimodal_context=multimodal_context, reranker_obj=reranker_obj, user_role=user_role, provider=provider)
        except TypeError:
            # older implementations may not accept provider; fall back
            return await _rag_chain(query, r, top_k=top_k, multimodal_context=multimodal_context, reranker_obj=reranker_obj, user_role=user_role)

    async def _transform_query(self, query: str, provider: str, rewrite: bool, decompose: bool, use_hyde: bool) -> List[str]:
        """Deprecated: use expand_query from rag_chain instead."""
        return await expand_query(query)

    async def _retrieve(self, query: str, search_queries: List[str], file_id: str | None) -> Tuple[List[Dict], List[Dict]]:
        """Deprecated: retrieve is now done directly in stream()."""
        chunks = await self.retriever.retrieve(query, document_id=file_id, top_k=12)
        metas = [{'source': 'elastic_or_vector'} for _ in chunks]
        return chunks, metas

    async def _rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Deprecated: reranking is now done in post_augmentation_rerank."""
        return chunks

    def _apply_context_window(self, chunks: List[Dict], max_chunks: int = 6) -> List[Dict]:
        # lost-in-middle fix: keep best[0], second best at end, and fill with others
        if not chunks:
            return []
        selected = []
        if len(chunks) <= max_chunks:
            return chunks
        # assume chunks sorted by relevance desc
        best = chunks[0]
        rest = chunks[1:]
        tail = rest[: max_chunks - 2]
        last = rest[-1]
        return [best] + tail + [last]

    def _format_evidence(self, chunks: List[Dict], cite: bool = True) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            heading = c.get('heading') or c.get('title') or ''
            page = c.get('page')
            loc = f", p.{page}" if page else ''
            text = (c.get('text') or c.get('content') or '').strip()
            parts.append(f"[{i}] ({heading}{loc}):\n{text}")
        return "\n\n".join(parts)

    async def _use_agent_retrieval(self, query: str, user_id: str | None, session_id: str | None, file_id: str | None, role: str | None = None) -> Dict:
        """Use agent-based retrieval when document retrieval returns no results.
        
        This method leverages the RAG agent's reflection pattern and tool-based fallback
        to answer queries that aren't found in the knowledge base.
        """
        if not self.agents_enabled or not self.rag_agent:
            return {
                "success": False,
                "answer": "Agent-based retrieval unavailable",
                "confidence": 0.0
            }
        
        try:
            # Create agent input
            agent_input = AgentInput(
                query=query,
                user_id=user_id or "unknown",
                session_id=session_id or str(uuid4()),
                document_id=file_id,
                parameters={
                    "top_k": 5,
                    "user_role": role or "general"
                }
            )
            
            # Create execution context
            context = ExecutionContext(
                execution_id=str(uuid4()),
                user_id=user_id or "unknown",
                session_id=session_id or str(uuid4())
            )
            
            # Execute RAG agent
            logger.info(f"Using agent-based retrieval for: {query}")
            result = await self.rag_agent.execute(agent_input, context)
            
            if result.status.name == "COMPLETED" and result.result:
                agent_result = result.result
                return {
                    "success": True,
                    "answer": agent_result.get("answer", ""),
                    "confidence": agent_result.get("confidence", 0.0),
                    "sources": agent_result.get("sources", []),
                    "from_agent": True,
                    "reflection_thoughts": agent_result.get("reflection_thoughts", [])
                }
        except Exception as e:
            logger.error(f"Agent-based retrieval failed: {e}")
        
        return {
            "success": False,
            "answer": "Unable to find answer in documents or using agents",
            "confidence": 0.0
        }

    async def stream(self, query: str, file_id: str | None, user_id: str | None, multimodal_context: dict | None = None, role: str | None = None, session_id: str | None = None, provider: str = "ollama", mode: str | None = None, cot: bool = False, cite: bool = True, decompose: bool = False, use_hyde: bool = False, rewrite: bool = False, refine: bool = False) -> AsyncGenerator[str, None]:
        """13-step RAG orchestrator streaming SSE-style tokens with enhanced chain integration.

        Yields SSE data frames as 'data: {"token": "..."}\n\n'
        """
        # Step 1: intent detection
        intent = 'answer'
        try:
            intent = detect_intent(query).intent
        except Exception:
            intent = 'answer'

        if intent == 'summarize':
            # redirect to summarization stream (not implemented here)
            yield f"data: {{\"token\": \"Starting summarization branch\"}}\n\n"
            return

        # Step 2: cache check
        cache_key = f"rag:{user_id}:{file_id}:{query}:{mode}"
        try:
            if cache_service:
                cached = cache_service.get_response(cache_key)
                if cached:
                    yield f"data: {{\"token\": \"{cached}\"}}\n\n"
                    return
        except Exception:
            logger.debug('Cache lookup failed')

        # Step 3: input guardrails
        ig = input_guardrail.validate_with_guardrails(query, role)
        if not ig.passed:
            yield f"data: {{\"token\": \"Input blocked: {ig.reason}\"}}\n\n"
            return

        # Step 4: transform query (query decomposition for comprehensive retrieval)
        search_queries = [query]
        try:
            search_queries = await decompose_query(query, provider=provider)
            logger.debug(f"Query decomposition: {search_queries}")
        except Exception as e:
            logger.warning(f"Query decomposition failed, falling back to expansion: {e}")
            try:
                search_queries = await expand_query(query)
            except Exception as e2:
                logger.warning(f"Query expansion also failed: {e2}")
                search_queries = [query]

        # Step 4.5: pre-retrieval query enhancement with multimodal context
        try:
            enhanced_query = await pre_retrieval(query, multimodal_context)
            if enhanced_query != query:
                logger.debug(f"Enhanced query with multimodal context")
                query = enhanced_query
        except Exception as e:
            logger.warning(f"Pre-retrieval enhancement failed: {e}")

        # Optional HYDE augmentation: generate a synthetic short summary to improve recall
        try:
            if use_hyde:
                hyde_q = await generate_hyde(query, provider=provider)
                if hyde_q and hyde_q != query:
                    search_queries.append(hyde_q)
                    logger.debug(f"HYDE query appended: {hyde_q}")
        except Exception as e:
            logger.warning(f"HYDE augmentation failed: {e}")

        # Step 5: retrieve using hybrid retriever with decomposed queries
        chunks = []
        all_chunks = []
        try:
            # Retrieve using each decomposed sub-question for comprehensive coverage
            for search_q in search_queries:
                try:
                    sub_chunks = await self.retriever.retrieve(search_q, document_id=file_id, top_k=12)
                    all_chunks.extend(sub_chunks)
                except Exception as e:
                    logger.debug(f"Sub-query retrieval failed for '{search_q}': {e}")
                    continue
            
            # Deduplicate by text content
            if all_chunks:
                seen_texts = set()
                deduped = []
                for c in all_chunks:
                    text_key = (c.get("text") or c.get("content") or "")[:200].lower()
                    if text_key and text_key not in seen_texts:
                        seen_texts.add(text_key)
                        deduped.append(c)
                    elif not text_key:
                        deduped.append(c)
                chunks = deduped
            else:
                # Fallback to simple retrieval
                chunks = await self.retriever.retrieve(query, document_id=file_id, top_k=12)
            
            logger.debug(f"Retrieved {len(chunks)} chunks from {len(search_queries)} decomposed queries")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")

        # Step 6: no chunks guard - fallback to agent-based retrieval
        if not chunks:
            logger.info(f"No chunks found for query, attempting agent-based retrieval")
            agent_result = await self._use_agent_retrieval(query, user_id, session_id, file_id, role)
            
            if agent_result.get("success"):
                # Agent found an answer - yield it
                answer = agent_result.get("answer", "")
                yield f"data: {{\"token\": \"{answer}\"}}\n\n"
                
                # Log reflection thoughts if available
                thoughts = agent_result.get("reflection_thoughts", [])
                if thoughts:
                    logger.info(f"Agent reflection: {thoughts}")
                return
            else:
                # No answer from agent either
                yield f"data: {{\"token\": \"No relevant content found in documents or knowledge base\"}}\n\n"
                return

        # Step 7: pre-augmentation (inject multimodal evidence with high confidence)
        try:
            chunks = await pre_augmentation(chunks, multimodal_context)
            logger.debug(f"Pre-augmented to {len(chunks)} chunks with multimodal context")
        except Exception as e:
            logger.warning(f"Pre-augmentation failed: {e}")

        # Step 8: apply context window (lost-in-middle fix)
        chunks = self._apply_context_window(chunks, max_chunks=6)

        # Step 9: post-augmentation with reranking
        try:
            reranker_obj = getattr(self.retriever, 'reranker', None)
            chunks = await post_augmentation_rerank(chunks, reranker_obj, query, top_k=6)
            logger.debug(f"Post-reranked to {len(chunks)} final chunks")
        except Exception as e:
            logger.warning(f"Post-augmentation/reranking failed: {e}")

        # Step 10: combine evidence with score prioritization for quality
        try:
            evidence = combine_evidence(chunks, max_chars=2000, deduplicate=True, prioritize_scores=True)
            logger.debug(f"Combined evidence with prioritization: {len(evidence)} chars")
        except Exception as e:
            logger.error(f"Evidence combining failed: {e}")
            evidence = self._format_evidence(chunks, cite)

        # Step 11: guard evidence length
        if len(evidence.strip()) < 30:
            yield f"data: {{\"token\": \"NOT FOUND IN DOCUMENT\"}}\n\n"
            return

        # Step 12: load history and build system prompt
        history_str = ''
        try:
            if memory_service:
                hist = await memory_service.load(user_id, session_id)
                history_str = memory_service.format_for_prompt(hist)
        except Exception:
            history_str = ''

        system_prompt = get_system_prompt(mode, cot) + "\n\n" + (history_str or '')
        
        # Step 13: use improved prompt template with stricter accuracy requirements
        PROMPT_TEMPLATE = """You are an expert AI assistant. Your task is to answer the user's question ONLY using the provided evidence. Accuracy and truthfulness are critical.

STRICT INSTRUCTIONS:
1. **Purpose**: Begin with a single-line "Purpose:" statement (<= 20 words) that captures the document's main focus or the answer's core idea.
2. **Key Points**: Provide 2–4 concise, specific bullet points (each <= 25 words). Each bullet must be directly supported by the evidence provided below.
3. **Evidence-Only**: Do NOT add any information, assumptions, or inferences beyond what appears in the evidence. If you are uncertain, say so.
4. **Citations**: When referencing specific facts or claims, include an inline citation like [Evidence 1] or [Chunk X] to show the source.
5. **Truthfulness**: If the evidence does not clearly answer the question, respond exactly: "I don't have enough information to answer that question accurately." NEVER guess or invent details.
6. **Format**: Use clear Markdown formatting (bold, bullets, numbered lists) for readability. Keep the full response under 150 words.
7. **Specificity**: Prefer concrete details (names, numbers, dates) over vague generalizations. Each claim must be verifiable from the evidence.

EVIDENCE:
{evidence}

QUESTION:
{question}

ANSWER:
"""
        full_prompt = system_prompt + "\n\n" + PROMPT_TEMPLATE.format(evidence=evidence, question=query)

        # Step 14: stream LLM response with improved provider routing
        try:
            async for tk in call_llm_stream(full_prompt, provider=provider):
                yield f"data: {{\"token\": \"{tk}\"}}\n\n"
        except Exception as e:
            logger.exception('LLM streaming failed: %s', e)
            yield f"data: {{\"token\": \"[ERROR] LLM failed: {str(e)[:50]}\"}}\n\n"
            return

        # Step 15: collect full answer using improved LLM routing
        full_answer = ''
        requested_provider = (provider or settings.DEFAULT_PROVIDER).lower()
        llm_response = None
        try:
            llm_response = await call_llm_with_metadata(full_prompt, provider=provider)
            full_answer = llm_response.text
            logger.debug(f"LLM answer generated: {len(full_answer)} chars")
            if llm_response.provider_used != requested_provider:
                logger.info(
                    "Requested LLM provider %s but routed to %s (fallback=%s)",
                    requested_provider,
                    llm_response.provider_used,
                    llm_response.fallback,
                )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            full_answer = ''

        # Step 15.5: verification — ask the LLM to validate claims against the evidence
        try:
            if full_answer:
                is_verified = await verify_answer(full_answer, evidence, provider=provider)
                if not is_verified:
                    logger.warning("Generated answer failed verification. Falling back to extractive summary.")
                    # If unsupported, replace with extractive evidence summary
                    fallback_extractive = combine_evidence(chunks, max_chars=500, deduplicate=True, prioritize_scores=True)
                    full_answer = fallback_extractive or "I don't have enough information to answer that question accurately."
        except Exception as e:
            logger.debug(f"Verification step failed: {e}")

        # Step 16: output guardrails
        try:
            og = output_guardrail.validate(full_answer, chunks)
            out_text = og.output
        except Exception:
            out_text = full_answer or ''

        # Step 17: cache & memory persist
        try:
            if cache_service:
                cache_service.set_response(cache_key, out_text)
        except Exception:
            logger.debug('Cache set failed')
        try:
            if memory_service and out_text:
                await memory_service.append(user_id, session_id, 'assistant', out_text)
        except Exception:
            logger.debug('Memory append failed')

        try:
            if persist_turn:
                # fire-and-forget
                persist_turn.delay(user_id, session_id, query, out_text)
        except Exception:
            logger.debug('persist_turn failed')

        # final SSE sentinel for clients to know stream is complete
        yield f"data: {{\"token\": \"[DONE]\"}}\n\n"


rag_pipeline = RagPipeline()

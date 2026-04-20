"""Skills for RAG Agent."""

import logging
from typing import Any, Dict, List, Optional
from app.agents.base_agent import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class RetrievalSkill(BaseSkill):
    """Document retrieval skill."""
    
    def __init__(self, retriever=None):
        super().__init__(
            name="retrieve",
            description="Retrieve relevant documents from knowledge base",
            version="1.0.0"
        )
        self.retriever = retriever
    
    async def validate_input(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate input parameters."""
        if "query" not in kwargs:
            return False, "Missing required parameter: query"
        if "top_k" not in kwargs:
            return False, "Missing required parameter: top_k"
        return True, None
    
    async def execute(self, **kwargs) -> SkillResult:
        """Execute retrieval."""
        try:
            query = kwargs.get("query")
            top_k = kwargs.get("top_k", 6)
            document_id = kwargs.get("document_id")
            strategy = kwargs.get("strategy", "standard")
            
            if not self.retriever:
                return SkillResult(
                    success=False,
                    error="Retriever not configured"
                )
            
            # Call retriever
            chunks = await self.retriever.retrieve(
                query=query,
                document_id=document_id,
                top_k=top_k
            )
            
            return SkillResult(
                success=True,
                data={
                    "chunks": chunks,
                    "total_count": len(chunks),
                    "strategy": strategy
                },
                metadata={
                    "top_k": top_k,
                    "document_id": document_id,
                    "strategy": strategy
                }
            )
        
        except Exception as e:
            logger.exception(f"Retrieval skill failed: {e}")
            return SkillResult(success=False, error=str(e))


class AugmentationSkill(BaseSkill):
    """Context augmentation skill."""
    
    def __init__(self):
        super().__init__(
            name="augment",
            description="Augment context from retrieved chunks",
            version="1.0.0"
        )
    
    async def validate_input(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate input parameters."""
        if "query" not in kwargs:
            return False, "Missing required parameter: query"
        if "chunks" not in kwargs:
            return False, "Missing required parameter: chunks"
        return True, None
    
    async def execute(self, **kwargs) -> SkillResult:
        """Execute augmentation."""
        try:
            query = kwargs.get("query")
            chunks = kwargs.get("chunks", [])
            rerank = kwargs.get("rerank", True)
            
            if not chunks:
                return SkillResult(
                    success=True,
                    data={
                        "context": "",
                        "reranked_chunks": [],
                        "augmentation_score": 0.0
                    }
                )
            
            # Combine chunks into context
            context_parts = []
            for i, chunk in enumerate(chunks[:10], 1):  # Limit to top 10
                text = chunk.get("text") or chunk.get("content", "")
                source = chunk.get("metadata", {}).get("source", "Unknown")
                context_parts.append(f"[{i}] {text}\n(Source: {source})")
            
            context = "\n\n".join(context_parts)
            
            # Calculate augmentation score (simple: based on chunks quality)
            score = min(1.0, len(chunks) / 6.0)  # Normalize to 6 chunks
            
            return SkillResult(
                success=True,
                data={
                    "context": context,
                    "reranked_chunks": chunks,
                    "augmentation_score": score,
                    "num_chunks": len(chunks)
                },
                metadata={
                    "rerank": rerank,
                    "chunk_count": len(chunks)
                }
            )
        
        except Exception as e:
            logger.exception(f"Augmentation skill failed: {e}")
            return SkillResult(success=False, error=str(e))


class SynthesisSkill(BaseSkill):
    """LLM-based synthesis skill."""
    
    def __init__(self, llm_call_fn=None):
        super().__init__(
            name="synthesize",
            description="Generate answer using LLM with augmented context",
            version="1.0.0"
        )
        self.llm_call_fn = llm_call_fn
    
    async def validate_input(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate input parameters."""
        if "query" not in kwargs:
            return False, "Missing required parameter: query"
        if "context" not in kwargs:
            return False, "Missing required parameter: context"
        return True, None
    
    async def execute(self, **kwargs) -> SkillResult:
        """Execute synthesis."""
        try:
            query = kwargs.get("query")
            context = kwargs.get("context", "")
            user_role = kwargs.get("user_role", "general")
            
            # Build prompt
            system_prompt = f"""You are a helpful AI assistant. 
Your role is to answer questions based on the provided context.
User role: {user_role}

Guidelines:
1. Answer based on the provided context
2. If context doesn't contain the answer, say "I cannot find this information in the provided context"
3. Be concise and clear
4. Cite sources when possible
5. Maintain professional tone"""
            
            user_prompt = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
            
            # Call LLM (using mock if not configured)
            if self.llm_call_fn:
                try:
                    answer = await self.llm_call_fn(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                    )
                except Exception as e:
                    logger.warning(f"LLM call failed, using fallback: {e}")
                    answer = self._generate_fallback_answer(query, context)
            else:
                answer = self._generate_fallback_answer(query, context)
            
            # Calculate confidence based on context coverage
            confidence = min(1.0, 0.7 if context else 0.3)  # Base confidence
            
            return SkillResult(
                success=True,
                data={
                    "answer": answer,
                    "confidence": confidence,
                    "user_role": user_role,
                    "has_context": bool(context)
                },
                metadata={
                    "system_prompt": system_prompt,
                    "user_role": user_role
                }
            )
        
        except Exception as e:
            logger.exception(f"Synthesis skill failed: {e}")
            return SkillResult(success=False, error=str(e))
    
    def _generate_fallback_answer(self, query: str, context: str) -> str:
        """Generate a simple answer without LLM."""
        if not context:
            return f"I cannot answer '{query}' because no relevant context was provided."
        
        # Simple extraction: return first relevant chunk
        lines = context.split("\n\n")
        if lines:
            return f"Based on the available information: {lines[0][:500]}..."
        return f"I found information about '{query}' in the knowledge base."

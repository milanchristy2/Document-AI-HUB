"""RAG Agent with reflection agentic design pattern and integrated guardrails."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent, BaseSkill, AgentConfig, AgentInput, AgentOutput, AgentStatus, AgentType, ExecutionContext, SkillResult
from app.agents.tools import ToolsManager, ToolResult
from app.ai.guardrails import guardrails_manager

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with TTL."""
    
    def __init__(self, key: str, value: Any, ttl_seconds: int = 3600):
        self.key = key
        self.value = value
        self.created_at = datetime.utcnow()
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.utcnow() - self.created_at > timedelta(seconds=self.ttl_seconds)


class QueryCache:
    """Query result caching with TTL."""
    
    def __init__(self, max_size: int = 50):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self._logger = logging.getLogger(f"{__name__}.QueryCache")
    
    def get(self, query: str) -> Optional[Any]:
        """Get cached result if not expired."""
        if query not in self.cache:
            return None
        
        entry = self.cache[query]
        if entry.is_expired():
            del self.cache[query]
            return None
        
        self._logger.debug(f"Cache hit for query: {query[:50]}...")
        return entry.value
    
    def set(self, query: str, result: Any, ttl_seconds: int = 3600) -> None:
        """Cache a query result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
            del self.cache[oldest_key]
        
        self.cache[query] = CacheEntry(query, result, ttl_seconds)
        self._logger.debug(f"Cached result for query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()


class ReflectionMemory:
    """Track agent thoughts and reflections."""
    
    def __init__(self):
        self.thoughts: List[str] = []
        self.decisions: List[Dict[str, Any]] = []
        self.reflections: List[str] = []
    
    def add_thought(self, thought: str) -> None:
        self.thoughts.append(thought)
        logger.debug(f"Thought: {thought}")
    
    def add_decision(self, decision: Dict[str, Any]) -> None:
        self.decisions.append(decision)
        logger.debug(f"Decision: {decision}")
    
    def add_reflection(self, reflection: str) -> None:
        self.reflections.append(reflection)
        logger.debug(f"Reflection: {reflection}")
    
    def clear(self) -> None:
        self.thoughts.clear()
        self.decisions.clear()
        self.reflections.clear()


class RAGAgent(BaseAgent):
    """RAG Agent with reflection agentic pattern.
    
    Implements: Observe -> Reflect -> Decide -> Act pattern
    """
    
    def __init__(self, config: AgentConfig, retriever=None):
        super().__init__(config)
        self.reflection_memory = ReflectionMemory()
        self.query_cache = QueryCache(max_size=50)
        self.tools_manager = ToolsManager(retriever=retriever)
        self.retriever = retriever
        self._logger.info(f"Initialized RAG Agent with reflection pattern, caching, and tools")
    
    async def _execute_impl(self, agent_input: AgentInput, context: ExecutionContext) -> Any:
        """Execute RAG agent with reflection pattern and guardrails."""
        self.reflection_memory.clear()
        
        # Check cache first
        cached_result = self.query_cache.get(agent_input.query)
        if cached_result:
            self.reflection_memory.add_thought("Using cached result for identical query")
            return {**cached_result, "from_cache": True}
        
        # GUARDRAIL: Check input validity
        input_check = await guardrails_manager.check_input(agent_input.query, agent_input.user_id)
        if not input_check.get("passed"):
            self.reflection_memory.add_reflection(f"Input blocked: {input_check.get('reason')}")
            return {
                "success": False,
                "error": input_check.get("reason"),
                "reflection_thoughts": self.reflection_memory.thoughts
            }
        
        # Step 1: OBSERVE - Analyze the query
        await self._observe(agent_input)
        
        # Step 2: REFLECT - Evaluate retrieval strategy
        strategy = await self._reflect(agent_input)
        
        # Step 3: DECIDE - Choose skills to execute
        plan = await self._decide(agent_input, strategy)
        
        # Step 4: ACT - Execute skills
        result = await self._act(agent_input, plan, context)
        
        # GUARDRAIL: Check output validity
        if result.get("answer"):
            output_check = await guardrails_manager.check_output(
                result.get("answer", ""),
                result.get("confidence", 0.0),
                len(result.get("sources", [])) > 0
            )
            if not output_check.get("passed"):
                self.reflection_memory.add_reflection(f"Output warning: {output_check.get('reason')}")
                result["output_warning"] = output_check.get("reason")
        
        # Step 5: FINAL REFLECTION - Evaluate result quality
        await self._final_reflection(agent_input, result)
        
        return result
    
    async def _observe(self, agent_input: AgentInput) -> None:
        """Observe: Analyze query and context."""
        thought = f"Observing query: '{agent_input.query}' from user {agent_input.user_id}"
        self.reflection_memory.add_thought(thought)
        
        # Analyze query characteristics
        query_len = len(agent_input.query.split())
        context_provided = bool(agent_input.context)
        
        observation = {
            "query_length": query_len,
            "context_available": context_provided,
            "has_document_id": agent_input.document_id is not None,
            "parameters": agent_input.parameters
        }
        self.reflection_memory.add_decision(observation)
    
    async def _reflect(self, agent_input: AgentInput) -> str:
        """Reflect: Evaluate retrieval strategy."""
        reflection_text = "Evaluating retrieval strategy based on query characteristics..."
        
        query = agent_input.query.lower()
        
        # Determine strategy based on query
        if any(word in query for word in ["summarize", "summary", "overview"]):
            strategy = "summary"
            self.reflection_memory.add_reflection("Using summary strategy for query")
        elif any(word in query for word in ["extract", "pull", "get all", "list"]):
            strategy = "extraction"
            self.reflection_memory.add_reflection("Using extraction strategy for query")
        elif any(word in query for word in ["compare", "difference", "versus"]):
            strategy = "comparison"
            self.reflection_memory.add_reflection("Using comparison strategy for query")
        else:
            strategy = "standard"
            self.reflection_memory.add_reflection("Using standard retrieval strategy")
        
        return strategy
    
    async def _decide(self, agent_input: AgentInput, strategy: str) -> Dict[str, Any]:
        """Decide: Plan which skills to execute."""
        plan = {
            "strategy": strategy,
            "skills": ["retrieve", "augment", "synthesize"],
            "rerank": True,
            "top_k": agent_input.parameters.get("top_k", 6),
        }
        
        self.reflection_memory.add_decision(plan)
        self._logger.info(f"Plan: {plan}")
        
        return plan
    
    async def _act(self, agent_input: AgentInput, plan: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Act: Execute skills in sequence with tool fallback."""
        results = {}
        chunks = []
        
        # Execute retrieve skill
        retrieval_result = await self.call_skill(
            "retrieve",
            query=agent_input.query,
            top_k=plan["top_k"],
            document_id=agent_input.document_id,
            strategy=plan["strategy"]
        )
        results["retrieval"] = retrieval_result
        
        # If retrieval fails or no results, try tools
        if not retrieval_result.success or not retrieval_result.data:
            self._logger.info("Retrieval failed or no results, attempting tool-based search")
            tool_result = await self.tools_manager.execute_tool(
                "web_search",
                query=agent_input.query
            )
            
            if tool_result.success and tool_result.data:
                self.reflection_memory.add_thought(f"Used web search tool as fallback for: {agent_input.query}")
                # Convert tool results to chunk format
                if "results" in tool_result.data:
                    for i, result in enumerate(tool_result.data["results"]):
                        chunks.append({
                            "text": result.get("snippet", ""),
                            "metadata": {
                                "source": result.get("url", "unknown"),
                                "title": result.get("title", "")
                            },
                            "score": 0.7 - (i * 0.1)
                        })
        else:
            chunks = retrieval_result.data.get("chunks", []) if retrieval_result.data else []
        
        # Execute augment skill
        augment_result = await self.call_skill(
            "augment",
            query=agent_input.query,
            chunks=chunks,
            rerank=plan["rerank"]
        )
        results["augmentation"] = augment_result
        
        if not augment_result.success:
            self._logger.error(f"Augmentation failed: {augment_result.error}")
            return {"success": False, "error": augment_result.error}
        
        # Execute synthesize skill
        context_data = augment_result.data if augment_result.data else {}
        context_text = context_data.get("context", "")
        synthesize_result = await self.call_skill(
            "synthesize",
            query=agent_input.query,
            context=context_text,
            user_role=agent_input.parameters.get("user_role")
        )
        results["synthesis"] = synthesize_result
        
        if not synthesize_result.success:
            self._logger.error(f"Synthesis failed: {synthesize_result.error}")
            return {"success": False, "error": synthesize_result.error}
        
        # Try fact-checking if confidence is low
        synth_data = synthesize_result.data if synthesize_result.data else {}
        answer = synth_data.get("answer", "")
        confidence = synth_data.get("confidence", 0.0)
        
        if confidence < 0.6 and chunks:
            fact_check_result = await self.tools_manager.execute_tool(
                "fact_check",
                claim=answer
            )
            if fact_check_result.success and fact_check_result.data:
                fact_data = fact_check_result.data
                confidence = max(confidence, fact_data.get("confidence", 0.0))
                self.reflection_memory.add_thought("Applied fact-checking to increase confidence")
        
        # Compile final result
        tools_used = []
        for k, v in results.items():
            if isinstance(v, dict):
                if v.get("success"):
                    tools_used.append(k)
            elif hasattr(v, 'success') and v.success:
                tools_used.append(k)
        
        final_result = {
            "success": True,
            "answer": answer,
            "confidence": confidence,
            "sources": chunks[:3],  # Top 3 sources
            "execution_steps": list(results.keys()),
            "reflection_thoughts": self.reflection_memory.thoughts,
            "tools_used": tools_used,
        }
        
        # Cache successful results
        if final_result.get("success") and final_result.get("confidence", 0) > 0.5:
            self.query_cache.set(agent_input.query, final_result, ttl_seconds=3600)
        
        return final_result
    
    async def _final_reflection(self, agent_input: AgentInput, result: Dict[str, Any]) -> None:
        """Final reflection: Evaluate result quality."""
        if result.get("success"):
            confidence = result.get("confidence", 0)
            reflection = f"Successfully generated answer with {confidence:.2%} confidence"
            if confidence > 0.8:
                reflection += " - High quality result"
            elif confidence > 0.6:
                reflection += " - Moderate quality result"
            else:
                reflection += " - Low confidence result, may need verification"
        else:
            reflection = f"Failed to generate answer: {result.get('error', 'Unknown error')}"
        
        self.reflection_memory.add_reflection(reflection)
        self._logger.info(reflection)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_queries": len(self.query_cache.cache),
            "max_cache_size": self.query_cache.max_size,
            "cache_hit_enabled": True
        }
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self.query_cache.clear()
        self._logger.info("Query cache cleared")

"""Router Agent for query routing and agent selection with guardrails."""

import logging
from typing import Any, Dict, Optional
from app.agents.base_agent import BaseAgent, AgentConfig, AgentInput, AgentOutput, AgentStatus, AgentType, ExecutionContext
from app.ai.guardrails import guardrails_manager

logger = logging.getLogger(__name__)


class RouterAgent(BaseAgent):
    """Routes queries to appropriate agents based on intent and context with guardrails.
    
    Features:
    - Intent-based routing
    - Rate limiting checks
    - Context-aware agent selection
    - Fallback handling
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.rag_agent = None  # Injected dependency
        self.routing_history = []  # Track routing decisions
        self.max_history = 100
        self._logger.info("Initialized Router Agent with guardrails")
    
    def set_rag_agent(self, rag_agent: BaseAgent) -> None:
        """Set the RAG agent for delegation."""
        self.rag_agent = rag_agent
    
    async def _execute_impl(self, agent_input: AgentInput, context: ExecutionContext) -> Any:
        """Route query to appropriate agent with guardrails."""
        
        # GUARDRAIL: Check rate limit
        rate_limit_check = await guardrails_manager.check_rate_limit(agent_input.user_id)
        if not rate_limit_check.get("passed"):
            return {
                "success": False,
                "error": rate_limit_check.get("reason"),
                "type": "rate_limit_exceeded"
            }
        
        # Determine intent
        intent = self._determine_intent(agent_input.query)
        self._logger.info(f"Detected intent: {intent}")
        
        # Route based on intent
        if intent == "rag_query":
            if not self.rag_agent:
                return {"success": False, "error": "RAG agent not configured"}
            
            result = await self.rag_agent.execute(agent_input, context)
            return {
                "success": result.status == AgentStatus.COMPLETED,
                "agent_used": "rag_agent",
                "answer": result.result.get("answer") if result.result else None,
                "confidence": result.result.get("confidence", 0) if result.result else 0,
                "sources": result.result.get("sources", []) if result.result else [],
                "tools_used": result.result.get("tools_used", []) if result.result else [],
                "execution_steps": result.result.get("execution_steps", []) if result.result else [],
                "reflection_thoughts": result.result.get("reflection_thoughts", []) if result.result else [],
            }
        
        elif intent == "clarification":
            return {
                "success": False,
                "error": "Clarification needed",
                "type": "clarification_request",
                "message": "Could you provide more details about your query?"
            }
        
        elif intent == "unsupported":
            return {
                "success": False,
                "error": "Query type not supported",
                "type": "unsupported_query",
            }
        
        else:  # default to RAG
            if not self.rag_agent:
                return {"success": False, "error": "RAG agent not configured"}
            result = await self.rag_agent.execute(agent_input, context)
            return {
                "success": result.status == AgentStatus.COMPLETED,
                "agent_used": "rag_agent",
                "answer": result.result.get("answer") if result.result else None,
                "confidence": result.result.get("confidence", 0) if result.result else 0,
                "sources": result.result.get("sources", []) if result.result else [],
                "tools_used": result.result.get("tools_used", []) if result.result else [],
                "execution_steps": result.result.get("execution_steps", []) if result.result else [],
                "reflection_thoughts": result.result.get("reflection_thoughts", []) if result.result else [],
            }
    
    def _determine_intent(self, query: str) -> str:
        """Determine query intent using simple heuristics."""
        query_lower = query.lower().strip()
        
        # Check for RAG-appropriate queries
        if any(word in query_lower for word in [
            "what", "how", "why", "explain", "describe",
            "find", "search", "tell me", "show me", "list",
            "summarize", "extract", "compare"
        ]):
            return "rag_query"
        
        # Check for clarification needed
        if len(query_lower) < 5 or query_lower.count("?") > 2:
            return "clarification"
        
        # Check for unsupported
        if any(word in query_lower for word in [
            "image", "video", "audio", "draw", "create image"
        ]):
            return "unsupported"
        
        return "rag_query"  # Default
    
    def _record_routing_decision(self, intent: str, user_id: str, query: str) -> None:
        """Record routing decision for analytics."""
        decision = {
            "intent": intent,
            "user_id": user_id,
            "query": query[:100],  # First 100 chars
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }
        self.routing_history.append(decision)
        
        # Keep history bounded
        if len(self.routing_history) > self.max_history:
            self.routing_history.pop(0)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        intent_counts = {}
        for decision in self.routing_history:
            intent = decision["intent"]
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "total_routes": len(self.routing_history),
            "intent_distribution": intent_counts,
            "max_history": self.max_history
        }

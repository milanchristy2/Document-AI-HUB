"""Agent tools system for RAG agent."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of tool execution."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTool(ABC):
    """Base class for all agent tools."""
    
    def __init__(self, name: str, description: str, required_params: List[str] = None):
        self.name = name
        self.description = description
        self.required_params = required_params or []
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "required_params": self.required_params
        }


class WebSearchTool(BaseTool):
    """Search the web for information."""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information using search queries",
            required_params=["query"]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute web search."""
        try:
            query = kwargs.get("query")
            if not query:
                return ToolResult(success=False, error="Missing required parameter: query")
            
            # Try to use external search API
            try:
                import httpx
                # Using a simple search approach (you can replace with actual API)
                async with httpx.AsyncClient(timeout=10) as client:
                    # This is a mock - replace with actual search API
                    logger.debug(f"Web search for: {query}")
                    # Simulating search results
                    results = [
                        {
                            "title": f"Result for '{query}'",
                            "url": f"https://example.com/search?q={query}",
                            "snippet": f"Information about {query}"
                        }
                    ]
                    return ToolResult(
                        success=True,
                        data={"results": results, "query": query},
                        metadata={"num_results": len(results)}
                    )
            except ImportError:
                logger.warning("httpx not available, using mock search")
                return ToolResult(
                    success=True,
                    data={
                        "results": [
                            {
                                "title": f"Information about {query}",
                                "url": "https://example.com",
                                "snippet": f"This would contain search results for: {query}"
                            }
                        ],
                        "query": query
                    },
                    metadata={"mock": True}
                )
        
        except Exception as e:
            logger.exception(f"Web search failed: {e}")
            return ToolResult(success=False, error=str(e))


class CalculatorTool(BaseTool):
    """Perform mathematical calculations."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations and computations",
            required_params=["expression"]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute calculator."""
        try:
            expression = kwargs.get("expression")
            if not expression:
                return ToolResult(success=False, error="Missing required parameter: expression")
            
            # Evaluate expression safely
            try:
                result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
                return ToolResult(
                    success=True,
                    data={"expression": expression, "result": result},
                    metadata={"calculation": expression}
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Calculation failed: {str(e)}"
                )
        
        except Exception as e:
            logger.exception(f"Calculator tool failed: {e}")
            return ToolResult(success=False, error=str(e))


class ContextLookupTool(BaseTool):
    """Look up additional context from knowledge base."""
    
    def __init__(self, retriever=None):
        super().__init__(
            name="context_lookup",
            description="Look up additional context or related information",
            required_params=["topic"]
        )
        self.retriever = retriever
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute context lookup."""
        try:
            topic = kwargs.get("topic")
            if not topic:
                return ToolResult(success=False, error="Missing required parameter: topic")
            
            if not self.retriever:
                return ToolResult(
                    success=False,
                    error="Retriever not configured"
                )
            
            # Retrieve additional context
            chunks = await self.retriever.retrieve(query=topic, top_k=3)
            
            return ToolResult(
                success=True,
                data={"topic": topic, "context": chunks},
                metadata={"num_chunks": len(chunks)}
            )
        
        except Exception as e:
            logger.exception(f"Context lookup failed: {e}")
            return ToolResult(success=False, error=str(e))


class FactCheckTool(BaseTool):
    """Fact-check claims against retrieved documents."""
    
    def __init__(self, retriever=None):
        super().__init__(
            name="fact_check",
            description="Verify facts and claims against available documents",
            required_params=["claim"]
        )
        self.retriever = retriever
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute fact checking."""
        try:
            claim = kwargs.get("claim")
            if not claim:
                return ToolResult(success=False, error="Missing required parameter: claim")
            
            if not self.retriever:
                return ToolResult(success=False, error="Retriever not configured")
            
            # Search for supporting evidence
            chunks = await self.retriever.retrieve(query=claim, top_k=5)
            
            # Simple fact-checking: check if claim keywords appear in documents
            claim_words = set(claim.lower().split())
            evidence_found = []
            
            for chunk in chunks:
                text = (chunk.get("text") or chunk.get("content", "")).lower()
                matching_words = claim_words.intersection(set(text.split()))
                if len(matching_words) > 0:
                    evidence_found.append({
                        "match_score": len(matching_words) / len(claim_words),
                        "text": text[:200]
                    })
            
            is_verified = len(evidence_found) > 0
            confidence = sum(e["match_score"] for e in evidence_found) / len(evidence_found) if evidence_found else 0.0
            
            return ToolResult(
                success=True,
                data={
                    "claim": claim,
                    "verified": is_verified,
                    "confidence": confidence,
                    "evidence": evidence_found
                },
                metadata={"evidence_count": len(evidence_found)}
            )
        
        except Exception as e:
            logger.exception(f"Fact checking failed: {e}")
            return ToolResult(success=False, error=str(e))


class SummarizationTool(BaseTool):
    """Summarize text content."""
    
    def __init__(self):
        super().__init__(
            name="summarize",
            description="Summarize text content into key points",
            required_params=["text"]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute summarization."""
        try:
            text = kwargs.get("text")
            if not text:
                return ToolResult(success=False, error="Missing required parameter: text")
            
            # Simple summarization: extract key sentences
            sentences = text.split(".")
            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
            summary = ". ".join(key_sentences) + "."
            
            return ToolResult(
                success=True,
                data={
                    "original_length": len(text),
                    "summary": summary,
                    "compression_ratio": len(summary) / len(text) if text else 0
                },
                metadata={"num_sentences": len(key_sentences)}
            )
        
        except Exception as e:
            logger.exception(f"Summarization failed: {e}")
            return ToolResult(success=False, error=str(e))


class ToolsManager:
    """Manages and orchestrates tool execution."""
    
    def __init__(self, retriever=None):
        self._logger = logging.getLogger(f"{__name__}.ToolsManager")
        self.tools: Dict[str, BaseTool] = {}
        
        # Register default tools
        self.register_tool(WebSearchTool())
        self.register_tool(CalculatorTool())
        self.register_tool(ContextLookupTool(retriever))
        self.register_tool(FactCheckTool(retriever))
        self.register_tool(SummarizationTool())
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        self._logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [tool.get_schema() for tool in self.tools.values()]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool."""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found"
            )
        
        # Validate required parameters
        for param in tool.required_params:
            if param not in kwargs:
                return ToolResult(
                    success=False,
                    error=f"Missing required parameter: {param}"
                )
        
        try:
            self._logger.debug(f"Executing tool: {tool_name} with params: {list(kwargs.keys())}")
            result = await tool.execute(**kwargs)
            return result
        except Exception as e:
            self._logger.exception(f"Tool execution failed: {e}")
            return ToolResult(success=False, error=str(e))

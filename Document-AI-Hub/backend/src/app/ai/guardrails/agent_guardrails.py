"""Guardrails integration for agents using langchain and guardrails."""

import logging
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    reason: Optional[str] = None
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class InputGuardrail:
    """Input validation guardrail using patterns and rules."""
    
    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.InputGuardrail")
        self.max_query_length = 2000
        self.min_query_length = 2
        
    async def validate(self, query: str, user_id: Optional[str] = None) -> GuardrailResult:
        """Validate input query."""
        try:
            # Check length
            if len(query) < self.min_query_length:
                return GuardrailResult(
                    passed=False,
                    reason="Query too short",
                    score=0.0
                )
            
            if len(query) > self.max_query_length:
                return GuardrailResult(
                    passed=False,
                    reason=f"Query exceeds maximum length of {self.max_query_length} characters",
                    score=0.0
                )
            
            # Check for malicious patterns
            dangerous_patterns = ["<script", "javascript:", "onclick", "onerror", "eval("]
            for pattern in dangerous_patterns:
                if pattern.lower() in query.lower():
                    return GuardrailResult(
                        passed=False,
                        reason="Query contains potentially malicious content",
                        score=0.0
                    )
            
            # Check for SQL injection patterns
            sql_patterns = ["union", "select", "insert", "delete", "drop", ";--"]
            sql_count = sum(1 for pattern in sql_patterns if pattern.lower() in query.lower())
            if sql_count > 1:
                return GuardrailResult(
                    passed=False,
                    reason="Query contains suspicious patterns",
                    score=0.2
                )
            
            self._logger.debug(f"Input validation passed for user {user_id}")
            return GuardrailResult(
                passed=True,
                reason="Input valid",
                score=1.0
            )
        
        except Exception as e:
            self._logger.error(f"Input validation error: {e}")
            return GuardrailResult(
                passed=False,
                reason=f"Validation error: {str(e)}",
                score=0.0
            )


class OutputGuardrail:
    """Output validation guardrail using confidence and evidence checking."""
    
    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.OutputGuardrail")
        self.min_confidence = 0.3
        self.hallucination_threshold = 0.5
        
    async def validate(self, answer: str, confidence: float, has_evidence: bool = True) -> GuardrailResult:
        """Validate generated answer."""
        try:
            # Check confidence
            if confidence < self.min_confidence:
                return GuardrailResult(
                    passed=False,
                    reason=f"Low confidence score: {confidence:.2%}",
                    score=confidence
                )
            
            # Check for hallucination patterns
            hallucination_markers = [
                "i don't have access",
                "i'm not sure",
                "i cannot find",
                "i was trained",
                "as far as i know",
                "i think",
                "i believe"
            ]
            
            answer_lower = answer.lower()
            hallucination_count = sum(1 for marker in hallucination_markers if marker in answer_lower)
            
            if hallucination_count > 2:
                return GuardrailResult(
                    passed=False,
                    reason="Output contains uncertainty markers",
                    score=0.5
                )
            
            # Check if has evidence
            if not has_evidence and confidence < 0.7:
                return GuardrailResult(
                    passed=False,
                    reason="Low confidence output without supporting evidence",
                    score=confidence
                )
            
            # Check length (too short might be incomplete)
            if len(answer) < 20:
                return GuardrailResult(
                    passed=False,
                    reason="Output too short or incomplete",
                    score=0.4
                )
            
            self._logger.debug("Output validation passed")
            return GuardrailResult(
                passed=True,
                reason="Output valid",
                score=confidence
            )
        
        except Exception as e:
            self._logger.error(f"Output validation error: {e}")
            return GuardrailResult(
                passed=False,
                reason=f"Validation error: {str(e)}",
                score=0.0
            )


class RateLimitGuardrail:
    """Rate limiting guardrail."""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self._logger = logging.getLogger(f"{__name__}.RateLimitGuardrail")
        self.max_requests_per_minute = max_requests_per_minute
        self.user_requests: Dict[str, list] = {}
        
    async def validate(self, user_id: str) -> GuardrailResult:
        """Check rate limit for user."""
        try:
            import time
            current_time = time.time()
            minute_ago = current_time - 60
            
            if user_id not in self.user_requests:
                self.user_requests[user_id] = []
            
            # Clean up old requests
            self.user_requests[user_id] = [
                req_time for req_time in self.user_requests[user_id]
                if req_time > minute_ago
            ]
            
            # Check limit
            if len(self.user_requests[user_id]) >= self.max_requests_per_minute:
                return GuardrailResult(
                    passed=False,
                    reason=f"Rate limit exceeded: {self.max_requests_per_minute} requests per minute",
                    score=0.0
                )
            
            # Add current request
            self.user_requests[user_id].append(current_time)
            
            return GuardrailResult(
                passed=True,
                reason="Rate limit OK",
                score=1.0,
                metadata={
                    "requests_used": len(self.user_requests[user_id]),
                    "limit": self.max_requests_per_minute
                }
            )
        
        except Exception as e:
            self._logger.error(f"Rate limit check error: {e}")
            return GuardrailResult(
                passed=False,
                reason=f"Validation error: {str(e)}",
                score=0.0
            )


class GuardrailsManager:
    """Manager for all guardrails."""
    
    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.GuardrailsManager")
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()
        self.rate_limit_guardrail = RateLimitGuardrail()
    
    async def check_input(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Check input guardrails."""
        result = await self.input_guardrail.validate(query, user_id)
        return {
            "passed": result.passed,
            "reason": result.reason,
            "score": result.score,
            "check_type": "input"
        }
    
    async def check_output(self, answer: str, confidence: float, has_evidence: bool = True) -> Dict[str, Any]:
        """Check output guardrails."""
        result = await self.output_guardrail.validate(answer, confidence, has_evidence)
        return {
            "passed": result.passed,
            "reason": result.reason,
            "score": result.score,
            "check_type": "output"
        }
    
    async def check_rate_limit(self, user_id: str) -> Dict[str, Any]:
        """Check rate limit guardrails."""
        result = await self.rate_limit_guardrail.validate(user_id)
        return {
            "passed": result.passed,
            "reason": result.reason,
            "score": result.score,
            "check_type": "rate_limit",
            "metadata": result.metadata
        }


# Global guardrails manager
guardrails_manager = GuardrailsManager()

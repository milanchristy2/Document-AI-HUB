"""Base Agent and Skill classes for the agentic RAG system."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
import time
from uuid import uuid4

logger = logging.getLogger(__name__)


# ==================== ENUMS ====================

class AgentType(str, Enum):
    """Agent types"""
    SEARCH = "search"
    SUMMARIZE = "summarize"
    EXTRACT = "extract"
    SYNTHESIS = "synthesis"
    ROUTE = "route"


class AgentStatus(str, Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ==================== DATA CLASSES ====================

@dataclass
class AgentConfig:
    """Agent configuration"""
    name: str
    agent_type: AgentType
    description: str
    version: str = "1.0.0"
    max_retries: int = 3
    timeout_seconds: int = 30
    enabled: bool = True
    priority: int = 0
    require_guardrails: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInput:
    """Agent input"""
    query: str
    user_id: str
    session_id: str
    document_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """Skill execution result"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Agent output"""
    agent_id: str
    agent_name: str
    status: AgentStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    skills_used: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "skills_used": self.skills_used,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class ExecutionContext:
    """Execution context for tracking agent chains"""
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_chain: List[str] = field(default_factory=list)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    execution_depth: int = 0
    max_depth: int = 5

    def can_execute(self) -> bool:
        """Check execution depth limit"""
        return self.execution_depth < self.max_depth

    def add_agent(self, name: str) -> None:
        """Add agent to chain"""
        self.agent_chain.append(name)
        self.execution_depth += 1


# ==================== BASE CLASSES ====================

class BaseSkill(ABC):
    """Base class for agent skills"""

    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.version = version
        self.skill_id = str(uuid4())
        self._stats = {"count": 0, "success": 0, "total_time": 0.0}
        self._logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def execute(self, **kwargs) -> SkillResult:
        """Execute the skill"""
        pass

    async def validate_input(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate input - override in subclasses"""
        return True, None

    async def pre_execute(self, **kwargs) -> None:
        """Pre-execution hook"""
        pass

    async def post_execute(self, result: SkillResult) -> SkillResult:
        """Post-execution hook"""
        return result

    def get_info(self) -> Dict[str, Any]:
        """Get skill statistics"""
        stats = self._stats
        return {
            "name": self.name,
            "version": self.version,
            "executions": stats["count"],
            "success_rate": (stats["success"] / stats["count"] * 100) if stats["count"] > 0 else 0,
            "avg_time_ms": stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        }

    def _update_stats(self, success: bool, elapsed_ms: float) -> None:
        """Update execution statistics"""
        self._stats["count"] += 1
        if success:
            self._stats["success"] += 1
        self._stats["total_time"] += elapsed_ms


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = str(uuid4())
        self.status = AgentStatus.IDLE
        self._skills: Dict[str, BaseSkill] = {}
        self._history: List[AgentOutput] = []
        self._logger = logging.getLogger(f"{__name__}.{config.name}")
        self._logger.info(f"Initialized: {config.name} ({config.agent_type.value})")

    # ==================== SKILL MANAGEMENT ====================

    def register_skill(self, skill: BaseSkill) -> None:
        """Register a skill"""
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' already registered")
        self._skills[skill.name] = skill
        self._logger.debug(f"Registered skill: {skill.name}")

    def unregister_skill(self, skill_name: str) -> None:
        """Unregister a skill"""
        self._skills.pop(skill_name, None)

    def get_skill(self, skill_name: str) -> Optional[BaseSkill]:
        """Get a skill by name"""
        return self._skills.get(skill_name)

    def list_skills(self) -> List[str]:
        """List all registered skills"""
        return list(self._skills.keys())

    # ==================== EXECUTION ====================

    async def execute(
        self,
        agent_input: AgentInput,
        context: Optional[ExecutionContext] = None,
        guardrails_fn: Optional[Callable] = None
    ) -> AgentOutput:
        """Execute the agent"""
        if not self.config.enabled:
            return self._create_output(
                AgentStatus.FAILED,
                error="Agent is disabled"
            )

        context = context or ExecutionContext(user_id=agent_input.user_id)

        if not context.can_execute():
            return self._create_output(
                AgentStatus.FAILED,
                error="Max execution depth exceeded"
            )

        context.add_agent(self.config.name)
        self.status = AgentStatus.RUNNING

        try:
            start = time.time()

            # Input guardrails
            if guardrails_fn and self.config.require_guardrails:
                check = await self._run_guardrails(guardrails_fn, "input", agent_input)
                if not check.get("passed"):
                    return self._create_output(AgentStatus.FAILED, error=check.get("reason"))

            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    self._execute_impl(agent_input, context),
                    timeout=self.config.timeout_seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"Timeout after {self.config.timeout_seconds}s")

            # Output guardrails
            if guardrails_fn and self.config.require_guardrails:
                check = await self._run_guardrails(guardrails_fn, "output", result)
                if not check.get("passed"):
                    return self._create_output(AgentStatus.FAILED, error=check.get("reason"))

            elapsed = (time.time() - start) * 1000
            output = self._create_output(
                AgentStatus.COMPLETED,
                result=result,
                execution_time_ms=elapsed,
                metadata={"chain": context.agent_chain}
            )
            self.status = AgentStatus.COMPLETED

        except Exception as e:
            self._logger.error(f"Execution failed: {e}", exc_info=True)
            output = self._create_output(AgentStatus.FAILED, error=str(e))
            self.status = AgentStatus.FAILED

        finally:
            self._history.append(output)
            if len(self._history) > 100:
                self._history = self._history[-100:]

        return output

    @abstractmethod
    async def _execute_impl(
        self,
        agent_input: AgentInput,
        context: ExecutionContext
    ) -> Any:
        """Implement agent-specific logic"""
        pass

    # ==================== SKILL CALLING ====================

    async def call_skill(
        self,
        skill_name: str,
        **kwargs
    ) -> SkillResult:
        """Call a registered skill with retry logic"""
        skill = self.get_skill(skill_name)
        if not skill:
            raise ValueError(f"Skill '{skill_name}' not found")

        for attempt in range(self.config.max_retries + 1):
            try:
                is_valid, error = await skill.validate_input(**kwargs)
                if not is_valid:
                    return SkillResult(success=False, error=error)

                await skill.pre_execute(**kwargs)
                start = time.time()
                result = await skill.execute(**kwargs)
                elapsed = (time.time() - start) * 1000
                result = await skill.post_execute(result)
                result.execution_time_ms = elapsed

                skill._update_stats(result.success, elapsed)
                if result.success:
                    return result

            except Exception as e:
                if attempt < self.config.max_retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return SkillResult(success=False, error=str(e))

        return SkillResult(success=False, error="Skill failed after retries")

    # ==================== UTILITIES ====================

    async def _run_guardrails(
        self,
        guardrails_fn: Callable,
        check_type: str,
        data: Any
    ) -> Dict[str, Any]:
        """Run guardrails check"""
        try:
            if asyncio.iscoroutinefunction(guardrails_fn):
                return await guardrails_fn(check_type, data)
            return guardrails_fn(check_type, data)
        except Exception as e:
            self._logger.error(f"Guardrails error: {e}")
            return {"passed": False, "reason": str(e)}

    def _create_output(
        self,
        status: AgentStatus,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        execution_time_ms: float = 0.0,
        **kwargs
    ) -> AgentOutput:
        """Create agent output"""
        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.config.name,
            status=status,
            result=result,
            error=error,
            execution_time_ms=execution_time_ms,
            **kwargs
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get agent execution summary"""
        total = len(self._history)
        completed = sum(1 for h in self._history if h.status == AgentStatus.COMPLETED)
        avg_time = sum(h.execution_time_ms for h in self._history) / total if total > 0 else 0

        return {
            "agent": self.config.name,
            "type": self.config.agent_type.value,
            "status": self.status.value,
            "executions": total,
            "success_rate": (completed / total * 100) if total > 0 else 0,
            "avg_time_ms": avg_time,
            "skills": list(self._skills.keys())
        }

    def clear_history(self) -> None:
        """Clear execution history"""
        self._history.clear()

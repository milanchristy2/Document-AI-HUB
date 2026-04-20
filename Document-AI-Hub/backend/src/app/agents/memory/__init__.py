"""Simple memory system for agents."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    entry_type: str  # "query", "response", "context", "thought"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    """A single conversation turn (query + response)."""
    turn_id: str
    query: str
    response: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: str = ""
    session_id: str = ""


class SimpleMemory:
    """Simple in-memory storage for agent interactions."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.entries: List[MemoryEntry] = []
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        self._logger = logging.getLogger(f"{__name__}.SimpleMemory")
    
    def add_entry(self, entry: MemoryEntry) -> None:
        """Add a memory entry."""
        self.entries.append(entry)
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size:]
        self._logger.debug(f"Added memory entry: {entry.id}")
    
    def add_conversation_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Add a conversation turn to a session."""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        self.conversations[session_id].append(turn)
        self._logger.debug(f"Added turn to session {session_id}")
    
    def get_recent_entries(self, entry_type: Optional[str] = None, limit: int = 10) -> List[MemoryEntry]:
        """Get recent memory entries, optionally filtered by type."""
        entries = self.entries
        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]
        return entries[-limit:]
    
    def get_session_context(self, session_id: str, limit: int = 5) -> List[ConversationTurn]:
        """Get recent conversation turns for a session."""
        if session_id not in self.conversations:
            return []
        turns = self.conversations[session_id]
        return turns[-limit:]
    
    def search_entries(self, keyword: str) -> List[MemoryEntry]:
        """Search entries by keyword."""
        keyword_lower = keyword.lower()
        return [e for e in self.entries if keyword_lower in e.content.lower()]
    
    def clear(self) -> None:
        """Clear all memory."""
        self.entries.clear()
        self.conversations.clear()
        self._logger.info("Memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_entries": len(self.entries),
            "total_sessions": len(self.conversations),
            "max_size": self.max_size,
            "entry_types": list(set(e.entry_type for e in self.entries)),
            "total_turns": sum(len(turns) for turns in self.conversations.values())
        }


# Global memory instance
shared_memory = SimpleMemory(max_size=200)

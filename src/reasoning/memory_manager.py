"""
Symbolic Memory Manager for Moveworks-style reasoning agent.

This module implements the symbolic memory architecture that Moveworks uses
for state management throughout conversational processes.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from ..models.moveworks import ConversationContext, Slot, SlotInferenceResult


logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory in the symbolic memory architecture."""
    CONVERSATION_STATE = "conversation_state"
    SLOT_VALUES = "slot_values"
    PROCESS_CONTEXT = "process_context"
    USER_ATTRIBUTES = "user_attributes"
    ACTIVITY_HISTORY = "activity_history"
    DECISION_CONTEXT = "decision_context"


@dataclass
class MemoryEntry:
    """A single entry in the symbolic memory."""
    memory_type: MemoryType
    key: str
    value: Any
    timestamp: datetime
    confidence: float = 1.0
    source: str = "system"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_type": self.memory_type.value,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata or {}
        }


class SymbolicMemoryManager:
    """
    Manages symbolic memory for conversational processes.
    
    This implements Moveworks' approach to state management where the AI agent
    keeps track of possible values and selected values using a symbolic memory
    architecture throughout the process.
    """
    
    def __init__(self):
        """Initialize the symbolic memory manager."""
        self.memory: Dict[str, Dict[str, MemoryEntry]] = {
            memory_type.value: {} for memory_type in MemoryType
        }
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
    def store_memory(
        self,
        memory_type: MemoryType,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a memory entry."""
        entry = MemoryEntry(
            memory_type=memory_type,
            key=key,
            value=value,
            timestamp=datetime.now(),
            confidence=confidence,
            source=source,
            metadata=metadata
        )
        
        self.memory[memory_type.value][key] = entry
        logger.debug(f"Stored memory: {memory_type.value}.{key} = {value}")
    
    def retrieve_memory(
        self,
        memory_type: MemoryType,
        key: str,
        default: Any = None
    ) -> Any:
        """Retrieve a memory value."""
        entry = self.memory[memory_type.value].get(key)
        if entry:
            return entry.value
        return default
    
    def get_memory_entry(
        self,
        memory_type: MemoryType,
        key: str
    ) -> Optional[MemoryEntry]:
        """Get the full memory entry."""
        return self.memory[memory_type.value].get(key)
    
    def update_slot_value(
        self,
        slot_name: str,
        value: Any,
        confidence: float,
        source: str = "inference"
    ) -> None:
        """Update a slot value in memory."""
        self.store_memory(
            MemoryType.SLOT_VALUES,
            slot_name,
            value,
            confidence=confidence,
            source=source,
            metadata={"inferred_at": datetime.now().isoformat()}
        )
    
    def get_slot_value(self, slot_name: str) -> Any:
        """Get a slot value from memory."""
        return self.retrieve_memory(MemoryType.SLOT_VALUES, slot_name)
    
    def get_all_slot_values(self) -> Dict[str, Any]:
        """Get all slot values."""
        slot_memory = self.memory[MemoryType.SLOT_VALUES.value]
        return {key: entry.value for key, entry in slot_memory.items()}
    
    def store_conversation_context(
        self,
        session_id: str,
        context: ConversationContext
    ) -> None:
        """Store conversation context."""
        self.conversation_contexts[session_id] = context
        
        # Also store key context elements in symbolic memory
        self.store_memory(
            MemoryType.CONVERSATION_STATE,
            f"{session_id}.current_process",
            context.current_process
        )
        
        self.store_memory(
            MemoryType.CONVERSATION_STATE,
            f"{session_id}.current_activity",
            context.current_activity
        )
        
        self.store_memory(
            MemoryType.USER_ATTRIBUTES,
            f"{session_id}.user_attributes",
            context.user_attributes
        )
    
    def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context."""
        return self.conversation_contexts.get(session_id)
    
    def store_activity_result(
        self,
        session_id: str,
        activity_index: int,
        result: Dict[str, Any]
    ) -> None:
        """Store activity execution result."""
        key = f"{session_id}.activity_{activity_index}"
        self.store_memory(
            MemoryType.ACTIVITY_HISTORY,
            key,
            result,
            source="activity_execution"
        )
    
    def get_activity_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get activity execution history."""
        history = []
        activity_memory = self.memory[MemoryType.ACTIVITY_HISTORY.value]
        
        for key, entry in activity_memory.items():
            if key.startswith(f"{session_id}.activity_"):
                history.append({
                    "activity_index": int(key.split("_")[-1]),
                    "result": entry.value,
                    "timestamp": entry.timestamp
                })
        
        return sorted(history, key=lambda x: x["activity_index"])
    
    def store_decision_context(
        self,
        session_id: str,
        decision_key: str,
        context: Dict[str, Any]
    ) -> None:
        """Store decision context for decision policies."""
        key = f"{session_id}.{decision_key}"
        self.store_memory(
            MemoryType.DECISION_CONTEXT,
            key,
            context,
            source="decision_policy"
        )
    
    def get_decision_context(
        self,
        session_id: str,
        decision_key: str
    ) -> Optional[Dict[str, Any]]:
        """Get decision context."""
        key = f"{session_id}.{decision_key}"
        return self.retrieve_memory(MemoryType.DECISION_CONTEXT, key)
    
    def clear_session_memory(self, session_id: str) -> None:
        """Clear all memory for a session."""
        # Remove from conversation contexts
        self.conversation_contexts.pop(session_id, None)
        
        # Remove from symbolic memory
        for memory_type in MemoryType:
            memory_dict = self.memory[memory_type.value]
            keys_to_remove = [
                key for key in memory_dict.keys()
                if key.startswith(f"{session_id}.")
            ]
            for key in keys_to_remove:
                del memory_dict[key]
        
        logger.info(f"Cleared memory for session: {session_id}")
    
    def get_memory_snapshot(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a snapshot of current memory state."""
        snapshot = {}
        
        for memory_type in MemoryType:
            memory_dict = self.memory[memory_type.value]
            
            if session_id:
                # Filter by session
                filtered_memory = {
                    key: entry.to_dict()
                    for key, entry in memory_dict.items()
                    if key.startswith(f"{session_id}.") or not "." in key
                }
            else:
                # All memory
                filtered_memory = {
                    key: entry.to_dict()
                    for key, entry in memory_dict.items()
                }
            
            if filtered_memory:
                snapshot[memory_type.value] = filtered_memory
        
        return snapshot
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            "total_entries": 0,
            "by_type": {},
            "active_sessions": len(self.conversation_contexts)
        }
        
        for memory_type in MemoryType:
            count = len(self.memory[memory_type.value])
            stats["by_type"][memory_type.value] = count
            stats["total_entries"] += count
        
        return stats

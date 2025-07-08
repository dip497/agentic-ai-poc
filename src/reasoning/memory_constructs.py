"""
Moveworks Memory Constructs Implementation.
Implements the four core memory types: Semantic, Episodic, Procedure, and Working Memory.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json


class MemoryType(Enum):
    """Types of memory constructs in the Moveworks architecture."""
    SEMANTIC = "semantic"      # Entity knowledge, domain awareness
    EPISODIC = "episodic"      # Conversation context, message history
    PROCEDURE = "procedure"    # Plugin capabilities, business processes
    WORKING = "working"        # Process state, variable tracking


@dataclass
class DomainDefinition:
    """Dynamic domain definition stored in database."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""  # e.g., "IT_DOMAIN", "HR_DOMAIN", etc.
    display_name: str = ""  # e.g., "Information Technology", "Human Resources"
    description: str = ""
    parent_domain: Optional[str] = None  # For hierarchical domains
    keywords: List[str] = field(default_factory=list)  # Domain keywords for classification
    trigger_phrases: List[str] = field(default_factory=list)  # Phrases that indicate this domain
    confidence_threshold: float = 0.7
    embedding: Optional[List[float]] = None  # Vector embedding for semantic classification
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SemanticMemoryEntry:
    """
    Semantic Memory: Knowledge of organizational content, entities, and terminology.
    Used to understand the semantics of conversations with users.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: str = ""  # u_JiraIssue, u_ServiceNowTicket, etc.
    entity_name: str = ""
    entity_description: str = ""
    domain: str = "GENERAL_DOMAIN"  # Dynamic domain name from database
    properties: Dict[str, Any] = field(default_factory=dict)
    synonyms: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None  # Vector embedding for similarity search
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class EpisodicMemoryEntry:
    """
    Episodic Memory: Current conversation context and history.
    Maintains 6-20 message window as per Moveworks standards.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    user_id: str = ""
    message_type: str = ""  # user, assistant, system
    content: str = ""
    intent: Optional[str] = None
    entities_extracted: Dict[str, Any] = field(default_factory=dict)
    slot_values: Dict[str, Any] = field(default_factory=dict)
    plugin_calls: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    sequence_number: int = 0


@dataclass
class ConversationContext:
    """
    Complete conversation context with episodic memory management.
    Implements Moveworks 6-20 message window.
    """
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    domain: Optional[str] = None  # Dynamic domain name from database
    route: str = ""  # DM, Notification, Ticket, Channel
    messages: List[EpisodicMemoryEntry] = field(default_factory=list)
    persistent_slots: Dict[str, Any] = field(default_factory=dict)
    active_plugins: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_message(self, message: EpisodicMemoryEntry):
        """Add message and maintain window size (6-20 messages)."""
        message.sequence_number = len(self.messages)
        self.messages.append(message)
        
        # Maintain window size - keep last 20 messages
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]
            # Update sequence numbers
            for i, msg in enumerate(self.messages):
                msg.sequence_number = i
        
        self.last_updated = datetime.now()
    
    def get_recent_context(self, max_messages: int = 6) -> List[EpisodicMemoryEntry]:
        """Get recent context (default 6 messages as per Moveworks)."""
        return self.messages[-max_messages:] if self.messages else []


@dataclass
class ProcedureMemoryEntry:
    """
    Procedure Memory: Knowledge of available tasks and business processes.
    Used to select the right tools and how to apply them.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    plugin_id: str = ""
    plugin_name: str = ""
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    trigger_utterances: List[str] = field(default_factory=list)
    domain_compatibility: List[str] = field(default_factory=list)  # Dynamic domain names
    required_slots: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    business_rules: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # Vector embedding for capability matching
    confidence_threshold: float = 0.7
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorkingMemoryEntry:
    """
    Working Memory: Tracks operations and processes in progress.
    Implements variable tracking framework for business object integrity.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    process_id: str = ""
    process_name: str = ""
    current_step: str = ""
    status: str = ""  # pending, in_progress, completed, failed
    variables: Dict[str, Any] = field(default_factory=dict)  # Variable tracking
    business_objects: Dict[str, Any] = field(default_factory=dict)  # Tracked objects
    step_history: List[Dict[str, Any]] = field(default_factory=list)
    references: List[str] = field(default_factory=list)  # For grounding responses
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def track_variable(self, name: str, value: Any, object_type: Optional[str] = None):
        """Track a variable with type safety for business objects."""
        self.variables[name] = {
            "value": value,
            "type": object_type or type(value).__name__,
            "tracked_at": datetime.now().isoformat()
        }
        
        # If it's a business object, track it separately
        if object_type and object_type.startswith("u_"):
            self.business_objects[name] = {
                "id": value.get("id") if isinstance(value, dict) else str(value),
                "type": object_type,
                "data": value,
                "tracked_at": datetime.now().isoformat()
            }
        
        self.updated_at = datetime.now()
    
    def add_step(self, step_name: str, result: Any, references: Optional[List[str]] = None):
        """Add a process step with results and references."""
        step = {
            "step": step_name,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "references": references or []
        }
        self.step_history.append(step)
        self.current_step = step_name
        
        if references:
            self.references.extend(references)
        
        self.updated_at = datetime.now()


@dataclass
class MemorySnapshot:
    """
    Complete memory state snapshot for reasoning engine.
    Combines all four memory types for decision making.
    """
    conversation_context: ConversationContext
    relevant_semantic_entries: List[SemanticMemoryEntry] = field(default_factory=list)
    available_procedures: List[ProcedureMemoryEntry] = field(default_factory=list)
    active_working_memory: List[WorkingMemoryEntry] = field(default_factory=list)
    snapshot_timestamp: datetime = field(default_factory=datetime.now)
    
    def get_domain_context(self) -> Optional[str]:
        """Get the primary domain for this conversation."""
        return self.conversation_context.domain
    
    def get_available_capabilities(self) -> List[str]:
        """Get all available capabilities from procedure memory."""
        capabilities = []
        for proc in self.available_procedures:
            capabilities.extend(proc.capabilities)
        return list(set(capabilities))
    
    def get_tracked_variables(self) -> Dict[str, Any]:
        """Get all tracked variables from working memory."""
        variables = {}
        for working_mem in self.active_working_memory:
            variables.update(working_mem.variables)
        return variables
    
    def get_business_objects(self) -> Dict[str, Any]:
        """Get all tracked business objects."""
        objects = {}
        for working_mem in self.active_working_memory:
            objects.update(working_mem.business_objects)
        return objects

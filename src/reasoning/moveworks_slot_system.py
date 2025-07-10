"""
Moveworks Slot-Based Memory System

This module implements the actual Moveworks slot-based memory architecture:
1. Slots: Core memory construct for conversational processes
2. Resolver Strategies: Convert user input to system data types  
3. Data Types: u_ prefixed custom types (u_JiraIssue, u_ServiceNowTicket)
4. Conversational Process State: Track process execution through slot resolution

This follows the official Moveworks architecture patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid
import json


class SlotInferencePolicy(Enum):
    """Slot inference policy options from Moveworks documentation."""
    INFER_IF_AVAILABLE = "Infer slot value if available"
    ALWAYS_ASK = "Always explicitly ask for slot"


class ResolverMethodType(Enum):
    """Types of resolver methods in Moveworks."""
    STATIC = "Static"
    DYNAMIC = "Dynamic" 
    API = "API"


@dataclass
class MoveworksDataType:
    """
    Moveworks data type definition with u_ prefix convention.
    Examples: u_JiraIssue, u_ServiceNowTicket, u_SalesforceAccount
    """
    name: str  # u_JiraIssue, u_ServiceNowTicket, etc.
    schema: Dict[str, Any]  # JSON schema for the data type
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure data type follows u_ naming convention."""
        if not self.name.startswith("u_"):
            self.name = f"u_{self.name}"


@dataclass
class StaticResolverOption:
    """Static resolver option mapping display value to raw value."""
    display_value: str  # What user sees: "Vacation"
    raw_value: str      # What system uses: "vacation"


@dataclass
class ResolverStrategy:
    """
    Moveworks resolver strategy for converting user input to system data types.
    Each strategy is bound to exactly one data type.
    """
    method_name: str
    method_type: ResolverMethodType
    output_data_type: str  # u_JiraIssue, u_ServiceNowTicket, etc.
    
    # For Static resolvers
    static_options: List[StaticResolverOption] = field(default_factory=list)
    
    # For Dynamic/API resolvers
    api_endpoint: Optional[str] = None
    api_method: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    
    # Resolver configuration
    description: str = ""
    
    def validate_output_type(self) -> bool:
        """Validate that output data type follows u_ convention."""
        return self.output_data_type.startswith("u_") or self.output_data_type in ["string", "number", "boolean", "User"]


@dataclass
class MoveworksSlot:
    """
    Core Moveworks slot definition following official patterns.
    Slots are the primary memory construct in Moveworks architecture.
    """
    name: str
    data_type: str  # string, number, boolean, User, or u_CustomType
    description: str
    
    # Slot policies
    inference_policy: SlotInferencePolicy = SlotInferencePolicy.INFER_IF_AVAILABLE
    validation_policy: str = ""  # DSL rule like "value > 0" or "$PARSE_TIME(value) > $TIME()"
    validation_description: str = ""
    
    # Resolver strategy for complex types
    resolver_strategy: Optional[ResolverStrategy] = None
    
    # Slot configuration
    is_required: bool = True
    is_list: bool = False  # List[type] vs single value
    default_value: Any = None
    
    # Metadata
    slot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate slot configuration."""
        if self.data_type.startswith("u_") and not self.resolver_strategy:
            raise ValueError(f"Custom data type {self.data_type} requires a resolver strategy")


@dataclass
class SlotResolutionResult:
    """Result of slot value resolution."""
    slot_name: str
    resolved_value: Any
    raw_value: Any  # Original user input
    confidence: float = 1.0
    resolution_method: str = ""  # "inferred", "explicit", "default"
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    # Metadata
    resolved_at: datetime = field(default_factory=datetime.now)
    resolver_used: Optional[str] = None


@dataclass
class ConversationMessage:
    """Simple conversation message for context."""
    content: str
    actor: str  # "user" or "assistant"
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: str = "text"  # "text", "form", "button_click"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MoveworksConversationContext:
    """
    Moveworks conversation context - replaces episodic memory.
    Tracks conversation state through slot resolution.
    """
    conversation_id: str
    user_id: str
    primary_domain: str  # IT, HR, Finance, etc.
    
    # Message history (simple list, not complex episodic memory)
    message_history: List[ConversationMessage] = field(default_factory=list)
    
    # Current process state
    current_process_id: Optional[str] = None
    resolved_slots: Dict[str, SlotResolutionResult] = field(default_factory=dict)
    pending_slots: List[str] = field(default_factory=list)
    
    # Activity tracking
    current_activity: Optional[str] = None
    activity_outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_message(self, content: str, actor: str, message_type: str = "text"):
        """Add message to conversation history."""
        message = ConversationMessage(
            content=content,
            actor=actor,
            message_type=message_type
        )
        self.message_history.append(message)
        self.last_updated = datetime.now()
    
    def resolve_slot(self, slot_name: str, result: SlotResolutionResult):
        """Mark slot as resolved."""
        self.resolved_slots[slot_name] = result
        if slot_name in self.pending_slots:
            self.pending_slots.remove(slot_name)
        self.last_updated = datetime.now()
    
    def get_slot_value(self, slot_name: str) -> Any:
        """Get resolved slot value."""
        if slot_name in self.resolved_slots:
            return self.resolved_slots[slot_name].resolved_value
        return None
    
    def is_slot_resolved(self, slot_name: str) -> bool:
        """Check if slot is resolved."""
        return slot_name in self.resolved_slots


@dataclass
class ConversationalProcessState:
    """
    Moveworks conversational process state - replaces working memory.
    Tracks process execution through slot resolution.
    """
    process_id: str
    process_name: str
    conversation_id: str
    
    # Process definition
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    
    # Execution state
    current_step: str = "slot_resolution"
    completed_activities: List[str] = field(default_factory=list)
    pending_activities: List[str] = field(default_factory=list)
    
    # Results
    activity_results: Dict[str, Any] = field(default_factory=dict)
    final_output: Optional[Dict[str, Any]] = None
    
    # Status
    status: str = "active"  # active, completed, failed, cancelled
    error_message: Optional[str] = None
    
    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def all_required_slots_resolved(self, context: MoveworksConversationContext) -> bool:
        """Check if all required slots are resolved."""
        return all(context.is_slot_resolved(slot) for slot in self.required_slots)
    
    def get_next_pending_slot(self, context: MoveworksConversationContext) -> Optional[str]:
        """Get next slot that needs resolution."""
        for slot_name in self.required_slots:
            if not context.is_slot_resolved(slot_name):
                return slot_name
        return None
    
    def complete_activity(self, activity_name: str, result: Any):
        """Mark activity as completed with result."""
        if activity_name in self.pending_activities:
            self.pending_activities.remove(activity_name)
        self.completed_activities.append(activity_name)
        self.activity_results[activity_name] = result
    
    def complete_process(self, final_output: Dict[str, Any]):
        """Mark process as completed."""
        self.status = "completed"
        self.final_output = final_output
        self.completed_at = datetime.now()


# Built-in data types following Moveworks patterns
MOVEWORKS_BUILTIN_TYPES = {
    "string": {"type": "string", "description": "Text value"},
    "number": {"type": "number", "description": "Floating point number"},
    "integer": {"type": "integer", "description": "Whole number"},
    "boolean": {"type": "boolean", "description": "True/false value"},
    "User": {"type": "object", "description": "Employee in organization", "resolver": "built_in_user_resolver"}
}

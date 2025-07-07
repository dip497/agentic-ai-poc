"""
Moveworks core data models - exact replication of Moveworks architecture.

Based on Moveworks documentation, this module defines:
- Conversational Processes (not intents)
- Activities (Action, Content, Decision)
- Slots with inference policies
- Built-in actions and DSL integration
"""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


# Enums based on Moveworks documentation

class DataType(str, Enum):
    """Moveworks data types for slots."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    LIST_STRING = "List[string]"
    LIST_NUMBER = "List[number]"
    OBJECT = "object"
    LIST_OBJECT = "List[object]"


class SlotInferencePolicy(str, Enum):
    """Moveworks slot inference policies."""
    INFER_IF_AVAILABLE = "Infer slot value if available"
    ALWAYS_ASK = "Always explicitly ask for slot"


class ActivityType(str, Enum):
    """Types of Activities in Conversational Processes."""
    ACTION = "action"
    CONTENT = "content"
    DECISION = "decision"


class ResolverMethodType(str, Enum):
    """Moveworks resolver method types."""
    STATIC = "Static"
    DYNAMIC = "Dynamic"
    VECTOR_SEARCH = "Vector Search"
    CUSTOM = "Custom"


class ConfirmationPolicy(str, Enum):
    """Activity confirmation policies."""
    REQUIRE_CONSENT = "Require consent from the user"
    NO_CONFIRMATION = "No confirmation required"


# Core Moveworks Models

class StaticOption(BaseModel):
    """Static resolver option."""
    display_value: str = Field(..., description="Display value shown to user")
    raw_value: str = Field(..., description="Raw value used by system")


class ResolverStrategy(BaseModel):
    """Moveworks resolver strategy configuration."""
    method_name: str = Field(..., description="Name of the resolver method")
    method_type: ResolverMethodType = Field(..., description="Type of resolver method")
    
    # Static resolver options
    static_options: Optional[List[StaticOption]] = None
    
    # Vector search config
    vector_store_name: Optional[str] = None
    similarity_threshold: Optional[float] = 0.7
    max_results: Optional[int] = 5
    
    # Dynamic resolver config
    api_endpoint: Optional[str] = None
    api_config: Optional[Dict[str, Any]] = None
    
    # Custom resolver config
    custom_function: Optional[str] = None


class Slot(BaseModel):
    """Moveworks Conversational Process Slot."""
    name: str = Field(..., description="Slot name")
    data_type: DataType = Field(..., description="Data type of the slot")
    slot_description: str = Field(..., description="Description that guides AI behavior")
    
    # Validation
    slot_validation_policy: Optional[str] = Field(None, description="DSL validation rule")
    slot_validation_description: Optional[str] = Field(None, description="Validation error message")
    
    # Inference policy
    slot_inference_policy: SlotInferencePolicy = Field(SlotInferencePolicy.INFER_IF_AVAILABLE)
    
    # Resolver strategy (optional)
    resolver_strategy: Optional[ResolverStrategy] = None


class InputMapping(BaseModel):
    """Input mapping for activities."""
    mappings: Dict[str, str] = Field(..., description="Input parameter mappings")


class OutputMapping(BaseModel):
    """Output mapping for activities."""
    dot_walk_path: Optional[str] = Field(None, description="Dot walk path for output extraction")
    output_key: str = Field(..., description="Key for storing activity output")


class ProgressUpdates(BaseModel):
    """Progress update messages for activities."""
    on_pending: Optional[str] = Field(None, description="Message while activity is pending")
    on_complete: Optional[str] = Field(None, description="Message when activity completes")


class Activity(BaseModel):
    """Moveworks Activity - core building block of Conversational Processes."""
    activity_type: ActivityType = Field(..., description="Type of activity")
    
    # Common fields
    required_slots: List[str] = Field(default_factory=list, description="Required slots for this activity")
    confirmation_policy: ConfirmationPolicy = Field(ConfirmationPolicy.NO_CONFIRMATION)
    
    # Action Activity fields
    action_name: Optional[str] = Field(None, description="Name of action to execute")
    input_mapping: Optional[InputMapping] = Field(None, description="Input parameter mappings")
    output_mapping: Optional[OutputMapping] = Field(None, description="Output mapping configuration")
    progress_updates: Optional[ProgressUpdates] = Field(None, description="Progress update messages")
    
    # Content Activity fields
    content_text: Optional[str] = Field(None, description="Content text to display")
    content_html: Optional[str] = Field(None, description="HTML content to display")
    
    # Decision Activity fields
    decision_cases: Optional[List[Dict[str, Any]]] = Field(None, description="Decision cases with DSL conditions")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ConversationalProcess(BaseModel):
    """Moveworks Conversational Process - houses the logic for a Plugin."""
    title: str = Field(..., description="Process title")
    description: str = Field(..., description="Process description")
    
    # Trigger utterances (not training examples)
    trigger_utterances: List[str] = Field(..., description="Natural language utterances that trigger this process")
    
    # Core components
    slots: List[Slot] = Field(default_factory=list, description="Slots required by the process")
    activities: List[Activity] = Field(default_factory=list, description="Activities to perform")
    
    # Decision policies (control when to run activities)
    decision_policies: Optional[List[Dict[str, Any]]] = Field(None, description="Decision policies for flow control")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class Plugin(BaseModel):
    """Moveworks Plugin - contains one or more Conversational Processes."""
    name: str = Field(..., description="Plugin name")
    description: str = Field(..., description="Plugin description")
    
    # Conversational processes
    conversational_processes: List[ConversationalProcess] = Field(..., description="Processes in this plugin")
    
    # Plugin-level configuration
    access_policies: Optional[List[str]] = Field(None, description="Access control policies")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


# Built-in Action Models (Moveworks style)

class GenerateTextActionInput(BaseModel):
    """Input for mw.generate_text_action equivalent."""
    system_prompt: Optional[str] = Field(None, description="System prompt to set model behavior")
    user_input: str = Field(..., description="User context for generation")
    model: Optional[str] = Field("gpt-4o-mini", description="Model to use for generation")


class GenerateStructuredValueActionInput(BaseModel):
    """Input for mw.generate_structured_value_action equivalent."""
    payload: Dict[str, Any] = Field(..., description="Payload to analyze")
    output_schema: Dict[str, Any] = Field(..., description="JSON schema for output")
    system_prompt: Optional[str] = Field(None, description="Instructions for extraction")
    strict: bool = Field(False, description="Enforce strict schema adherence")
    model: Optional[str] = Field("gpt-4o-mini", description="Model to use")


class GetUserByEmailInput(BaseModel):
    """Input for mw.get_user_by_email equivalent."""
    user_email: str = Field(..., description="Email address of user to retrieve")


class SendChatNotificationInput(BaseModel):
    """Input for mw.send_plaintext_chat_notification equivalent."""
    message: str = Field(..., description="Message to send")
    user_record_id: str = Field(..., description="User record ID of recipient")


# Conversation State Models

class ConversationContext(BaseModel):
    """Context for a conversation."""
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    thread_id: str = Field(..., description="Thread identifier")
    
    # User attributes (meta_info.user in Moveworks)
    user_attributes: Dict[str, Any] = Field(default_factory=dict, description="User attributes")
    
    # Conversation data
    conversation_data: Dict[str, Any] = Field(default_factory=dict, description="Conversation data")
    
    # Current process state
    current_process: Optional[str] = Field(None, description="Current conversational process")
    current_activity: Optional[int] = Field(None, description="Current activity index")
    
    # Slot values
    slot_values: Dict[str, Any] = Field(default_factory=dict, description="Collected slot values")


class ProcessMatchResult(BaseModel):
    """Result of process matching."""
    process_name: str = Field(..., description="Matched process name")
    confidence: float = Field(..., description="Matching confidence")
    trigger_utterance: str = Field(..., description="Matched trigger utterance")


class SlotInferenceResult(BaseModel):
    """Result of slot inference."""
    slot_name: str = Field(..., description="Slot name")
    inferred_value: Optional[Any] = Field(None, description="Inferred value")
    confidence: float = Field(..., description="Inference confidence")
    needs_clarification: bool = Field(False, description="Whether clarification is needed")
    clarification_options: Optional[List[str]] = Field(None, description="Clarification options")


class ActivityResult(BaseModel):
    """Result of activity execution."""
    success: bool = Field(..., description="Whether activity succeeded")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Activity output data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    next_activity: Optional[int] = Field(None, description="Next activity to execute")
    requires_user_input: bool = Field(False, description="Whether user input is required")
    user_prompt: Optional[str] = Field(None, description="Prompt for user input")

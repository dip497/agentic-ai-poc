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
    """Moveworks data types for slots - complete system based on official docs."""
    # Primitive Data Types
    STRING = "string"
    INTEGER = "integer"  # Added: separate from number (floating point)
    NUMBER = "number"    # Floating point numbers
    BOOLEAN = "boolean"

    # Object Data Types
    USER = "User"        # Added: built-in User type with built-in resolver
    OBJECT = "object"    # Generic object type

    # List Data Types (for all primitives and objects)
    LIST_STRING = "List[string]"
    LIST_INTEGER = "List[integer]"    # Added: integer arrays
    LIST_NUMBER = "List[number]"
    LIST_BOOLEAN = "List[boolean]"    # Added: boolean arrays
    LIST_USER = "List[User]"          # Added: User arrays
    LIST_OBJECT = "List[object]"

    # Custom data types will be handled separately with u_<DataTypeName> convention


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
    """Moveworks resolver method types - simplified to match official docs."""
    STATIC = "Static"
    DYNAMIC = "Dynamic"


class ConfirmationPolicy(str, Enum):
    """Activity confirmation policies."""
    REQUIRE_CONSENT = "Require consent from the user"
    NO_CONFIRMATION = "No confirmation required"


# Core Moveworks Models

class CustomDataType(BaseModel):
    """Custom data type with u_<DataTypeName> convention."""
    name: str = Field(..., description="Data type name (must follow u_<DataTypeName> convention)")
    description: str = Field(..., description="Detailed description for AI triggering accuracy")
    data_schema: Dict[str, Any] = Field(..., description="JSON schema definition")
    default_resolver_strategy: Optional[str] = Field(None, description="Default resolver strategy name")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def validate_name_convention(self) -> bool:
        """Validate that name follows u_<DataTypeName> convention."""
        return self.name.startswith("u_") and len(self.name) > 2


class StaticOption(BaseModel):
    """Static resolver option."""
    display_value: str = Field(..., description="Display value shown to user")
    raw_value: str = Field(..., description="Raw value used by system")


class ResolverMethod(BaseModel):
    """Individual resolver method within a strategy - matches Moveworks architecture."""
    name: str = Field(..., description="Method name (must be snake_case)")
    method_type: ResolverMethodType = Field(..., description="Static or Dynamic")

    # Static method configuration (only for Static type)
    static_options: Optional[List[StaticOption]] = Field(None, description="Static options (Static methods only)")

    # Dynamic method configuration (only for Dynamic type)
    action_name: Optional[str] = Field(None, description="Action to execute (Dynamic methods only)")
    input_arguments: Optional[Dict[str, Any]] = Field(None, description="Input argument schema")
    output_mapping: Optional[str] = Field(None, description="Output mapping path (e.g., '.issues')")

    # Metadata
    description: Optional[str] = Field(None, description="Method description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def validate_method_constraints(self) -> bool:
        """Validate method type constraints."""
        if self.method_type == ResolverMethodType.STATIC:
            return self.static_options is not None and self.action_name is None
        elif self.method_type == ResolverMethodType.DYNAMIC:
            return self.action_name is not None and self.static_options is None
        return False


class ResolverStrategy(BaseModel):
    """Moveworks resolver strategy - collection of methods for one data type."""
    name: str = Field(..., description="Strategy name (e.g., 'JiraIssueResolver')")
    data_type: Union[DataType, str] = Field(..., description="Data type this strategy resolves")
    description: str = Field(..., description="Strategy description")

    # Collection of methods - AI agent picks the best one
    methods: List[ResolverMethod] = Field(..., description="Resolver methods")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def validate_method_types(self) -> bool:
        """Validate method type constraints: 1 Static OR Multiple Dynamic (never both)."""
        static_methods = [m for m in self.methods if m.method_type == ResolverMethodType.STATIC]
        dynamic_methods = [m for m in self.methods if m.method_type == ResolverMethodType.DYNAMIC]

        # Rule: 1 Static Method OR Multiple Dynamic Methods (never both)
        if len(static_methods) > 0 and len(dynamic_methods) > 0:
            return False  # Cannot mix static and dynamic
        if len(static_methods) > 1:
            return False  # Only 1 static method allowed
        if len(static_methods) == 0 and len(dynamic_methods) == 0:
            return False  # Must have at least one method

        return True

    def get_method_by_name(self, method_name: str) -> Optional[ResolverMethod]:
        """Get method by name."""
        return next((m for m in self.methods if m.name == method_name), None)


class Slot(BaseModel):
    """Moveworks Conversational Process Slot - updated to match new resolver architecture."""
    name: str = Field(..., description="Slot name")
    data_type: Union[DataType, str] = Field(..., description="Data type of the slot (built-in or custom)")
    slot_description: str = Field(..., description="Description that guides AI behavior")

    # Custom data type support
    custom_data_type_name: Optional[str] = Field(None, description="Name of custom data type if data_type is custom")

    # Validation
    slot_validation_policy: Optional[str] = Field(None, description="DSL validation rule")
    slot_validation_description: Optional[str] = Field(None, description="Validation error message")

    # Inference policy
    slot_inference_policy: SlotInferencePolicy = Field(SlotInferencePolicy.INFER_IF_AVAILABLE)

    # Resolver strategy reference (optional - overrides data type default)
    resolver_strategy_name: Optional[str] = Field(None, description="Name of specific resolver strategy to use")


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

    # Moveworks Plugin Selection Metadata
    capabilities: List[str] = Field(default_factory=list, description="List of capabilities this plugin provides")
    domain_compatibility: List[str] = Field(default_factory=list, description="Domains this plugin is compatible with (e.g., 'hr', 'it', 'finance')")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold for plugin selection")

    # Examples for AI-powered selection
    positive_examples: List[str] = Field(default_factory=list, description="Example queries that should trigger this plugin")
    negative_examples: List[str] = Field(default_factory=list, description="Example queries that should NOT trigger this plugin")

    # Plugin-level configuration
    access_policies: Optional[List[str]] = Field(None, description="Access control policies")
    launch_permissions: Dict[str, Any] = Field(default_factory=dict, description="Launch permission configuration")

    # Performance tracking
    usage_stats: Dict[str, Any] = Field(default_factory=dict, description="Plugin usage statistics")
    success_rate: float = Field(default=0.0, description="Historical success rate for this plugin")

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


# API Request/Response Models for Resolver Strategies

class ResolverMethodCreateRequest(BaseModel):
    """Request model for creating a resolver method."""
    name: str = Field(..., description="Method name (must be snake_case)")
    method_type: ResolverMethodType = Field(..., description="Static or Dynamic")
    description: Optional[str] = Field(None, description="Method description")

    # Static method fields
    static_options: Optional[List[StaticOption]] = Field(None, description="Static options (Static methods only)")

    # Dynamic method fields
    action_name: Optional[str] = Field(None, description="Action to execute (Dynamic methods only)")
    input_arguments: Optional[Dict[str, Any]] = Field(None, description="Input argument schema")
    output_mapping: Optional[str] = Field(None, description="Output mapping path")


class ResolverMethodUpdateRequest(BaseModel):
    """Request model for updating a resolver method."""
    name: Optional[str] = Field(None, description="Method name")
    method_type: Optional[ResolverMethodType] = Field(None, description="Static or Dynamic")
    description: Optional[str] = Field(None, description="Method description")
    static_options: Optional[List[StaticOption]] = Field(None, description="Static options")
    action_name: Optional[str] = Field(None, description="Action to execute")
    input_arguments: Optional[Dict[str, Any]] = Field(None, description="Input argument schema")
    output_mapping: Optional[str] = Field(None, description="Output mapping path")


class ResolverStrategyCreateRequest(BaseModel):
    """Request model for creating a resolver strategy."""
    name: str = Field(..., description="Strategy name")
    data_type: Union[DataType, str] = Field(..., description="Data type this strategy resolves")
    description: str = Field(..., description="Strategy description")
    methods: List[ResolverMethodCreateRequest] = Field(..., description="Resolver methods")


class ResolverStrategyUpdateRequest(BaseModel):
    """Request model for updating a resolver strategy."""
    name: Optional[str] = Field(None, description="Strategy name")
    data_type: Optional[Union[DataType, str]] = Field(None, description="Data type")
    description: Optional[str] = Field(None, description="Strategy description")
    methods: Optional[List[ResolverMethodCreateRequest]] = Field(None, description="Resolver methods")


class ResolverStrategyResponse(BaseModel):
    """Response model for resolver strategy."""
    id: str = Field(..., description="Strategy ID")
    name: str = Field(..., description="Strategy name")
    data_type: Union[DataType, str] = Field(..., description="Data type")
    description: str = Field(..., description="Strategy description")
    methods: List[ResolverMethod] = Field(..., description="Resolver methods")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")


class ResolverStrategyListResponse(BaseModel):
    """Response model for listing resolver strategies."""
    strategies: List[ResolverStrategyResponse] = Field(..., description="List of resolver strategies")
    total: int = Field(..., description="Total number of strategies")


class MethodSelectionRequest(BaseModel):
    """Request model for AI method selection."""
    strategy_name: str = Field(..., description="Resolver strategy name")
    user_input: str = Field(..., description="User input to analyze")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class MethodSelectionResponse(BaseModel):
    """Response model for AI method selection."""
    selected_method: str = Field(..., description="Selected method name")
    confidence: float = Field(..., description="Selection confidence")
    reasoning: str = Field(..., description="Why this method was selected")
    alternative_methods: Optional[List[str]] = Field(None, description="Alternative methods considered")

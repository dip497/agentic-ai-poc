"""
Moveworks Reasoning State Management.
Defines the state structure for the three reasoning loops using LangGraph.
"""

from typing import Dict, List, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from config.reasoning_config import get_reasoning_config

from .memory_constructs import MemorySnapshot, WorkingMemoryEntry


class ReasoningPhase(Enum):
    """Current phase in the reasoning process."""
    PLANNING = "planning"
    EXECUTION = "execution"
    USER_FEEDBACK = "user_feedback"
    COMPLETED = "completed"
    ERROR = "error"


class PlanStatus(Enum):
    """Status of a plan in the planning iteration loop."""
    DRAFT = "draft"
    EVALUATING = "evaluating"
    NEEDS_REVISION = "needs_revision"
    APPROVED = "approved"
    REJECTED = "rejected"


class ExecutionStatus(Enum):
    """Status of execution steps."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    ERROR = "error"
    NEEDS_USER_INPUT = "needs_user_input"
    COMPLETED = "completed"


@dataclass
class Tool:
    """Represents an available tool/plugin for execution."""
    id: str
    name: str
    description: str
    capabilities: List[str]
    required_inputs: List[str]
    optional_inputs: List[str] = field(default_factory=list)
    domain_compatibility: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str = ""
    action: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    dependencies: List[str] = field(default_factory=list)  # IDs of prerequisite steps
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time: Optional[datetime] = None


@dataclass
class Plan:
    """A complete execution plan from the planning iteration loop."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    confidence_score: float = 0.0
    evaluation_feedback: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next step that can be executed."""
        for step in self.steps:
            if step.status == ExecutionStatus.PENDING:
                # Check if all dependencies are completed
                if all(
                    any(s.id == dep_id and s.status == ExecutionStatus.SUCCESS 
                        for s in self.steps)
                    for dep_id in step.dependencies
                ):
                    return step
        return None
    
    def is_complete(self) -> bool:
        """Check if all steps are completed successfully."""
        return all(step.status == ExecutionStatus.SUCCESS for step in self.steps)
    
    def has_errors(self) -> bool:
        """Check if any steps have errors."""
        return any(step.status == ExecutionStatus.ERROR for step in self.steps)


@dataclass
class UserInteraction:
    """Represents a user interaction in the feedback loop."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: Literal["confirmation", "input_request", "clarification", "progress_update"] = "confirmation"
    message: str = ""
    options: List[str] = field(default_factory=list)
    user_response: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    is_resolved: bool = False


@dataclass
class ReasoningState:
    """
    Complete state for the Moveworks Reasoning Engine.
    Manages state across all three reasoning loops.
    """
    # Session Information
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    user_id: str = ""
    
    # Current State
    phase: ReasoningPhase = ReasoningPhase.PLANNING
    current_goal: str = ""
    original_query: str = ""
    
    # Memory Context
    memory_snapshot: Optional[MemorySnapshot] = None
    working_memory_id: Optional[str] = None
    
    # Planning Iteration Loop State
    available_tools: List[Tool] = field(default_factory=list)
    current_plan: Optional[Plan] = None
    plan_history: List[Plan] = field(default_factory=list)
    planning_iterations: int = 0
    max_planning_iterations: int = field(default_factory=lambda: get_reasoning_config().planning_iterations_max)
    
    # Execution Iteration Loop State
    current_step: Optional[PlanStep] = None
    execution_results: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    # User-Facing Feedback Loop State
    pending_user_interactions: List[UserInteraction] = field(default_factory=list)
    user_feedback_history: List[UserInteraction] = field(default_factory=list)
    requires_user_input: bool = False
    
    # Progress Tracking
    progress_messages: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_phase(self, new_phase: ReasoningPhase):
        """Update the current reasoning phase."""
        self.phase = new_phase
        self.updated_at = datetime.now()
        self.progress_messages.append(f"Entered {new_phase.value} phase")
    
    def add_progress_message(self, message: str):
        """Add a progress message."""
        self.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.updated_at = datetime.now()
    
    def add_error_message(self, message: str):
        """Add an error message."""
        self.error_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.updated_at = datetime.now()
    
    def set_current_plan(self, plan: Plan):
        """Set the current plan and archive the previous one."""
        if self.current_plan:
            self.plan_history.append(self.current_plan)
        self.current_plan = plan
        self.updated_at = datetime.now()
    
    def add_user_interaction(self, interaction: UserInteraction):
        """Add a pending user interaction."""
        self.pending_user_interactions.append(interaction)
        self.requires_user_input = True
        self.updated_at = datetime.now()
    
    def resolve_user_interaction(self, interaction_id: str, response: str):
        """Resolve a user interaction with the user's response."""
        for interaction in self.pending_user_interactions:
            if interaction.id == interaction_id:
                interaction.user_response = response
                interaction.is_resolved = True
                self.user_feedback_history.append(interaction)
                self.pending_user_interactions.remove(interaction)
                break
        
        # Check if all interactions are resolved
        self.requires_user_input = len(self.pending_user_interactions) > 0
        self.updated_at = datetime.now()
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get a summary of the current reasoning state."""
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "goal": self.current_goal,
            "plan_status": self.current_plan.status.value if self.current_plan else None,
            "current_step": self.current_step.action if self.current_step else None,
            "requires_user_input": self.requires_user_input,
            "pending_interactions": len(self.pending_user_interactions),
            "progress_messages": self.progress_messages[-3:],  # Last 3 messages
            "error_messages": self.error_messages[-3:] if self.error_messages else []
        }
    
    def is_complete(self) -> bool:
        """Check if the reasoning process is complete."""
        return (
            self.phase == ReasoningPhase.COMPLETED and
            self.current_plan is not None and
            self.current_plan.is_complete() and
            not self.requires_user_input
        )
    
    def has_errors(self) -> bool:
        """Check if there are any errors in the reasoning process."""
        return (
            self.phase == ReasoningPhase.ERROR or
            len(self.error_messages) > 0 or
            (self.current_plan is not None and self.current_plan.has_errors())
        )


@dataclass
class ReasoningEvent:
    """Events that can trigger state transitions in the reasoning loops."""
    type: Literal[
        "user_query", "plan_created", "plan_evaluated", "step_executed", 
        "user_response", "error_occurred", "process_completed"
    ]
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""  # Which loop generated this event


# Type aliases for LangGraph integration
ReasoningStateDict = Dict[str, Any]
ReasoningConfig = Dict[str, Any]

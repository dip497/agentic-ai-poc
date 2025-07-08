"""
Moveworks Three-Loop Reasoning Architecture.
Implements the actual Moveworks reasoning loops: Planning, Execution, and User Feedback.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated

from reasoning.plugin_selection_engine import MoveworksPluginSelector
from reasoning.multi_plugin_response import MultiPluginResponseEngine
from reasoning.memory_constructs import MemorySnapshot
from reasoning.moveworks_memory_manager import MoveworksMemoryManager
from config.loader import MoveworksConfigLoader
from llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class MoveworksReasoningState(TypedDict):
    """State for Moveworks three-loop reasoning."""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    user_context: Dict[str, Any]
    conversation_id: str

    # Planning Loop State
    current_plan: Optional[Dict[str, Any]]
    plan_iterations: int
    plan_feedback: List[str]

    # Execution Loop State
    execution_results: List[Dict[str, Any]]
    execution_step: int
    needs_user_input: bool

    # User Feedback Loop State
    user_feedback: Optional[str]
    awaiting_confirmation: bool

    # Slot Management State
    required_slots: List[str]
    filled_slots: Dict[str, Any]
    missing_slots: List[str]

    # Action Tracking State
    high_impact_actions: List[Dict[str, Any]]
    pending_confirmations: List[Dict[str, Any]]

    # Final Results
    final_response: str
    user_actions: List[Dict[str, Any]]
    memory_snapshot: Optional[Dict[str, Any]]


@dataclass
class MoveworksReasoningResult:
    """Result from Moveworks three-loop reasoning."""
    success: bool
    response: str
    selected_plugins: List[str] = field(default_factory=list)
    user_actions: List[Dict[str, Any]] = field(default_factory=list)
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    conversation_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class MoveworksThreeLoopEngine:
    """
    Implements the actual Moveworks three-loop reasoning architecture:
    1. Planning Iteration Loop (planner + plan evaluator)
    2. Execution Iteration Loop (execute + observe + adapt)
    3. User-facing Feedback Loop (user interaction)
    """
    
    def __init__(self, config_loader: MoveworksConfigLoader):
        self.config_loader = config_loader
        self.plugin_selector = None
        self.multi_plugin_engine = None
        self.memory_manager = None
        self.llm = None
        self.graph = None
        
        # Configuration
        self.max_planning_iterations = 3
        self.max_execution_steps = 5
        
    async def initialize(self):
        """Initialize all components and build the reasoning graph."""
        logger.info("Initializing Moveworks Three-Loop Reasoning Engine...")
        
        # Initialize LLM
        llm_factory = LLMFactory()
        self.llm = llm_factory.create_llm("gemini", model="gemini-1.5-flash")
        
        # Initialize plugin selector
        self.plugin_selector = MoveworksPluginSelector(self.config_loader)
        await self.plugin_selector.initialize()
        
        # Initialize multi-plugin engine
        self.multi_plugin_engine = MultiPluginResponseEngine(self.config_loader)
        await self.multi_plugin_engine.initialize()
        
        # Initialize memory manager
        self.memory_manager = MoveworksMemoryManager()
        await self.memory_manager.initialize()
        
        # Build the three-loop reasoning graph
        self._build_three_loop_graph()
        
        logger.info("Moveworks three-loop reasoning engine initialized")
    
    def _build_three_loop_graph(self):
        """Build the LangGraph for three-loop reasoning."""
        workflow = StateGraph(MoveworksReasoningState)
        
        # Add nodes for each loop
        workflow.add_node("load_memory", self._load_memory_node)
        workflow.add_node("planning_loop", self._planning_iteration_loop)
        workflow.add_node("execution_loop", self._execution_iteration_loop)
        workflow.add_node("user_feedback_loop", self._user_feedback_loop)
        workflow.add_node("finalize_response", self._finalize_response_node)
        
        # Define the three-loop flow
        workflow.set_entry_point("load_memory")
        workflow.add_edge("load_memory", "planning_loop")
        workflow.add_edge("planning_loop", "execution_loop")
        
        # Conditional edges for user feedback
        workflow.add_conditional_edges(
            "execution_loop",
            self._should_seek_user_feedback,
            {
                "user_feedback": "user_feedback_loop",
                "finalize": "finalize_response"
            }
        )
        
        workflow.add_conditional_edges(
            "user_feedback_loop",
            self._should_continue_execution,
            {
                "continue": "execution_loop",
                "replan": "planning_loop",
                "finalize": "finalize_response"
            }
        )
        
        workflow.add_edge("finalize_response", END)

        # Compile the graph with checkpointer for human-in-the-loop
        checkpointer = MemorySaver()
        self.graph = workflow.compile(checkpointer=checkpointer)
    
    async def process_request(
        self,
        user_query: str,
        user_context: Dict[str, Any],
        conversation_id: str = None
    ) -> MoveworksReasoningResult:
        """Process request through Moveworks three-loop reasoning."""
        if not conversation_id:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Processing via Moveworks three-loop reasoning: {user_query[:100]}...")
        
        try:
            # Initialize state
            initial_state = MoveworksReasoningState(
                messages=[HumanMessage(content=user_query)],
                user_query=user_query,
                user_context=user_context,
                conversation_id=conversation_id,
                current_plan=None,
                plan_iterations=0,
                plan_feedback=[],
                execution_results=[],
                execution_step=0,
                needs_user_input=False,
                user_feedback=None,
                awaiting_confirmation=False,
                required_slots=[],
                filled_slots={},
                missing_slots=[],
                high_impact_actions=[],
                pending_confirmations=[],
                final_response="",
                user_actions=[],
                memory_snapshot=None
            )
            
            # Execute the three-loop reasoning with proper config
            config = {"configurable": {"thread_id": conversation_id}}
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            # Extract results
            plugin_names = []
            for result in final_state["execution_results"]:
                if "plugins" in result:
                    plugin_names.extend([p.get("name", "unknown") for p in result["plugins"]])
            
            return MoveworksReasoningResult(
                success=True,
                response=final_state["final_response"],
                selected_plugins=list(set(plugin_names)),
                user_actions=final_state["user_actions"],
                execution_summary={
                    "plan_iterations": final_state["plan_iterations"],
                    "execution_steps": final_state["execution_step"],
                    "plugins_used": len(set(plugin_names)),
                    "conversation_id": conversation_id
                },
                conversation_id=conversation_id
            )
            
        except Exception as e:
            logger.error(f"Error in three-loop reasoning: {e}")
            return MoveworksReasoningResult(
                success=False,
                response=f"I encountered an error processing your request: {str(e)}",
                execution_summary={"error": str(e)},
                conversation_id=conversation_id
            )
    
    async def _load_memory_node(self, state: MoveworksReasoningState) -> MoveworksReasoningState:
        """Load conversation memory."""
        try:
            memory_snapshot = await self.memory_manager.get_memory_snapshot(
                state["conversation_id"], state["user_context"]
            )
            state["memory_snapshot"] = memory_snapshot.model_dump() if memory_snapshot else None
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")
            state["memory_snapshot"] = None
        
        return state
    
    async def _planning_iteration_loop(self, state: MoveworksReasoningState) -> MoveworksReasoningState:
        """
        Loop 1: Planning Iteration Loop
        Implements planner + plan evaluator cycle.
        """
        logger.info("Starting Planning Iteration Loop...")
        
        for iteration in range(self.max_planning_iterations):
            state["plan_iterations"] = iteration + 1
            
            # Step 1: Generate plan using LLM planner
            plan = await self._generate_plan(state)
            state["current_plan"] = plan
            
            # Step 2: Evaluate plan using LLM plan evaluator
            evaluation = await self._evaluate_plan(state, plan)
            
            if evaluation["approved"]:
                logger.info(f"Plan approved after {iteration + 1} iterations")
                break
            else:
                # Add feedback for next iteration
                state["plan_feedback"].append(evaluation["feedback"])
                logger.debug(f"Plan needs revision: {evaluation['feedback']}")
        
        if not state["current_plan"] or not evaluation.get("approved", False):
            raise RuntimeError("Failed to generate approved plan after maximum iterations")
        
        return state
    
    async def _execution_iteration_loop(self, state: MoveworksReasoningState) -> MoveworksReasoningState:
        """
        Loop 2: Execution Iteration Loop
        Implements execute -> observe -> adapt cycle.
        """
        logger.info("Starting Execution Iteration Loop...")
        
        if not state["current_plan"]:
            raise RuntimeError("No plan available for execution")
        
        plan_steps = state["current_plan"].get("steps", [])
        
        for step_idx, step in enumerate(plan_steps):
            if state["execution_step"] >= self.max_execution_steps:
                break
                
            state["execution_step"] = step_idx + 1
            logger.debug(f"Executing step {step_idx + 1}: {step}")
            
            # Execute step
            execution_result = await self._execute_step(state, step)
            state["execution_results"].append(execution_result)
            
            # Observe outcome and adapt
            adaptation = await self._observe_and_adapt(state, execution_result)
            
            if adaptation["needs_user_input"]:
                state["needs_user_input"] = True
                break
            elif adaptation["should_stop"]:
                break
        
        return state
    
    async def _user_feedback_loop(self, state: MoveworksReasoningState) -> MoveworksReasoningState:
        """
        Loop 3: User-facing Feedback Loop
        Handles user interaction and confirmation using LangGraph interrupts.
        """
        logger.info("Starting User-facing Feedback Loop...")

        # Generate status update for user
        status_update = await self._generate_status_update(state)

        # Check if we need specific user confirmation
        confirmation_needed = await self._check_confirmation_needed(state)

        if confirmation_needed["required"]:
            # Use LangGraph interrupt for real human-in-the-loop
            user_response = interrupt({
                "type": "user_confirmation",
                "status": status_update,
                "question": confirmation_needed["message"],
                "execution_results": state["execution_results"],
                "options": ["continue", "retry", "stop"]
            })

            # Process user response
            if user_response == "retry":
                # Reset execution to retry
                state["execution_step"] = 0
                state["execution_results"] = []
                state["needs_user_input"] = False
            elif user_response == "stop":
                # Stop execution
                state["final_response"] = "Execution stopped by user request."
                state["needs_user_input"] = False
            else:
                # Continue execution
                state["needs_user_input"] = False

            state["awaiting_confirmation"] = False
            state["user_feedback"] = user_response
        else:
            # No confirmation needed, continue
            state["awaiting_confirmation"] = False
            state["needs_user_input"] = False

        return state
    
    # Helper methods continue in next part...
    
    async def _generate_plan(self, state: MoveworksReasoningState) -> Dict[str, Any]:
        """Generate execution plan using LLM planner."""
        context = f"""User Query: {state['user_query']}
User Context: {state['user_context']}
Previous Feedback: {'; '.join(state['plan_feedback']) if state['plan_feedback'] else 'None'}

Create a step-by-step plan to address the user's request.
Respond with: step1|step2|step3

Example: Select relevant plugins|Execute plugin workflows|Combine responses"""
        
        try:
            response = await self.llm.ainvoke(context)
            response_text = response.content if hasattr(response, 'content') else str(response)
            steps = [step.strip() for step in response_text.split("|")]
            
            return {
                "id": f"plan_{state['plan_iterations']}",
                "steps": steps,
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            raise RuntimeError(f"Plan generation failed: {e}") from e
    
    async def _evaluate_plan(self, state: MoveworksReasoningState, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate plan using LLM plan evaluator."""
        steps_text = "; ".join(plan.get("steps", []))
        
        context = f"""Evaluate this execution plan:
User Query: {state['user_query']}
Plan Steps: {steps_text}

Is this plan complete and executable? Respond with: approved|feedback

Example: approved|Plan addresses all user requirements
Example: rejected|Missing step to validate user permissions"""
        
        try:
            response = await self.llm.ainvoke(context)
            response_text = response.content if hasattr(response, 'content') else str(response)

            if "|" not in response_text:
                return {"approved": False, "feedback": "Invalid evaluation response"}

            decision, feedback = response_text.split("|", 1)
            approved = decision.strip().lower() == "approved"
            
            return {
                "approved": approved,
                "feedback": feedback.strip()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating plan: {e}")
            return {"approved": False, "feedback": f"Evaluation error: {e}"}
    
    async def _execute_step(self, state: MoveworksReasoningState, step: str) -> Dict[str, Any]:
        """Execute a single plan step."""
        try:
            # Select plugins for this step
            selected_plugins = await self.plugin_selector.select_plugins(
                user_query=step,
                user_context=state["user_context"],
                max_plugins=2
            )
            
            if not selected_plugins:
                return {
                    "step": step,
                    "success": False,
                    "error": "No suitable plugins found",
                    "plugins": []
                }
            
            # Execute plugins
            multi_response = await self.multi_plugin_engine.execute_plugins(
                selected_plugins=selected_plugins,
                user_query=step,
                user_context=state["user_context"],
                execution_context={
                    "conversation_id": state["conversation_id"],
                    "step": step
                }
            )
            
            return {
                "step": step,
                "success": True,
                "plugins": [{"name": p.plugin.name, "confidence": p.confidence} for p in selected_plugins],
                "response": multi_response.primary_response,
                "actions": multi_response.combined_actions
            }
            
        except Exception as e:
            logger.error(f"Error executing step '{step}': {e}")
            return {
                "step": step,
                "success": False,
                "error": str(e),
                "plugins": []
            }
    
    async def _observe_and_adapt(self, state: MoveworksReasoningState, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Observe execution outcome and determine next action."""

        # Analyze execution result
        if not execution_result["success"]:
            error_msg = execution_result.get("error", "Unknown error")

            # Check if error requires user intervention
            critical_errors = ["permission denied", "authentication failed", "user confirmation required"]
            if any(critical in error_msg.lower() for critical in critical_errors):
                return {
                    "needs_user_input": True,
                    "should_stop": False,
                    "adaptation": f"User intervention needed: {error_msg}"
                }

            # Check if we should retry or continue
            retry_errors = ["timeout", "network error", "temporary failure"]
            if any(retry in error_msg.lower() for retry in retry_errors):
                return {
                    "needs_user_input": False,
                    "should_stop": False,
                    "adaptation": "Retry with exponential backoff"
                }

            # Fatal error - stop execution
            return {
                "needs_user_input": True,
                "should_stop": True,
                "adaptation": f"Fatal error encountered: {error_msg}"
            }

        # Success case - analyze if we need user confirmation
        plugins_used = execution_result.get("plugins", [])
        actions_taken = execution_result.get("actions", [])

        # Check for sensitive operations in the original user query
        user_query = state.get("user_query", "")
        sensitive_keywords = ["delete", "remove", "reset", "password", "permissions", "access"]
        if any(keyword in user_query.lower() for keyword in sensitive_keywords):
            return {
                "needs_user_input": True,
                "should_stop": False,
                "adaptation": f"Sensitive operation detected in query: '{user_query}' - requesting user confirmation"
            }

        # Check if any high-impact actions were taken
        high_impact_actions = ["delete", "reset", "modify_permissions", "send_email", "create_ticket"]
        if any(action.get("type", "").lower() in high_impact_actions for action in actions_taken):
            return {
                "needs_user_input": True,
                "should_stop": False,
                "adaptation": "High-impact actions taken, requesting user confirmation"
            }

        # Check if we have all required information
        required_slots = state.get("required_slots", [])
        filled_slots = state.get("filled_slots", {})
        missing_slots = [slot for slot in required_slots if slot not in filled_slots]

        if missing_slots:
            return {
                "needs_user_input": True,
                "should_stop": False,
                "adaptation": f"Missing required information: {', '.join(missing_slots)}"
            }

        # Normal success - continue execution
        return {
            "needs_user_input": False,
            "should_stop": False,
            "adaptation": "Execution successful, continuing to next step"
        }
    
    def _should_seek_user_feedback(self, state: MoveworksReasoningState) -> str:
        """Determine if user feedback is needed."""
        return "user_feedback" if state["needs_user_input"] else "finalize"
    
    def _should_continue_execution(self, state: MoveworksReasoningState) -> str:
        """Determine next action after user feedback."""
        user_feedback = state.get("user_feedback")

        if user_feedback == "retry":
            # User wants to retry - go back to planning
            return "replan"
        elif user_feedback == "stop":
            # User wants to stop - finalize
            return "finalize"
        elif user_feedback == "continue":
            # User wants to continue - go back to execution
            return "continue"
        else:
            # Default: finalize
            return "finalize"
    
    async def _finalize_response_node(self, state: MoveworksReasoningState) -> MoveworksReasoningState:
        """Generate final response from execution results."""
        successful_results = [r for r in state["execution_results"] if r["success"]]
        
        if successful_results:
            responses = [r["response"] for r in successful_results if r.get("response")]
            if responses:
                state["final_response"] = " ".join(responses)
            else:
                state["final_response"] = f"I've completed {len(successful_results)} step(s) successfully."
            
            # Collect user actions
            all_actions = []
            for result in successful_results:
                all_actions.extend(result.get("actions", []))
            state["user_actions"] = all_actions
        else:
            state["final_response"] = "I encountered issues processing your request. Please try again."
        
        # Add AI response to messages
        state["messages"].append(AIMessage(content=state["final_response"]))

        return state

    async def _generate_status_update(self, state: MoveworksReasoningState) -> str:
        """Generate status update for user."""
        completed_steps = len([r for r in state["execution_results"] if r["success"]])
        total_steps = len(state["execution_results"])

        if completed_steps == 0:
            return "I'm working on your request..."
        elif completed_steps == total_steps:
            return f"I've completed all {total_steps} steps of your request."
        else:
            return f"I've completed {completed_steps} out of {total_steps} steps so far."

    async def _check_confirmation_needed(self, state: MoveworksReasoningState) -> Dict[str, Any]:
        """Check if user confirmation is needed based on execution results and context."""
        execution_results = state["execution_results"]

        # Check for failed results
        failed_results = [r for r in execution_results if not r["success"]]
        if failed_results:
            error_types = [r.get("error", "unknown") for r in failed_results]
            return {
                "required": True,
                "message": f"I encountered {len(failed_results)} issue(s): {', '.join(error_types)}. How would you like me to proceed?",
                "options": ["retry", "continue", "stop"],
                "severity": "high"
            }

        # Check for high-impact actions
        all_actions = []
        for result in execution_results:
            all_actions.extend(result.get("actions", []))

        high_impact_actions = [
            action for action in all_actions
            if action.get("impact", "").lower() in ["high", "critical"]
        ]

        if high_impact_actions:
            action_descriptions = [action.get("description", "Unknown action") for action in high_impact_actions]
            return {
                "required": True,
                "message": f"I'm about to perform {len(high_impact_actions)} high-impact action(s): {', '.join(action_descriptions)}. Should I proceed?",
                "options": ["proceed", "review", "cancel"],
                "severity": "medium"
            }

        # Check for incomplete information
        user_query = state["user_query"]
        ambiguous_keywords = ["maybe", "possibly", "not sure", "think", "might"]
        if any(keyword in user_query.lower() for keyword in ambiguous_keywords):
            return {
                "required": True,
                "message": "Your request seems to have some uncertainty. Would you like me to proceed with my best interpretation or would you prefer to clarify?",
                "options": ["proceed", "clarify", "cancel"],
                "severity": "low"
            }

        # Check for sensitive operations
        sensitive_keywords = ["delete", "remove", "reset", "password", "permissions", "access"]
        if any(keyword in user_query.lower() for keyword in sensitive_keywords):
            return {
                "required": True,
                "message": "This operation involves sensitive actions. Are you sure you want to proceed?",
                "options": ["confirm", "cancel"],
                "severity": "high"
            }

        # No confirmation needed
        return {
            "required": False,
            "message": "",
            "options": [],
            "severity": "none"
        }


# Factory function
async def create_moveworks_reasoning_engine(config_path: str = "config/moveworks_config.yml") -> MoveworksThreeLoopEngine:
    """Create and initialize Moveworks three-loop reasoning engine."""
    config_loader = MoveworksConfigLoader(config_path)
    config_loader.load_config()
    
    engine = MoveworksThreeLoopEngine(config_loader)
    await engine.initialize()
    
    return engine

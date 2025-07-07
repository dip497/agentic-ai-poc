"""
LangGraph-based Reasoning Agent following Moveworks patterns.
Implements proper state management, conversational processes, and activities.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Annotated, TypedDict, Literal
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from ..llm.llm_factory import LLMFactory


class MoveworksState(TypedDict):
    """State schema for Moveworks reasoning agent following LangGraph patterns."""
    messages: Annotated[List[BaseMessage], add_messages]
    current_process: Optional[str]
    slots: Dict[str, Any]
    activities: List[Dict[str, Any]]
    reasoning_trace: List[Dict[str, Any]]
    user_id: str
    session_id: str
    user_attributes: Dict[str, Any]


@dataclass
class ReasoningStep:
    """Represents a single reasoning step in the process."""
    step: str
    action: str
    timestamp: str
    status: Literal["pending", "complete", "error"] = "pending"
    details: Optional[Dict[str, Any]] = None


@dataclass
class ConversationalProcess:
    """Represents a Moveworks-style conversational process."""
    name: str
    description: str
    activities: List[Dict[str, Any]]
    required_slots: List[str]
    triggers: List[str]


class MoveworksReasoningAgent:
    """
    LangGraph-based reasoning agent implementing Moveworks patterns.
    
    Features:
    - Proper state management with TypedDict
    - Conversational processes with activities and slots
    - Memory management with checkpointing
    - Real-time reasoning trace generation
    """
    
    def __init__(self):
        self.llm = LLMFactory.create_llm()
        self.checkpointer = MemorySaver()
        self.processes = self._initialize_processes()
        self.graph = self._build_graph()
    
    def _initialize_processes(self) -> Dict[str, ConversationalProcess]:
        """Initialize available conversational processes."""
        return {
            "password_reset": ConversationalProcess(
                name="password_reset",
                description="Help users reset their passwords",
                activities=[
                    {
                        "type": "slot_collection",
                        "name": "collect_system",
                        "slot": "target_system"
                    },
                    {
                        "type": "action_activity", 
                        "name": "reset_password",
                        "action": "password_reset_action"
                    }
                ],
                required_slots=["target_system"],
                triggers=["password reset", "forgot password", "can't login"]
            ),
            "pto_balance": ConversationalProcess(
                name="pto_balance",
                description="Check PTO balance for users",
                activities=[
                    {
                        "type": "action_activity",
                        "name": "get_pto_balance", 
                        "action": "pto_balance_action"
                    }
                ],
                required_slots=["user_email"],
                triggers=["pto balance", "time off", "vacation days"]
            ),
            "general_assistance": ConversationalProcess(
                name="general_assistance",
                description="Provide general IT and HR assistance",
                activities=[
                    {
                        "type": "content_activity",
                        "name": "provide_assistance",
                        "content": "I'll help you with your request."
                    }
                ],
                required_slots=[],
                triggers=["help", "assistance", "support"]
            )
        }
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph reasoning graph."""
        builder = StateGraph(MoveworksState)
        
        # Add nodes for each reasoning step
        builder.add_node("process_matching", self._process_matching_node)
        builder.add_node("slot_inference", self._slot_inference_node) 
        builder.add_node("activity_execution", self._activity_execution_node)
        builder.add_node("response_generation", self._response_generation_node)
        
        # Define the flow
        builder.add_edge(START, "process_matching")
        builder.add_edge("process_matching", "slot_inference")
        builder.add_edge("slot_inference", "activity_execution")
        builder.add_edge("activity_execution", "response_generation")
        builder.add_edge("response_generation", END)
        
        return builder.compile(checkpointer=self.checkpointer)
    
    async def _process_matching_node(self, state: MoveworksState) -> Dict[str, Any]:
        """Match user input to available conversational processes."""
        user_message = state["messages"][-1].content.lower()
        
        # Add reasoning step
        reasoning_step = ReasoningStep(
            step="process_matching",
            action="Analyzing user request to match with available processes",
            timestamp=datetime.now().isoformat()
        )
        
        # Simple keyword matching (in production, use embeddings/ML)
        matched_process = None
        confidence = 0.0
        
        for process_name, process in self.processes.items():
            for trigger in process.triggers:
                if trigger in user_message:
                    matched_process = process_name
                    confidence = 0.85  # Simulated confidence
                    break
            if matched_process:
                break
        
        if not matched_process:
            # Fallback to any available process if general_assistance doesn't exist
            if "general_assistance" in self.processes:
                matched_process = "general_assistance"
            elif self.processes:
                # Use the first available process as fallback
                matched_process = list(self.processes.keys())[0]
            else:
                # No processes available
                matched_process = None
            confidence = 0.60
        
        reasoning_step.status = "complete"
        reasoning_step.details = {
            "matched_process": matched_process,
            "confidence": confidence
        }
        
        return {
            "current_process": matched_process,
            "reasoning_trace": state["reasoning_trace"] + [reasoning_step.__dict__]
        }
    
    async def _slot_inference_node(self, state: MoveworksState) -> Dict[str, Any]:
        """Infer and collect required slots for the process."""
        process_name = state["current_process"]

        # Handle case where no process was matched
        if not process_name or process_name not in self.processes:
            reasoning_step = ReasoningStep(
                step="slot_inference",
                action="No valid process found, skipping slot collection",
                timestamp=datetime.now().isoformat(),
                status="complete"
            )
            return {
                "slots": {},
                "reasoning_trace": state["reasoning_trace"] + [reasoning_step.__dict__]
            }

        process = self.processes[process_name]
        
        reasoning_step = ReasoningStep(
            step="slot_inference", 
            action=f"Analyzing required information for {process_name}",
            timestamp=datetime.now().isoformat()
        )
        
        slots = {}
        
        # Auto-infer slots from user attributes and context
        if "user_email" in process.required_slots:
            slots["user_email"] = state["user_attributes"].get("email", "user@example.com")
        
        if "target_system" in process.required_slots:
            # Try to infer from message content
            user_message = state["messages"][-1].content.lower()
            if "laptop" in user_message or "computer" in user_message:
                slots["target_system"] = "laptop"
            else:
                slots["target_system"] = "general"
        
        reasoning_step.status = "complete"
        reasoning_step.details = {"inferred_slots": slots}
        
        return {
            "slots": {**state["slots"], **slots},
            "reasoning_trace": state["reasoning_trace"] + [reasoning_step.__dict__]
        }
    
    async def _activity_execution_node(self, state: MoveworksState) -> Dict[str, Any]:
        """Execute activities defined in the conversational process."""
        process_name = state["current_process"]

        # Handle case where no process was matched
        if not process_name or process_name not in self.processes:
            reasoning_step = ReasoningStep(
                step="activity_execution",
                action="No valid process found, skipping activity execution",
                timestamp=datetime.now().isoformat(),
                status="complete"
            )
            return {
                "activities": [],
                "reasoning_trace": state["reasoning_trace"] + [reasoning_step.__dict__]
            }

        process = self.processes[process_name]
        
        reasoning_step = ReasoningStep(
            step="activity_execution",
            action=f"Executing activities for {process_name}",
            timestamp=datetime.now().isoformat()
        )
        
        executed_activities = []
        
        for activity in process.activities:
            if activity["type"] == "action_activity":
                # Simulate action execution
                result = await self._execute_action(activity["action"], state["slots"])
                executed_activities.append({
                    "activity": activity["name"],
                    "type": "action",
                    "result": result
                })
            elif activity["type"] == "content_activity":
                executed_activities.append({
                    "activity": activity["name"],
                    "type": "content",
                    "content": activity["content"]
                })
        
        reasoning_step.status = "complete"
        reasoning_step.details = {"executed_activities": len(executed_activities)}
        
        return {
            "activities": executed_activities,
            "reasoning_trace": state["reasoning_trace"] + [reasoning_step.__dict__]
        }
    
    async def _response_generation_node(self, state: MoveworksState) -> Dict[str, Any]:
        """Generate final response based on executed activities."""
        reasoning_step = ReasoningStep(
            step="response_generation",
            action="Generating response based on executed activities",
            timestamp=datetime.now().isoformat()
        )
        
        # Generate response based on activities
        if state["activities"]:
            activity = state["activities"][0]
            if activity["type"] == "action":
                response_content = f"I've processed your request. {activity['result'].get('message', 'Task completed successfully.')}"
            else:
                response_content = activity.get("content", "I've processed your request.")
        else:
            response_content = "I've processed your request and will help you with this matter."
        
        reasoning_step.status = "complete"
        
        response_message = AIMessage(content=response_content)
        
        return {
            "messages": [response_message],
            "reasoning_trace": state["reasoning_trace"] + [reasoning_step.__dict__]
        }
    
    async def _execute_action(self, action_name: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific action with given slots."""
        # Simulate action execution
        await asyncio.sleep(0.1)  # Simulate processing time
        
        if action_name == "password_reset_action":
            return {
                "success": True,
                "message": f"Password reset initiated for {slots.get('target_system', 'system')}",
                "ticket_id": "PWD-12345"
            }
        elif action_name == "pto_balance_action":
            return {
                "success": True,
                "message": "PTO balance retrieved successfully",
                "vacation_days": 15,
                "sick_days": 8
            }
        else:
            return {
                "success": True,
                "message": "Action completed successfully"
            }
    
    async def process_message(
        self, 
        content: str, 
        user_id: str, 
        session_id: str, 
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message through the reasoning graph.
        
        Returns:
            Dict containing response and reasoning trace
        """
        if user_attributes is None:
            user_attributes = {"email": f"{user_id}@example.com"}
        
        # Create initial state
        initial_state = MoveworksState(
            messages=[HumanMessage(content=content)],
            current_process=None,
            slots={},
            activities=[],
            reasoning_trace=[],
            user_id=user_id,
            session_id=session_id,
            user_attributes=user_attributes
        )
        
        # Run the graph
        config = {"configurable": {"thread_id": session_id}}
        final_state = await self.graph.ainvoke(initial_state, config)
        
        # Extract response
        response_message = final_state["messages"][-1]
        
        return {
            "response": response_message.content,
            "reasoning_trace": final_state["reasoning_trace"],
            "process_used": final_state["current_process"],
            "slots_collected": final_state["slots"],
            "activities_executed": final_state["activities"]
        }

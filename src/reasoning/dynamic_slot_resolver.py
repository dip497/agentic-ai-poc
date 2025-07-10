"""
Dynamic Slot Resolver - Configurable slot-based HITL system.
Reads process definitions from Agent Studio and handles any process type dynamically.
"""

import uuid
import json
import logging
from typing import Dict, Any, List, Optional, Annotated, TypedDict
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from llm.llm_factory import LLMFactory
from agent_studio.database import agent_studio_db

logger = logging.getLogger(__name__)

# LangGraph State Schema for dynamic processes
class DynamicSlotState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    session_id: str
    process_id: str
    process_name: str
    collected_slots: Dict[str, Any]
    required_slots: List[str]
    optional_slots: List[str]

class DynamicSlotResolver:
    """
    Dynamic slot resolver that works with any Agent Studio process.
    Replaces hardcoded PTO handler with configurable system.
    """
    
    def __init__(self, shared_memory=None):
        self.llm_factory = LLMFactory()
        self.llm = None
        self.graph = None
        self.memory = shared_memory or MemorySaver()
        
    async def initialize(self):
        """Initialize the dynamic slot resolver."""
        try:
            # Initialize LLM using static method
            self.llm = self.llm_factory.get_default_llm()

            # Build the workflow
            self._build_workflow()

            logger.info("✅ Dynamic slot resolver initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize dynamic slot resolver: {e}")
            raise
    
    def _build_workflow(self):
        """Build the LangGraph workflow for dynamic slot resolution."""
        
        # Create the state graph
        workflow = StateGraph(DynamicSlotState)
        
        # Add nodes
        workflow.add_node("load_process", self._load_process_node)
        workflow.add_node("collect_slots", self._collect_slots_node)
        workflow.add_node("execute_process", self._execute_process_node)
        
        # Add edges
        workflow.add_edge(START, "load_process")
        workflow.add_edge("load_process", "collect_slots")
        workflow.add_conditional_edges(
            "collect_slots",
            self._should_execute_process,
            {
                "execute": "execute_process",
                "continue": END
            }
        )
        workflow.add_edge("execute_process", END)
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.memory)
    
    async def _load_process_node(self, state: DynamicSlotState):
        """Load process definition from Agent Studio."""
        try:
            # Get process from database
            process = await agent_studio_db.get_process(state["process_id"])
            
            if not process:
                state["messages"].append(AIMessage(
                    content="I couldn't find the process definition. Please try again."
                ))
                return state
            
            # Parse slots from process definition
            slots_data = process.get("slots", [])
            if isinstance(slots_data, str):
                slots_data = json.loads(slots_data)
            required_slots = []
            optional_slots = []
            
            for slot in slots_data:
                if slot.get("is_required", True):
                    required_slots.append(slot["name"])
                else:
                    optional_slots.append(slot["name"])
            
            # Update state
            state["required_slots"] = required_slots
            state["optional_slots"] = optional_slots
            state["collected_slots"] = {}
            
            # Generate system prompt based on process
            system_prompt = self._generate_system_prompt(process, slots_data)
            state["messages"] = [SystemMessage(content=system_prompt)] + state["messages"]
            
            logger.info(f"✅ Loaded process: {state['process_name']} with {len(required_slots)} required slots")
            
        except Exception as e:
            logger.error(f"❌ Failed to load process: {e}")
            state["messages"].append(AIMessage(
                content="I encountered an error loading the process. Please try again."
            ))
        
        return state
    
    def _generate_system_prompt(self, process: Dict[str, Any], slots_data: List[Dict[str, Any]]) -> str:
        """Generate dynamic system prompt based on process definition."""
        
        process_name = process.get("name", "Unknown Process")
        process_desc = process.get("description", "")
        
        # Build slot descriptions
        slot_descriptions = []
        for slot in slots_data:
            slot_name = slot.get("name", "")
            slot_desc = slot.get("description", "")
            is_required = slot.get("is_required", True)
            req_text = "Required" if is_required else "Optional"
            slot_descriptions.append(f"- {slot_name}: {slot_desc} ({req_text})")
        
        slots_text = "\n".join(slot_descriptions) if slot_descriptions else "No specific slots defined"
        
        return f"""You are helping with: {process_name}

{process_desc}

Required information to collect:
{slots_text}

Guidelines:
1. Ask for one piece of information at a time
2. Be helpful and guide the user through the process
3. Once you have all required information, proceed with execution
4. Be friendly and professional"""

    async def _collect_slots_node(self, state: DynamicSlotState):
        """Collect slots dynamically based on process definition."""
        try:
            # Get the latest user message
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            if not user_messages:
                return state
            
            latest_message = user_messages[-1].content
            
            # Simple slot extraction (can be enhanced with proper NLU)
            await self._extract_slots_from_message(state, latest_message)
            
            # Check what slots are still needed
            missing_required = [slot for slot in state["required_slots"] 
                              if slot not in state["collected_slots"]]
            
            if missing_required:
                # Ask for the next missing slot
                next_slot = missing_required[0]
                response = await self._ask_for_slot(state, next_slot)
                state["messages"].append(AIMessage(content=response))
            else:
                # All required slots collected
                state["messages"].append(AIMessage(
                    content="Great! I have all the information needed. Let me process your request..."
                ))
            
        except Exception as e:
            logger.error(f"❌ Failed to collect slots: {e}")
            state["messages"].append(AIMessage(
                content="I encountered an error collecting information. Please try again."
            ))
        
        return state
    
    async def _extract_slots_from_message(self, state: DynamicSlotState, message: str):
        """Extract slot values from user message (simplified implementation)."""
        
        # This is a simplified implementation
        # In production, this would use proper NLU and slot resolvers
        
        message_lower = message.lower()
        
        # Simple keyword-based extraction
        if "vacation" in message_lower or "holiday" in message_lower:
            state["collected_slots"]["pto_type"] = "vacation"
        elif "sick" in message_lower:
            state["collected_slots"]["pto_type"] = "sick"
        elif "personal" in message_lower:
            state["collected_slots"]["pto_type"] = "personal"
        
        # Date extraction (simplified)
        if "tomorrow" in message_lower:
            from datetime import datetime, timedelta
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            state["collected_slots"]["start_date"] = tomorrow
        
        # Duration extraction
        if "2 days" in message_lower:
            state["collected_slots"]["duration_days"] = 2
        elif "3 days" in message_lower:
            state["collected_slots"]["duration_days"] = 3
    
    async def _ask_for_slot(self, state: DynamicSlotState, slot_name: str) -> str:
        """Generate question for a specific slot."""
        
        # Load slot definition from process
        try:
            process = await agent_studio_db.get_process(state["process_id"])
            slots_data = process.get("slots", [])
            if isinstance(slots_data, str):
                slots_data = json.loads(slots_data)
            
            slot_def = next((s for s in slots_data if s.get("name") == slot_name), None)
            
            if slot_def:
                slot_desc = slot_def.get("description", "")
                return f"Could you please provide the {slot_name}? {slot_desc}"
            else:
                return f"Could you please provide the {slot_name}?"
                
        except Exception as e:
            logger.error(f"❌ Failed to get slot definition: {e}")
            return f"Could you please provide the {slot_name}?"
    
    def _should_execute_process(self, state: DynamicSlotState) -> str:
        """Determine if we should execute the process or continue collecting."""
        missing_required = [slot for slot in state["required_slots"] 
                          if slot not in state["collected_slots"]]
        
        return "execute" if not missing_required else "continue"
    
    async def _execute_process_node(self, state: DynamicSlotState):
        """Execute the process activities with collected slots."""
        try:
            # Load process activities
            process = await agent_studio_db.get_process(state["process_id"])
            activities = process.get("activities", [])
            if isinstance(activities, str):
                activities = json.loads(activities)
            
            # Execute activities (simplified implementation)
            result = await self._execute_activities(activities, state["collected_slots"])
            
            state["messages"].append(AIMessage(content=result))
            
        except Exception as e:
            logger.error(f"❌ Failed to execute process: {e}")
            state["messages"].append(AIMessage(
                content="I encountered an error processing your request. Please try again."
            ))
        
        return state
    
    async def _execute_activities(self, activities: List[Dict[str, Any]], slots: Dict[str, Any]) -> str:
        """Execute process activities with collected slot values."""
        
        # This is a simplified implementation
        # In production, this would execute actual HTTP actions, etc.
        
        results = []
        for activity in activities:
            activity_type = activity.get("type", "")
            activity_name = activity.get("name", "")
            
            if activity_type == "http_action":
                # Simulate HTTP action execution
                results.append(f"✅ Executed {activity_name}")
            elif activity_type == "content_action":
                # Generate content response
                template = activity.get("content_template", "")
                # Simple template substitution
                for slot_name, slot_value in slots.items():
                    template = template.replace(f"{{{slot_name}}}", str(slot_value))
                results.append(template)
        
        if results:
            return "\n".join(results)
        else:
            return f"✅ Successfully processed your request with: {', '.join(f'{k}={v}' for k, v in slots.items())}"
    
    async def handle_message(self, content: str, user_id: str, session_id: str, process_id: str, process_name: str) -> str:
        """Handle a message for dynamic slot resolution."""
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=content)],
            "user_id": user_id,
            "session_id": session_id,
            "process_id": process_id,
            "process_name": process_name,
            "collected_slots": {},
            "required_slots": [],
            "optional_slots": []
        }
        
        # Configuration for this session
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            # Run the graph
            result = await self.graph.ainvoke(initial_state, config=config)
            
            # Get the last AI message
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                return ai_messages[-1].content
            else:
                return "I'm ready to help you with your request!"
                
        except Exception as e:
            logger.error(f"❌ Failed to handle message: {e}")
            return "I encountered an error processing your request. Please try again."

# Global instance - will be initialized with shared memory in main.py
dynamic_slot_resolver = None

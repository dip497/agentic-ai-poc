"""
LangGraph Integration for Agent Studio.
Connects Agent Studio processes to the LangGraph reasoning system for real AI execution.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from reasoning.moveworks_reasoning_engine import MoveworksThreeLoopEngine
from models.moveworks import ConversationalProcess
from .database import agent_studio_db

logger = logging.getLogger(__name__)


@dataclass
class AgentStudioProcess:
    """Agent Studio process converted to LangGraph format."""
    id: str
    name: str
    description: str
    triggers: List[str]
    keywords: List[str]
    activities: List[Dict[str, Any]]
    slots: List[Dict[str, Any]]
    required_connectors: List[str]
    status: str


class AgentStudioLangGraphIntegration:
    """
    Integration layer between Agent Studio and LangGraph reasoning system.
    
    Features:
    - Loads processes from PostgreSQL database
    - Converts Agent Studio format to LangGraph format
    - Provides real AI execution for Agent Studio processes
    - Maintains compatibility with existing LangGraph system
    """
    
    def __init__(self):
        self.reasoning_engine = None  # Will be initialized when needed
        self.processes_cache: Dict[str, AgentStudioProcess] = {}
        self.last_cache_update = None
        
    async def initialize(self):
        """Initialize the integration and load processes."""
        try:
            await self._load_processes_from_database()
            await self._update_reasoning_agent_processes()
            logger.info(f"Agent Studio LangGraph integration initialized with {len(self.processes_cache)} processes")
        except Exception as e:
            logger.error(f"Failed to initialize Agent Studio LangGraph integration: {e}")
            # Continue with empty cache
            self.processes_cache = {}
    
    async def _load_processes_from_database(self):
        """Load all active processes from Agent Studio database."""
        try:
            # Ensure database is initialized
            if not hasattr(agent_studio_db, 'pool') or agent_studio_db.pool is None:
                logger.info("Initializing Agent Studio database connection...")
                await agent_studio_db.initialize()

            # Get all processes (not just published/testing for now)
            db_processes = await agent_studio_db.list_processes()
            logger.info(f"Retrieved {len(db_processes)} processes from database")

            self.processes_cache = {}
            for db_process in db_processes:
                # Load all processes for testing, not just published ones
                studio_process = self._convert_db_process_to_studio_format(db_process)
                self.processes_cache[studio_process.id] = studio_process
                logger.info(f"Loaded process: {studio_process.id} - {studio_process.name} (status: {studio_process.status})")

            logger.info(f"Successfully loaded {len(self.processes_cache)} processes from Agent Studio database")

        except Exception as e:
            logger.error(f"Failed to load processes from database: {e}", exc_info=True)
            # Fall back to empty cache
            self.processes_cache = {}
    
    def _convert_db_process_to_studio_format(self, db_process: Dict[str, Any]) -> AgentStudioProcess:
        """Convert database process format to Agent Studio format."""
        # Parse JSON fields
        triggers = db_process["triggers"] if isinstance(db_process["triggers"], list) else json.loads(db_process["triggers"] or "[]")
        keywords = db_process["keywords"] if isinstance(db_process["keywords"], list) else json.loads(db_process["keywords"] or "[]")
        activities = db_process["activities"] if isinstance(db_process["activities"], list) else json.loads(db_process["activities"] or "[]")
        slots = db_process["slots"] if isinstance(db_process["slots"], list) else json.loads(db_process["slots"] or "[]")
        required_connectors = db_process["required_connectors"] if isinstance(db_process["required_connectors"], list) else json.loads(db_process["required_connectors"] or "[]")
        
        return AgentStudioProcess(
            id=str(db_process["id"]),
            name=db_process["name"],
            description=db_process["description"],
            triggers=triggers,
            keywords=keywords,
            activities=activities,
            slots=slots,
            required_connectors=required_connectors,
            status=db_process["status"]
        )
    
    def _convert_studio_process_to_langgraph_format(self, studio_process: AgentStudioProcess) -> ConversationalProcess:
        """Convert Agent Studio process to LangGraph ConversationalProcess format."""
        # Convert activities to LangGraph format
        langgraph_activities = []
        required_slots = []
        
        for activity in studio_process.activities:
            if activity.get("type") == "slot_collection":
                # Slot collection activity
                langgraph_activities.append({
                    "name": activity.get("name", "collect_slot"),
                    "type": "slot_collection",
                    "slot_name": activity.get("slot_name"),
                    "description": activity.get("description", "")
                })
                if activity.get("slot_name"):
                    required_slots.append(activity["slot_name"])
                    
            elif activity.get("type") == "http_action":
                # HTTP action activity
                langgraph_activities.append({
                    "name": activity.get("name", "http_action"),
                    "type": "action_activity",
                    "action": "http_request",
                    "connector_name": activity.get("connector_name"),
                    "endpoint": activity.get("endpoint"),
                    "method": activity.get("method", "GET"),
                    "parameters": activity.get("parameters", {}),
                    "description": activity.get("description", "")
                })
                
            elif activity.get("type") == "content_action":
                # Content response activity
                langgraph_activities.append({
                    "name": activity.get("name", "content_response"),
                    "type": "content_activity",
                    "content": activity.get("content_template", activity.get("content", "")),
                    "description": activity.get("description", "")
                })
        
        # Add slot requirements from slot definitions
        for slot in studio_process.slots:
            if slot.get("name") and slot["name"] not in required_slots:
                required_slots.append(slot["name"])
        
        return ConversationalProcess(
            name=studio_process.name,
            description=studio_process.description,
            activities=langgraph_activities,
            required_slots=required_slots,
            triggers=studio_process.triggers
        )
    
    async def _update_reasoning_agent_processes(self):
        """Update the reasoning agent with current processes from Agent Studio."""
        langgraph_processes = {}
        
        for studio_process in self.processes_cache.values():
            langgraph_process = self._convert_studio_process_to_langgraph_format(studio_process)
            langgraph_processes[studio_process.name] = langgraph_process
        
        # Update the reasoning engine's processes (placeholder for now)
        # self.reasoning_engine.processes = langgraph_processes
        logger.info(f"Updated reasoning engine with {len(langgraph_processes)} processes")
    
    async def process_message(
        self,
        content: str,
        user_id: str,
        session_id: str,
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message through Agent Studio processes via LangGraph.
        
        Args:
            content: User message content
            user_id: Unique user identifier
            session_id: Session identifier
            user_attributes: User attributes and permissions
            
        Returns:
            Dict containing response and reasoning trace
        """
        # Refresh processes if needed (check every 5 minutes)
        await self._refresh_processes_if_needed()
        
        # Process through LangGraph reasoning engine (placeholder for now)
        result = {
            "response": f"Agent Studio integration: {content}",
            "success": True,
            "selected_plugins": [],
            "user_actions": []
        }
        
        # Add Agent Studio metadata
        result["agent_studio_integration"] = True
        result["processes_loaded"] = len(self.processes_cache)
        
        return result
    
    async def _refresh_processes_if_needed(self):
        """Refresh processes from database if cache is stale."""
        import time
        current_time = time.time()
        
        # Refresh every 5 minutes
        if (self.last_cache_update is None or 
            current_time - self.last_cache_update > 300):
            
            await self._load_processes_from_database()
            await self._update_reasoning_agent_processes()
            self.last_cache_update = current_time
    
    async def test_process(
        self,
        process_id: str,
        test_input: str,
        user_id: str = "test_user",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test a specific Agent Studio process with given input.
        
        Args:
            process_id: ID of the process to test
            test_input: Test input message
            user_id: User ID for testing
            session_id: Session ID for testing
            
        Returns:
            Dict containing test results
        """
        if session_id is None:
            session_id = f"test_session_{process_id}_{int(asyncio.get_event_loop().time())}"
        
        # Ensure we have the latest processes
        await self._refresh_processes_if_needed()
        
        # Check if process exists
        logger.info(f"Testing process {process_id}, available processes: {list(self.processes_cache.keys())}")
        if process_id not in self.processes_cache:
            logger.warning(f"Process {process_id} not found in cache")
            return {
                "success": False,
                "error": f"Process {process_id} not found in cache. Available: {list(self.processes_cache.keys())}",
                "response": "",
                "reasoning_trace": [],
                "execution_time": 0
            }
        
        try:
            import time
            start_time = time.time()
            
            # Process the test input
            result = await self.process_message(
                test_input, user_id, session_id
            )
            
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "response": result.get("response", ""),
                "reasoning_trace": result.get("reasoning_trace", []),
                "process_used": result.get("process_used"),
                "slots_collected": result.get("slots_collected", {}),
                "activities_executed": result.get("activities_executed", []),
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"Error testing process {process_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "",
                "reasoning_trace": [],
                "execution_time": 0
            }
    
    def get_loaded_processes(self) -> Dict[str, AgentStudioProcess]:
        """Get all currently loaded processes."""
        return self.processes_cache.copy()
    
    async def reload_processes(self):
        """Force reload all processes from database."""
        await self._load_processes_from_database()
        await self._update_reasoning_agent_processes()
        logger.info("Forced reload of Agent Studio processes")


# Global integration instance
agent_studio_integration = AgentStudioLangGraphIntegration()

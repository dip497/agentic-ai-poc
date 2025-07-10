"""
LangGraph Integration for Agent Studio.
Connects Agent Studio processes to the LangGraph reasoning system for real AI execution.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from src.reasoning.moveworks_smart_reasoning_agent import MoveworksSmartReasoningAgent
from src.models.moveworks import ConversationalProcess, Plugin
from src.agent_studio.database import agent_studio_db

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
        logger.info("Starting Agent Studio LangGraph integration initialization", extra={
            "component": "agent_studio_integration",
            "action": "initialize",
            "stage": "begin"
        })

        try:
            # Load processes from database
            logger.info("Loading processes from Agent Studio database", extra={
                "component": "agent_studio_integration",
                "action": "load_processes",
                "stage": "begin"
            })
            await self._load_processes_from_database()
            logger.info("Processes loaded from database", extra={
                "component": "agent_studio_integration",
                "action": "load_processes",
                "stage": "complete",
                "process_count": len(self.processes_cache)
            })

            # Initialize reasoning engine first
            if not self.reasoning_engine:
                logger.info("Creating reasoning engine for Agent Studio integration", extra={
                    "component": "agent_studio_integration",
                    "action": "create_reasoning_engine",
                    "engine_type": "MoveworksSmartReasoningAgent"
                })
                self.reasoning_engine = MoveworksSmartReasoningAgent()

                logger.info("Initializing reasoning engine", extra={
                    "component": "agent_studio_integration",
                    "action": "initialize_reasoning_engine"
                })
                await self.reasoning_engine.initialize()
                logger.info("Reasoning engine initialized successfully", extra={
                    "component": "agent_studio_integration",
                    "action": "initialize_reasoning_engine",
                    "status": "success"
                })

            # Now register Agent Studio processes as plugins
            logger.info("Registering Agent Studio processes as Moveworks plugins", extra={
                "component": "agent_studio_integration",
                "action": "register_plugins",
                "stage": "begin"
            })
            await self._update_reasoning_agent_processes()

            logger.info("Agent Studio LangGraph integration initialization completed", extra={
                "component": "agent_studio_integration",
                "action": "initialize",
                "stage": "complete",
                "status": "success",
                "process_count": len(self.processes_cache)
            })
        except Exception as e:
            logger.error("Failed to initialize Agent Studio LangGraph integration", extra={
                "component": "agent_studio_integration",
                "action": "initialize",
                "stage": "failed",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Continue with empty cache
            self.processes_cache = {}
    
    async def _load_processes_from_database(self):
        """Load all active processes from Agent Studio database."""
        try:
            # Ensure database is initialized
            if not hasattr(agent_studio_db, 'pool') or agent_studio_db.pool is None:
                logger.info("Database connection not found, initializing", extra={
                    "component": "agent_studio_integration",
                    "action": "initialize_database",
                    "database": "agent_studio"
                })
                await agent_studio_db.initialize()

            # Get all processes (not just published/testing for now)
            logger.info("Querying database for processes", extra={
                "component": "agent_studio_integration",
                "action": "query_processes",
                "table": "processes"
            })
            db_processes = await agent_studio_db.list_processes()
            logger.info("Database query completed", extra={
                "component": "agent_studio_integration",
                "action": "query_processes",
                "status": "success",
                "record_count": len(db_processes)
            })

            self.processes_cache = {}
            for db_process in db_processes:
                try:
                    # Load all processes for testing, not just published ones
                    studio_process = self._convert_db_process_to_studio_format(db_process)
                    self.processes_cache[studio_process.id] = studio_process
                    logger.debug("Process loaded successfully", extra={
                        "component": "agent_studio_integration",
                        "action": "load_process",
                        "process_id": studio_process.id,
                        "process_name": studio_process.name,
                        "status": studio_process.status,
                        "keywords": studio_process.keywords,
                        "triggers": studio_process.triggers
                    })
                except Exception as process_error:
                    logger.warning("Failed to convert process format", extra={
                        "component": "agent_studio_integration",
                        "action": "convert_process",
                        "status": "error",
                        "error": str(process_error),
                        "process_id": db_process.get("id", "unknown")
                    })

            logger.info("Process loading completed successfully", extra={
                "component": "agent_studio_integration",
                "action": "load_processes",
                "status": "success",
                "total_records": len(db_processes),
                "loaded_processes": len(self.processes_cache)
            })

        except Exception as e:
            logger.error("Failed to load processes from database", extra={
                "component": "agent_studio_integration",
                "action": "load_processes",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            })
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
            title=studio_process.name,  # Map name to title
            description=studio_process.description,
            trigger_utterances=studio_process.triggers,  # Map triggers to trigger_utterances
            activities=langgraph_activities,
            slots=[]  # Will be populated from slot definitions if needed
        )

    def _convert_studio_process_to_moveworks_plugin(self, studio_process: AgentStudioProcess) -> Plugin:
        """Convert Agent Studio process to proper Moveworks Plugin format."""
        # Convert to ConversationalProcess first
        conversational_process = self._convert_studio_process_to_langgraph_format(studio_process)

        # Create Moveworks Plugin using Agent Studio data directly
        # Empty capabilities/domains will be derived by the Plugin Selector
        return Plugin(
            name=studio_process.name,
            description=studio_process.description,
            conversational_processes=[conversational_process],
            capabilities=[],  # Will be derived by Plugin Selector
            domain_compatibility=[],  # Will be derived by Plugin Selector
            positive_examples=studio_process.triggers,
            confidence_threshold=0.7,
            success_rate=0.85
        )


    
    async def _update_reasoning_agent_processes(self):
        """Update the reasoning agent with current processes from Agent Studio as proper Moveworks Plugins."""
        moveworks_plugins = []

        for studio_process in self.processes_cache.values():
            # Convert Agent Studio process to proper Moveworks Plugin
            plugin = self._convert_studio_process_to_moveworks_plugin(studio_process)
            moveworks_plugins.append(plugin)

        # Register plugins with the reasoning engine's plugin selector
        logger.info("Starting plugin registration with reasoning engine", extra={
            "component": "agent_studio_integration",
            "action": "register_plugins",
            "plugin_count": len(moveworks_plugins)
        })

        if self.reasoning_engine and hasattr(self.reasoning_engine, 'three_loop_engine'):
            if (self.reasoning_engine.three_loop_engine and
                hasattr(self.reasoning_engine.three_loop_engine, 'plugin_selector')):

                plugin_selector = self.reasoning_engine.three_loop_engine.plugin_selector
                logger.info("Plugin selector found, registering Agent Studio plugins", extra={
                    "component": "agent_studio_integration",
                    "action": "register_plugins",
                    "selector_type": type(plugin_selector).__name__
                })

                # Add Agent Studio plugins and let the selector derive capabilities/domains
                for plugin in moveworks_plugins:
                    logger.debug("Processing plugin for registration", extra={
                        "component": "agent_studio_integration",
                        "action": "process_plugin",
                        "plugin_name": plugin.name,
                        "plugin_description": plugin.description
                    })

                    # Let the plugin selector derive capabilities and domains if empty
                    if not plugin.capabilities:
                        logger.debug("Deriving capabilities for plugin", extra={
                            "component": "agent_studio_integration",
                            "action": "derive_capabilities",
                            "plugin_name": plugin.name
                        })
                        plugin.capabilities = await plugin_selector._derive_capabilities(plugin)
                        logger.debug("Capabilities derived", extra={
                            "component": "agent_studio_integration",
                            "action": "derive_capabilities",
                            "plugin_name": plugin.name,
                            "capabilities": plugin.capabilities
                        })

                    if not plugin.domain_compatibility:
                        logger.debug("Deriving domain compatibility for plugin", extra={
                            "component": "agent_studio_integration",
                            "action": "derive_domains",
                            "plugin_name": plugin.name
                        })
                        plugin.domain_compatibility = await plugin_selector._derive_domain_compatibility(plugin)
                        logger.debug("Domain compatibility derived", extra={
                            "component": "agent_studio_integration",
                            "action": "derive_domains",
                            "plugin_name": plugin.name,
                            "domains": plugin.domain_compatibility
                        })

                    if not plugin.positive_examples:
                        plugin.positive_examples = plugin_selector._extract_trigger_utterances(plugin)

                    # Register the enhanced plugin
                    plugin_selector.plugins[plugin.name] = plugin
                    logger.info("Plugin registered successfully", extra={
                        "component": "agent_studio_integration",
                        "action": "register_plugin",
                        "plugin_name": plugin.name,
                        "capabilities": plugin.capabilities,
                        "domains": plugin.domain_compatibility,
                        "examples_count": len(plugin.positive_examples)
                    })

                logger.info("All Agent Studio plugins registered successfully", extra={
                    "component": "agent_studio_integration",
                    "action": "register_plugins",
                    "status": "success",
                    "registered_count": len(moveworks_plugins),
                    "total_plugins_in_selector": len(plugin_selector.plugins)
                })
            else:
                logger.warning("Three-loop engine or plugin selector not available", extra={
                    "component": "agent_studio_integration",
                    "action": "register_plugins",
                    "status": "warning",
                    "has_three_loop": bool(self.reasoning_engine.three_loop_engine),
                    "has_plugin_selector": hasattr(self.reasoning_engine.three_loop_engine, 'plugin_selector') if self.reasoning_engine.three_loop_engine else False
                })
        else:
            logger.warning("Reasoning engine not available for plugin registration", extra={
                "component": "agent_studio_integration",
                "action": "register_plugins",
                "status": "warning",
                "has_reasoning_engine": bool(self.reasoning_engine),
                "prepared_plugins": len(moveworks_plugins)
            })
    
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
        logger.info("Processing message through Agent Studio integration", extra={
            "component": "agent_studio_integration",
            "action": "process_message",
            "user_id": user_id,
            "session_id": session_id,
            "message_length": len(content),
            "has_user_attributes": bool(user_attributes),
            "cached_processes": len(self.processes_cache)
        })

        # Refresh processes if needed (check every 5 minutes)
        logger.debug("Checking if process refresh is needed", extra={
            "component": "agent_studio_integration",
            "action": "check_refresh"
        })
        await self._refresh_processes_if_needed()

        # Ensure reasoning engine is initialized
        if not self.reasoning_engine:
            logger.warning("Reasoning engine not initialized, initializing now", extra={
                "component": "agent_studio_integration",
                "action": "initialize_reasoning_engine",
                "status": "warning"
            })
            self.reasoning_engine = MoveworksSmartReasoningAgent()
            await self.reasoning_engine.initialize()
            await self._update_reasoning_agent_processes()

        # Use proper Moveworks reasoning engine with Agent Studio plugins registered
        try:
            user_context = user_attributes or {}
            user_context.update({
                "user_id": user_id,
                "session_id": session_id,
                "available_processes": list(self.processes_cache.keys())
            })

            reasoning_result = await self.reasoning_engine.process_request(
                user_query=content,
                user_context=user_context,
                conversation_id=session_id
            )

            result = {
                "response": reasoning_result.response,
                "success": reasoning_result.success,
                "selected_plugins": reasoning_result.selected_plugins,
                "user_actions": reasoning_result.user_actions,
                "reasoning_trace": [
                    {
                        "step": "reasoning_engine",
                        "action": "process_request",
                        "status": "completed" if reasoning_result.success else "failed",
                        "timestamp": reasoning_result.timestamp.isoformat(),
                        "details": reasoning_result.execution_summary
                    }
                ]
            }

        except Exception as e:
            logger.error(f"Error in reasoning engine: {e}")
            result = {
                "response": f"I encountered an error processing your request: {str(e)}",
                "success": False,
                "selected_plugins": [],
                "user_actions": [],
                "reasoning_trace": [
                    {
                        "step": "reasoning_engine",
                        "action": "process_request",
                        "status": "failed",
                        "timestamp": datetime.now().isoformat(),
                        "details": {"error": str(e)}
                    }
                ]
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

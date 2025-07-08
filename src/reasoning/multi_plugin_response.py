"""
Moveworks Multi-Plugin Response System.
Implements parallel plugin execution and response combination with improved UX.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, field

from .plugin_selection_engine import PluginSelectionResult
from models.moveworks import Plugin, ConversationalProcess
from config.loader import MoveworksConfigLoader
from llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


@dataclass
class PluginExecutionResult:
    """Result of executing a single plugin."""
    plugin_name: str
    process_name: str
    success: bool
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    confidence: float = 0.0
    user_facing_message: str = ""
    action_buttons: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MultiPluginResponse:
    """Combined response from multiple plugins."""
    primary_response: str
    plugin_results: List[PluginExecutionResult] = field(default_factory=list)
    combined_actions: List[Dict[str, Any]] = field(default_factory=list)
    response_type: str = "multi_plugin"  # single_plugin, multi_plugin, no_plugin
    confidence: float = 0.0
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MultiPluginResponseEngine:
    """
    Handles parallel plugin execution and response combination.
    Implements Moveworks multi-plugin response patterns with improved button UX.
    """
    
    def __init__(self, config_loader: MoveworksConfigLoader):
        self.config_loader = config_loader
        self.llm = None
        self.max_parallel_plugins = 3
        self.response_timeout = 30.0  # seconds

    async def initialize(self):
        """Initialize the multi-plugin response engine."""
        # Initialize LLM for response combination
        llm_factory = LLMFactory()
        self.llm = llm_factory.create_llm("gemini", model="gemini-1.5-flash")

        logger.info("Multi-plugin response engine initialized")
    
    async def execute_plugins(
        self,
        selected_plugins: List[PluginSelectionResult],
        user_query: str,
        user_context: Dict[str, Any],
        execution_context: Dict[str, Any] = None
    ) -> MultiPluginResponse:
        """
        Execute multiple plugins in parallel and combine their responses.
        
        Args:
            selected_plugins: List of selected plugins to execute
            user_query: Original user query
            user_context: User attributes and context
            execution_context: Additional execution context
            
        Returns:
            MultiPluginResponse with combined results
        """
        if not selected_plugins:
            return MultiPluginResponse(
                primary_response="I couldn't find any relevant plugins to help with your request.",
                response_type="no_plugin",
                confidence=0.0
            )
        
        execution_context = execution_context or {}
        start_time = datetime.now()
        
        # Limit number of plugins to execute in parallel
        plugins_to_execute = selected_plugins[:self.max_parallel_plugins]
        
        logger.info(f"Executing {len(plugins_to_execute)} plugins in parallel for query: {user_query[:50]}...")
        
        # Execute plugins in parallel
        execution_tasks = []
        for plugin_result in plugins_to_execute:
            task = asyncio.create_task(
                self._execute_single_plugin(
                    plugin_result,
                    user_query,
                    user_context,
                    execution_context
                )
            )
            execution_tasks.append(task)
        
        # Wait for all plugins to complete or timeout
        try:
            plugin_results = await asyncio.wait_for(
                asyncio.gather(*execution_tasks, return_exceptions=True),
                timeout=self.response_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Plugin execution timeout after {self.response_timeout}s")
            plugin_results = [
                PluginExecutionResult(
                    plugin_name="timeout",
                    process_name="timeout",
                    success=False,
                    error_message="Plugin execution timed out"
                )
            ]
        
        # Filter successful results and handle exceptions
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(plugin_results):
            if isinstance(result, Exception):
                failed_result = PluginExecutionResult(
                    plugin_name=plugins_to_execute[i].plugin.name,
                    process_name=plugins_to_execute[i].matching_process.title if plugins_to_execute[i].matching_process else "unknown",
                    success=False,
                    error_message=str(result)
                )
                failed_results.append(failed_result)
            elif isinstance(result, PluginExecutionResult):
                if result.success:
                    successful_results.append(result)
                else:
                    failed_results.append(result)
        
        # Combine responses
        combined_response = await self._combine_plugin_responses(
            successful_results,
            failed_results,
            user_query,
            user_context
        )
        
        # Calculate execution summary
        execution_time = (datetime.now() - start_time).total_seconds()
        combined_response.execution_summary = {
            "total_plugins_attempted": len(plugins_to_execute),
            "successful_plugins": len(successful_results),
            "failed_plugins": len(failed_results),
            "execution_time": execution_time,
            "timeout_occurred": execution_time >= self.response_timeout
        }
        
        logger.info(f"Multi-plugin execution completed: {len(successful_results)} successful, {len(failed_results)} failed")
        
        return combined_response
    
    async def _execute_single_plugin(
        self,
        plugin_result: PluginSelectionResult,
        user_query: str,
        user_context: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> PluginExecutionResult:
        """Execute a single plugin and return the result."""
        start_time = datetime.now()
        plugin = plugin_result.plugin
        process = plugin_result.matching_process
        
        try:
            logger.debug(f"Executing plugin: {plugin.name}")

            # Execute the actual plugin process
            result_data = await self._execute_plugin_process(
                plugin, process, user_query, user_context, execution_context
            )
            
            # Generate user-facing message
            user_message = await self._generate_user_message(
                plugin, process, result_data, user_query
            )
            
            # Generate action buttons
            action_buttons = await self._generate_action_buttons(
                plugin, process, result_data
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return PluginExecutionResult(
                plugin_name=plugin.name,
                process_name=process.title if process else "unknown",
                success=True,
                result_data=result_data,
                execution_time=execution_time,
                confidence=plugin_result.confidence,
                user_facing_message=user_message,
                action_buttons=action_buttons
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error executing plugin {plugin.name}: {e}")
            
            return PluginExecutionResult(
                plugin_name=plugin.name,
                process_name=process.title if process else "unknown",
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                confidence=plugin_result.confidence
            )
    
    async def _execute_plugin_process(
        self,
        plugin: Plugin,
        process: Optional[ConversationalProcess],
        user_query: str,
        user_context: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the actual plugin process using our orchestration system."""
        from ..orchestration.activity_orchestrator import ActivityOrchestrator
        from ..llm.slot_inference import SlotInferenceEngine
        from ..models.moveworks import ConversationContext

        if not process:
            return {
                "status": "error",
                "message": "No process found for plugin execution",
                "data": {}
            }

        try:
            # Create conversation context
            context = ConversationContext(
                conversation_id=execution_context.get("conversation_id", "temp"),
                user_id=user_context.get("user_id", "unknown"),
                user_attributes=user_context,
                conversation_history=[{
                    "role": "user",
                    "content": user_query,
                    "timestamp": datetime.now().isoformat()
                }]
            )

            # Initialize slot inference engine
            slot_engine = SlotInferenceEngine(self.config_manager)
            await slot_engine.initialize()

            # Initialize activity orchestrator
            orchestrator = ActivityOrchestrator(self.config_manager)
            await orchestrator.initialize()

            # Extract and resolve slots from user query
            slot_values = {}
            for slot in process.slots:
                logger.debug(f"Resolving slot: {slot.name}")

                # Use slot inference to extract value from user query
                inferred_value = await slot_engine.infer_slot_value(
                    slot, user_query, context
                )

                if inferred_value:
                    slot_values[slot.name] = {
                        "value": inferred_value,
                        "confidence": 0.8,  # Default confidence
                        "source": "inference"
                    }
                    logger.debug(f"Resolved slot {slot.name}: {inferred_value}")
                else:
                    logger.debug(f"Could not resolve slot {slot.name}")

            # Prepare process data
            process_data = {
                "slots": slot_values,
                "user_query": user_query,
                "user_context": user_context,
                "execution_context": execution_context
            }

            # Execute the process activities
            logger.info(f"Executing process: {process.title}")
            activity_results = await orchestrator.execute_process_activities(
                process, context, process_data
            )

            # Combine results
            execution_result = {
                "status": "success",
                "message": f"Successfully executed {process.title}",
                "data": {
                    "plugin_name": plugin.name,
                    "process_name": process.title,
                    "slot_values": slot_values,
                    "activity_results": [
                        {
                            "activity_type": result.activity_type,
                            "success": result.success,
                            "output_data": result.output_data,
                            "error_message": result.error_message
                        }
                        for result in activity_results
                    ],
                    "final_output": activity_results[-1].output_data if activity_results else {}
                }
            }

            # Check if any activity failed
            failed_activities = [r for r in activity_results if not r.success]
            if failed_activities:
                execution_result["status"] = "partial_success"
                execution_result["message"] = f"Process completed with {len(failed_activities)} failed activities"
                execution_result["data"]["failed_activities"] = [
                    {
                        "activity_type": r.activity_type,
                        "error_message": r.error_message
                    }
                    for r in failed_activities
                ]

            return execution_result

        except Exception as e:
            logger.error(f"Error executing plugin process {process.title}: {e}")
            return {
                "status": "error",
                "message": f"Plugin execution failed: {str(e)}",
                "data": {
                    "plugin_name": plugin.name,
                    "process_name": process.title if process else "unknown",
                    "error": str(e)
                }
            }
    
    async def _generate_user_message(
        self,
        plugin: Plugin,
        process: Optional[ConversationalProcess],
        result_data: Dict[str, Any],
        user_query: str
    ) -> str:
        """Generate user-facing message for plugin result."""
        # Check if process has content activities with templates
        if process:
            for activity in process.activities:
                if activity.activity_type == "content":
                    # Use the content template from the process
                    if hasattr(activity, 'content_text') and activity.content_text:
                        # Simple template substitution (can be enhanced)
                        message = activity.content_text
                        # Replace placeholders with actual data
                        for key, value in result_data.get("data", {}).items():
                            message = message.replace(f"{{{{{key}}}}}", str(value))
                        return message
        
        # Fallback to generic message
        return f"I've processed your request using {plugin.name}. {result_data.get('message', '')}"
    
    async def _generate_action_buttons(
        self,
        plugin: Plugin,
        process: Optional[ConversationalProcess],
        result_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate action buttons for plugin result."""
        buttons = []
        
        # Generate action buttons from plugin capabilities
        for capability in plugin.capabilities[:3]:  # Limit to top 3 capabilities
            action_name = capability.replace(" ", "_").lower()
            button_text = capability.title()
            buttons.append({
                "text": button_text,
                "action": action_name,
                "style": "primary" if len(buttons) == 0 else "secondary"
            })

        # Add help action if no capabilities
        if not buttons:
            buttons.append({"text": "Get Help", "action": "get_help", "style": "tertiary"})
        
        return buttons

    async def _combine_plugin_responses(
        self,
        successful_results: List[PluginExecutionResult],
        failed_results: List[PluginExecutionResult],
        user_query: str,
        user_context: Dict[str, Any]
    ) -> MultiPluginResponse:
        """Combine multiple plugin responses into a coherent response."""

        if not successful_results and not failed_results:
            return MultiPluginResponse(
                primary_response="No plugins were executed successfully.",
                response_type="no_plugin",
                confidence=0.0
            )

        # Determine response type
        if len(successful_results) == 1 and not failed_results:
            response_type = "single_plugin"
        elif len(successful_results) > 1:
            response_type = "multi_plugin"
        else:
            response_type = "partial_failure"

        # Generate primary response using LLM
        primary_response = await self._generate_combined_response(
            successful_results, failed_results, user_query, user_context
        )

        # Combine action buttons from all successful plugins
        combined_actions = []
        for result in successful_results:
            for button in result.action_buttons:
                # Add plugin context to button
                button_with_context = button.copy()
                button_with_context["plugin_name"] = result.plugin_name
                button_with_context["process_name"] = result.process_name
                combined_actions.append(button_with_context)

        # Calculate overall confidence
        if successful_results:
            avg_confidence = sum(r.confidence for r in successful_results) / len(successful_results)
            # Reduce confidence if there were failures
            if failed_results:
                failure_penalty = len(failed_results) / (len(successful_results) + len(failed_results))
                avg_confidence *= (1 - failure_penalty * 0.3)
        else:
            avg_confidence = 0.0

        return MultiPluginResponse(
            primary_response=primary_response,
            plugin_results=successful_results + failed_results,
            combined_actions=combined_actions,
            response_type=response_type,
            confidence=avg_confidence
        )

    async def _generate_combined_response(
        self,
        successful_results: List[PluginExecutionResult],
        failed_results: List[PluginExecutionResult],
        user_query: str,
        user_context: Dict[str, Any]
    ) -> str:
        """Generate a combined response using LLM."""

        if not successful_results and not failed_results:
            return "I wasn't able to process your request with any available plugins."

        # For single successful plugin, use its message
        if len(successful_results) == 1 and not failed_results:
            return successful_results[0].user_facing_message

        # For multiple plugins or mixed results, use LLM to combine
        context_parts = [
            f"User Query: {user_query}",
            f"User Department: {user_context.get('department', 'Unknown')}",
            f"User Role: {user_context.get('role', 'Unknown')}"
        ]

        if successful_results:
            context_parts.append("Successful Plugin Results:")
            for result in successful_results:
                context_parts.append(f"- {result.plugin_name}: {result.user_facing_message}")

        if failed_results:
            context_parts.append("Failed Plugin Results:")
            for result in failed_results:
                context_parts.append(f"- {result.plugin_name}: {result.error_message}")

        context = "\n".join(context_parts)

        system_prompt = """You are an enterprise AI assistant that combines results from multiple plugins.

Create a coherent, professional response that:
1. Acknowledges what was accomplished successfully
2. Mentions any failures briefly and professionally
3. Provides clear next steps or actions for the user
4. Maintains a helpful and positive tone
5. Is concise but informative

Do not repeat technical details or error messages verbatim. Focus on what the user needs to know."""

        try:
            prompt = f"{system_prompt}\n\n{context}"
            response = await self.llm.ainvoke(prompt)
            return response.strip()

        except Exception as e:
            logger.error(f"Error generating combined response: {e}")

            # Fallback to simple combination
            if successful_results:
                return f"I've processed your request using {len(successful_results)} plugin(s). " + \
                       " ".join([r.user_facing_message for r in successful_results[:2]])
            else:
                return "I encountered some issues while processing your request. Please try again or contact support."

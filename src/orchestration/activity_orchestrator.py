"""
Activity orchestrator for Moveworks-style conversational processes.

This module manages the execution of Activities within Conversational Processes,
handling Action Activities, Content Activities, and Decision Activities.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
import logging
from langchain_core.prompts import ChatPromptTemplate

from ..models.moveworks import (
    Activity, ActivityType, ActivityResult, ConversationContext,
    ConversationalProcess, ConfirmationPolicy
)
from ..actions.builtin_actions import MoveworksBuiltinActions
from ..llm.slot_inference import MoveworksSlotInference
from ..llm.llm_factory import LLMFactory


logger = logging.getLogger(__name__)


class DSLEvaluator:
    """Simple DSL evaluator for decision policies and mappings."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize DSL evaluator."""
        if llm_config is None:
            llm_config = {
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "temperature": 0.1,
                "max_tokens": 1000
            }

        self.llm = LLMFactory.create_llm(**llm_config)
    
    async def evaluate_expression(self, expression: str, context: Dict[str, Any]) -> Any:
        """
        Evaluate a DSL expression in the given context.
        
        Args:
            expression: DSL expression to evaluate
            context: Context data for evaluation
            
        Returns:
            Evaluation result
        """
        try:
            # Handle simple data references
            if expression.startswith("data."):
                return self._extract_data_reference(expression, context)
            elif expression.startswith("meta_info."):
                return self._extract_meta_reference(expression, context)
            elif expression.startswith("$"):
                return await self._evaluate_function_call(expression, context)
            else:
                # Simple literal or comparison
                return self._evaluate_simple_expression(expression, context)
        
        except Exception as e:
            logger.error(f"Error evaluating DSL expression '{expression}': {e}")
            return None
    
    def _extract_data_reference(self, expression: str, context: Dict[str, Any]) -> Any:
        """Extract data reference like data.slot_name.value"""
        parts = expression.split(".")
        current = context.get("data", {})
        
        for part in parts[1:]:  # Skip "data"
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        
        return current
    
    def _extract_meta_reference(self, expression: str, context: Dict[str, Any]) -> Any:
        """Extract meta reference like meta_info.user.email_addr"""
        parts = expression.split(".")
        current = context.get("meta_info", {})
        
        for part in parts[1:]:  # Skip "meta_info"
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        
        return current
    
    async def _evaluate_function_call(self, expression: str, context: Dict[str, Any]) -> Any:
        """Evaluate function calls like $CONCAT, $LOWERCASE, etc."""
        if expression.startswith("$CONCAT"):
            # Extract arguments from $CONCAT([...], "")
            import re
            match = re.search(r'\$CONCAT\(\[(.*?)\], ""\)', expression)
            if match:
                args_str = match.group(1)
                # Simple concatenation for demo
                return "concatenated_result"
        
        elif expression.startswith("$LOWERCASE"):
            # Extract argument from $LOWERCASE(...)
            import re
            match = re.search(r'\$LOWERCASE\((.*?)\)', expression)
            if match:
                arg = match.group(1)
                value = await self.evaluate_expression(arg, context)
                return str(value).lower() if value else ""
        
        return expression
    
    def _evaluate_simple_expression(self, expression: str, context: Dict[str, Any]) -> Any:
        """Evaluate simple expressions and comparisons."""
        # Handle boolean comparisons
        if "==" in expression:
            left, right = expression.split("==", 1)
            left_val = left.strip().strip('"\'')
            right_val = right.strip().strip('"\'')
            return left_val == right_val
        
        # Handle string literals
        if expression.startswith('"') and expression.endswith('"'):
            return expression[1:-1]
        
        return expression


class MoveworksActivityOrchestrator:
    """
    Orchestrates the execution of Activities in Conversational Processes.
    
    Handles:
    - Action Activities (execute actions with input/output mapping)
    - Content Activities (display content to users)
    - Decision Activities (route based on conditions)
    """
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the activity orchestrator."""
        if llm_config is None:
            llm_config = {
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "temperature": 0.1,
                "max_tokens": 2000
            }

        self.builtin_actions = MoveworksBuiltinActions(llm_config)
        self.slot_inference = MoveworksSlotInference(llm_config)
        self.dsl_evaluator = DSLEvaluator(llm_config)
        
        # Custom actions registry (for user-defined actions)
        self.custom_actions: Dict[str, Any] = {}
    
    async def execute_activity(
        self,
        activity: Activity,
        context: ConversationContext,
        process_data: Dict[str, Any]
    ) -> ActivityResult:
        """
        Execute a single activity.
        
        Args:
            activity: Activity to execute
            context: Conversation context
            process_data: Current process data
            
        Returns:
            ActivityResult with execution outcome
        """
        try:
            # Check if required slots are filled
            missing_slots = await self._check_required_slots(activity, process_data)
            if missing_slots:
                return ActivityResult(
                    success=False,
                    requires_user_input=True,
                    user_prompt=f"I need more information: {', '.join(missing_slots)}",
                    error_message=f"Missing required slots: {missing_slots}"
                )
            
            # Execute based on activity type
            if activity.activity_type == ActivityType.ACTION:
                return await self._execute_action_activity(activity, context, process_data)
            elif activity.activity_type == ActivityType.CONTENT:
                return await self._execute_content_activity(activity, context, process_data)
            elif activity.activity_type == ActivityType.DECISION:
                return await self._execute_decision_activity(activity, context, process_data)
            else:
                return ActivityResult(
                    success=False,
                    error_message=f"Unknown activity type: {activity.activity_type}"
                )
        
        except Exception as e:
            logger.error(f"Error executing activity: {e}")
            return ActivityResult(
                success=False,
                error_message=f"Activity execution failed: {str(e)}"
            )
    
    async def _execute_action_activity(
        self,
        activity: Activity,
        context: ConversationContext,
        process_data: Dict[str, Any]
    ) -> ActivityResult:
        """Execute an Action Activity."""
        if not activity.action_name:
            return ActivityResult(
                success=False,
                error_message="Action activity missing action_name"
            )
        
        # Map input parameters
        input_args = {}
        if activity.input_mapping:
            for param_name, mapping_expr in activity.input_mapping.mappings.items():
                mapped_value = await self.dsl_evaluator.evaluate_expression(
                    mapping_expr, 
                    {"data": process_data, "meta_info": {"user": context.user_attributes}}
                )
                input_args[param_name] = mapped_value
        
        # Check for confirmation if required
        if activity.confirmation_policy == ConfirmationPolicy.REQUIRE_CONSENT:
            # In a real implementation, this would check if user has confirmed
            # For demo, we'll assume confirmation is handled elsewhere
            pass
        
        # Execute the action
        if activity.action_name.startswith("mw.") or activity.action_name in ["generate_text_action", "get_user_by_email"]:
            # Built-in action
            action_name = activity.action_name.replace("mw.", "")
            result = await self.builtin_actions.execute_builtin_action(action_name, input_args)
        else:
            # Custom action
            result = await self._execute_custom_action(activity.action_name, input_args)
        
        # Apply output mapping if specified
        if result.success and activity.output_mapping:
            output_data = result.output_data
            
            # Apply dot walk path if specified
            if activity.output_mapping.dot_walk_path:
                output_data = self._apply_dot_walk(output_data, activity.output_mapping.dot_walk_path)
            
            # Store in process data with output key
            if activity.output_mapping.output_key:
                process_data[activity.output_mapping.output_key] = output_data
        
        return result
    
    async def _execute_content_activity(
        self,
        activity: Activity,
        context: ConversationContext,
        process_data: Dict[str, Any]
    ) -> ActivityResult:
        """Execute a Content Activity."""
        content = activity.content_text or activity.content_html or "No content specified"
        
        # Process any template variables in content
        processed_content = await self._process_content_template(content, process_data, context)
        
        return ActivityResult(
            success=True,
            output_data={"content": processed_content, "type": "content"},
            requires_user_input=False
        )
    
    async def _execute_decision_activity(
        self,
        activity: Activity,
        context: ConversationContext,
        process_data: Dict[str, Any]
    ) -> ActivityResult:
        """Execute a Decision Activity."""
        if not activity.decision_cases:
            return ActivityResult(
                success=False,
                error_message="Decision activity missing decision_cases"
            )
        
        # Evaluate decision cases
        for i, case in enumerate(activity.decision_cases):
            condition = case.get("condition")
            if condition:
                result = await self.dsl_evaluator.evaluate_expression(
                    condition,
                    {"data": process_data, "meta_info": {"user": context.user_attributes}}
                )
                
                if result:
                    # This case matches, return the next activity index
                    next_activity = case.get("next_activity", i + 1)
                    return ActivityResult(
                        success=True,
                        output_data={"decision_result": case, "case_index": i},
                        next_activity=next_activity
                    )
        
        # No case matched, continue to next activity
        return ActivityResult(
            success=True,
            output_data={"decision_result": "no_match"},
            next_activity=None
        )
    
    async def _check_required_slots(self, activity: Activity, process_data: Dict[str, Any]) -> List[str]:
        """Check if all required slots are filled."""
        missing_slots = []
        
        for slot_name in activity.required_slots:
            if slot_name not in process_data or process_data[slot_name] is None:
                missing_slots.append(slot_name)
        
        return missing_slots
    
    async def _execute_custom_action(self, action_name: str, input_args: Dict[str, Any]) -> ActivityResult:
        """Execute a custom action."""
        if action_name in self.custom_actions:
            action_func = self.custom_actions[action_name]
            try:
                result = await action_func(input_args)
                return ActivityResult(
                    success=True,
                    output_data=result
                )
            except Exception as e:
                return ActivityResult(
                    success=False,
                    error_message=f"Custom action failed: {str(e)}"
                )
        else:
            return ActivityResult(
                success=False,
                error_message=f"Custom action not found: {action_name}"
            )
    
    def _apply_dot_walk(self, data: Any, dot_path: str) -> Any:
        """Apply dot walk path to extract nested data."""
        if not dot_path.startswith("."):
            return data
        
        path_parts = dot_path[1:].split(".")  # Remove leading dot
        current = data
        
        for part in path_parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                current = current[index] if 0 <= index < len(current) else None
            else:
                return None
        
        return current
    
    async def _process_content_template(
        self,
        content: str,
        process_data: Dict[str, Any],
        context: ConversationContext
    ) -> str:
        """Process template variables in content."""
        # Simple template processing for demo
        # In production, this would use a proper template engine
        
        processed = content
        
        # Replace data references
        import re
        data_refs = re.findall(r'\{\{data\.(.*?)\}\}', content)
        for ref in data_refs:
            value = process_data.get(ref, f"{{{{data.{ref}}}}}")
            processed = processed.replace(f"{{{{data.{ref}}}}}", str(value))
        
        # Replace user references
        user_refs = re.findall(r'\{\{user\.(.*?)\}\}', content)
        for ref in user_refs:
            value = context.user_attributes.get(ref, f"{{{{user.{ref}}}}}")
            processed = processed.replace(f"{{{{user.{ref}}}}}", str(value))
        
        return processed
    
    def register_custom_action(self, action_name: str, action_func: Any) -> None:
        """Register a custom action function."""
        self.custom_actions[action_name] = action_func
        logger.info(f"Registered custom action: {action_name}")
    
    async def execute_process_activities(
        self,
        process: ConversationalProcess,
        context: ConversationContext,
        process_data: Dict[str, Any],
        start_activity: int = 0
    ) -> List[ActivityResult]:
        """
        Execute all activities in a conversational process.
        
        Args:
            process: Conversational process to execute
            context: Conversation context
            process_data: Process data and slot values
            start_activity: Index of activity to start from
            
        Returns:
            List of ActivityResult for each executed activity
        """
        results = []
        current_activity = start_activity
        
        while current_activity < len(process.activities):
            activity = process.activities[current_activity]
            
            logger.info(f"Executing activity {current_activity}: {activity.activity_type}")
            
            result = await self.execute_activity(activity, context, process_data)
            results.append(result)
            
            if not result.success:
                logger.error(f"Activity {current_activity} failed: {result.error_message}")
                break
            
            if result.requires_user_input:
                logger.info(f"Activity {current_activity} requires user input")
                break
            
            # Determine next activity
            if result.next_activity is not None:
                current_activity = result.next_activity
            else:
                current_activity += 1
        
        return results


# Example usage and testing
async def test_activity_orchestrator():
    """Test the activity orchestrator."""
    from ..models.moveworks import (
        Activity, ActivityType, InputMapping, OutputMapping,
        ConversationContext, GenerateTextActionInput
    )
    
    # Sample context
    context = ConversationContext(
        user_id="test_user",
        session_id="test_session",
        thread_id="test_thread",
        user_attributes={"email_addr": "john.doe@company.com", "department": "Engineering"}
    )
    
    # Sample process data
    process_data = {
        "item_name": "laptop",
        "quantity": 1
    }
    
    # Sample action activity
    action_activity = Activity(
        activity_type=ActivityType.ACTION,
        action_name="generate_text_action",
        required_slots=["item_name"],
        input_mapping=InputMapping(mappings={
            "system_prompt": "Classify this purchase as Opex or Capex",
            "user_input": "data.item_name"
        }),
        output_mapping=OutputMapping(
            dot_walk_path=".openai_chat_completions_response.choices[0].message.content",
            output_key="classification_result"
        )
    )
    
    # Sample content activity
    content_activity = Activity(
        activity_type=ActivityType.CONTENT,
        content_text="Your purchase request for {{data.item_name}} has been classified as {{data.classification_result}}"
    )
    
    orchestrator = MoveworksActivityOrchestrator()
    
    print("Testing Activity Orchestrator:")
    print("=" * 50)
    
    # Test action activity
    print("1. Testing Action Activity:")
    result = await orchestrator.execute_activity(action_activity, context, process_data)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Output data keys: {list(result.output_data.keys())}")
        print(f"Process data after: {list(process_data.keys())}")
    print("-" * 30)
    
    # Test content activity
    print("2. Testing Content Activity:")
    result = await orchestrator.execute_activity(content_activity, context, process_data)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Content: {result.output_data.get('content')}")
    print("-" * 30)


if __name__ == "__main__":
    asyncio.run(test_activity_orchestrator())

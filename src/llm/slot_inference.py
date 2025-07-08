"""
LangChain-based slot inference for Moveworks-style conversational processes.

This module uses LLM to infer slot values from conversation context,
following Moveworks' slot inference policies and slot descriptions.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..models.moveworks import (
    Slot, SlotInferencePolicy, SlotInferenceResult,
    ConversationContext, ResolverStrategy, StaticOption
)
# Import moved to avoid circular dependency
from .llm_factory import LLMFactory


logger = logging.getLogger(__name__)


class SlotInferenceOutput(BaseModel):
    """Structured output for slot inference."""
    slot_name: str = Field(..., description="Name of the slot")
    inferred_value: Optional[Union[str, int, float, bool]] = Field(None, description="Inferred value for the slot")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    reasoning: str = Field(..., description="Explanation of how the value was inferred")
    needs_clarification: bool = Field(False, description="Whether user clarification is needed")
    clarification_question: Optional[str] = Field(None, description="Question to ask user for clarification")


class MoveworksSlotInference:
    """
    LangChain-based slot inference that follows Moveworks patterns.
    
    Handles:
    - "Infer slot value if available" policy
    - "Always explicitly ask for slot" policy
    - Slot descriptions to guide AI behavior
    - Static resolver integration
    """
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the slot inference system."""
        # Use LLM factory to create Gemini instance
        import os
        from .llm_factory import LLMFactory

        # Set the API key
        os.environ["GOOGLE_API_KEY"] = "AIzaSyC9vwixXT9XdbFzCfFapNR21wYA5ma7LDg"

        self.llm = LLMFactory.create_llm(
            provider="gemini",
            model="gemini-1.5-flash",
            temperature=0.1,
            max_tokens=1000
        )

        self.output_parser = PydanticOutputParser(pydantic_object=SlotInferenceOutput)

        # Initialize dynamic resolver execution engine (lazy import to avoid circular dependency)
        self.resolver_engine = None
        
        # Slot inference prompt template
        self.inference_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at inferring slot values from conversational context in a Moveworks-style AI assistant.

Your task is to analyze the conversation and determine if you can infer a value for a specific slot based on:
1. The slot's description and requirements
2. The slot's inference policy
3. The conversation context and user utterances
4. Available resolver options (if any)

Key principles:
- Follow the slot's inference policy strictly
- Use the slot description to guide your understanding
- Be conservative - only infer if you're confident
- Consider the user's intent and context
- If static options are available, match to those options

{format_instructions}"""),
            ("human", """Conversation Context:
User Utterance: "{user_utterance}"
Previous Messages: {conversation_history}
User Attributes: {user_attributes}

Slot to Infer:
Name: {slot_name}
Data Type: {slot_data_type}
Description: {slot_description}
Inference Policy: {inference_policy}
Validation Policy: {validation_policy}
Static Options: {static_options}

Based on the conversation context and slot requirements, determine if you can infer a value for this slot.""")
        ])
    
    async def infer_slot_value(
        self,
        slot: Slot,
        user_utterance: str,
        context: ConversationContext,
        conversation_history: Optional[List[str]] = None
    ) -> SlotInferenceResult:
        """
        Infer a slot value from conversation context.
        
        Args:
            slot: Slot definition to infer
            user_utterance: Current user utterance
            context: Conversation context
            conversation_history: Previous conversation messages
            
        Returns:
            SlotInferenceResult with inference outcome
        """
        try:
            # Check if we should always ask (skip inference)
            if slot.slot_inference_policy == SlotInferencePolicy.ALWAYS_ASK:
                return await self._generate_clarification_question(slot, context)

            # Handle custom data types with dynamic resolver execution
            if slot.custom_data_type_name:
                logger.info(f"Executing dynamic resolver for custom data type: {slot.custom_data_type_name}")

                # Lazy import to avoid circular dependency
                if self.resolver_engine is None:
                    from .resolver_execution_engine import DynamicResolverExecutionEngine
                    self.resolver_engine = DynamicResolverExecutionEngine(llm=self.llm)

                resolver_result = await self.resolver_engine.execute_resolver_for_slot(
                    slot, user_utterance, context, conversation_history
                )

                if resolver_result.success:
                    return SlotInferenceResult(
                        slot_name=slot.name,
                        inferred_value=resolver_result.resolved_value,
                        confidence=0.9,  # High confidence for successful resolver execution
                        needs_clarification=False,
                        clarification_options=None
                    )
                else:
                    # Resolver failed, ask for clarification
                    return SlotInferenceResult(
                        slot_name=slot.name,
                        inferred_value=None,
                        confidence=0.0,
                        needs_clarification=True,
                        clarification_options=[f"Resolver execution failed: {resolver_result.error_message}"]
                    )

            # Prepare context for LLM (for non-custom data types)
            history_text = "\n".join(conversation_history or [])
            static_options_text = self._format_static_options(slot.resolver_strategy)
            
            # Execute inference
            chain = self.inference_prompt | self.llm | self.output_parser
            
            result = await chain.ainvoke({
                "user_utterance": user_utterance,
                "conversation_history": history_text,
                "user_attributes": str(context.user_attributes),
                "slot_name": slot.name,
                "slot_data_type": slot.data_type.value,
                "slot_description": slot.slot_description,
                "inference_policy": slot.slot_inference_policy.value,
                "validation_policy": slot.slot_validation_policy or "None",
                "static_options": static_options_text,
                "format_instructions": self.output_parser.get_format_instructions()
            })
            
            # Validate inferred value if provided
            if result.inferred_value is not None:
                validation_result = await self._validate_slot_value(slot, result.inferred_value)
                if not validation_result:
                    result.needs_clarification = True
                    result.clarification_question = f"The value '{result.inferred_value}' is not valid for {slot.name}. {slot.slot_validation_description or 'Please provide a valid value.'}"
            
            return SlotInferenceResult(
                slot_name=result.slot_name,
                inferred_value=result.inferred_value,
                confidence=result.confidence,
                needs_clarification=result.needs_clarification,
                clarification_options=self._get_clarification_options(slot) if result.needs_clarification else None
            )
        
        except Exception as e:
            logger.error(f"Error in slot inference for {slot.name}: {e}")
            return await self._generate_clarification_question(slot, context)
    
    async def infer_multiple_slots(
        self,
        slots: List[Slot],
        user_utterance: str,
        context: ConversationContext,
        conversation_history: Optional[List[str]] = None
    ) -> List[SlotInferenceResult]:
        """
        Infer values for multiple slots simultaneously.
        
        Args:
            slots: List of slots to infer
            user_utterance: Current user utterance
            context: Conversation context
            conversation_history: Previous conversation messages
            
        Returns:
            List of SlotInferenceResult
        """
        tasks = [
            self.infer_slot_value(slot, user_utterance, context, conversation_history)
            for slot in slots
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error inferring slot {slots[i].name}: {result}")
                processed_results.append(await self._generate_clarification_question(slots[i], context))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _format_static_options(self, resolver_strategy: Optional[ResolverStrategy]) -> str:
        """Format static options for LLM consumption."""
        if not resolver_strategy or not resolver_strategy.static_options:
            return "None"
        
        options = []
        for option in resolver_strategy.static_options:
            options.append(f"  - {option.display_value} (value: {option.raw_value})")
        
        return "\n".join(options)
    
    async def _validate_slot_value(self, slot: Slot, value: Any) -> bool:
        """
        Validate a slot value against its validation policy.
        
        Args:
            slot: Slot definition
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not slot.slot_validation_policy:
            return True
        
        try:
            # Simple validation for common cases
            # In a full implementation, this would use the DSL engine
            validation_policy = slot.slot_validation_policy.lower()
            
            if "value > 0" in validation_policy:
                return isinstance(value, (int, float)) and value > 0
            elif "value == true" in validation_policy:
                return value is True
            elif "value in [" in validation_policy:
                # Extract list from validation policy
                import re
                match = re.search(r"value in \[(.*?)\]", validation_policy)
                if match:
                    valid_values = [v.strip().strip("'\"") for v in match.group(1).split(",")]
                    return str(value) in valid_values
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating slot value: {e}")
            return False
    
    async def _generate_clarification_question(
        self,
        slot: Slot,
        context: ConversationContext
    ) -> SlotInferenceResult:
        """
        Generate a clarification question for a slot.
        
        Args:
            slot: Slot that needs clarification
            context: Conversation context
            
        Returns:
            SlotInferenceResult with clarification question
        """
        # Generate contextual clarification question
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are generating a natural clarification question for a slot in a conversational AI. Be friendly and specific."),
            ("human", """Generate a clarification question for this slot:

Slot Name: {slot_name}
Slot Description: {slot_description}
Data Type: {data_type}
Static Options: {static_options}

The question should be natural and help the user understand what information is needed.""")
        ])
        
        try:
            chain = question_prompt | self.llm
            
            result = await chain.ainvoke({
                "slot_name": slot.name,
                "slot_description": slot.slot_description,
                "data_type": slot.data_type.value,
                "static_options": self._format_static_options(slot.resolver_strategy)
            })
            
            clarification_question = result.content
        
        except Exception as e:
            logger.error(f"Error generating clarification question: {e}")
            clarification_question = f"Could you please provide a value for {slot.name}? {slot.slot_description}"
        
        return SlotInferenceResult(
            slot_name=slot.name,
            inferred_value=None,
            confidence=0.0,
            needs_clarification=True,
            clarification_options=self._get_clarification_options(slot)
        )
    
    def _get_clarification_options(self, slot: Slot) -> Optional[List[str]]:
        """Get clarification options for a slot (e.g., from static resolver)."""
        if slot.resolver_strategy and slot.resolver_strategy.static_options:
            return [option.display_value for option in slot.resolver_strategy.static_options]
        return None
    
    async def resolve_slot_with_static_options(
        self,
        slot: Slot,
        user_input: str
    ) -> Optional[str]:
        """
        Resolve user input against static options using semantic matching.
        
        Args:
            slot: Slot with static resolver
            user_input: User's input to match
            
        Returns:
            Raw value if matched, None otherwise
        """
        if not slot.resolver_strategy or not slot.resolver_strategy.static_options:
            return None
        
        # Use LLM for semantic matching
        matching_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are matching user input to predefined options. Find the best semantic match."),
            ("human", """User input: "{user_input}"

Available options:
{options}

Return the raw_value of the best matching option, or "NO_MATCH" if no good match exists.""")
        ])
        
        options_text = "\n".join([
            f"- {opt.display_value} (raw_value: {opt.raw_value})"
            for opt in slot.resolver_strategy.static_options
        ])
        
        try:
            chain = matching_prompt | self.llm
            
            result = await chain.ainvoke({
                "user_input": user_input,
                "options": options_text
            })
            
            matched_value = result.content.strip()
            
            if matched_value == "NO_MATCH":
                return None
            
            # Verify the returned value exists in our options
            valid_values = [opt.raw_value for opt in slot.resolver_strategy.static_options]
            if matched_value in valid_values:
                return matched_value
            
            return None
        
        except Exception as e:
            logger.error(f"Error in static option matching: {e}")
            return None


# Example usage and testing
async def test_slot_inference():
    """Test the slot inference system."""
    from ..models.moveworks import (
        Slot, DataType, SlotInferencePolicy, ResolverStrategy, 
        ResolverMethodType, StaticOption, ConversationContext
    )
    
    # Sample slot with static options
    pto_type_slot = Slot(
        name="pto_type",
        data_type=DataType.STRING,
        slot_description="Type of PTO balance to fetch (vacation, sick leave, or personal time)",
        slot_inference_policy=SlotInferencePolicy.INFER_IF_AVAILABLE,
        slot_validation_policy="value IN ['vacation', 'sick', 'personal']",
        resolver_strategy=ResolverStrategy(
            method_name="choose_pto_type",
            method_type=ResolverMethodType.STATIC,
            static_options=[
                StaticOption(display_value="Vacation", raw_value="vacation"),
                StaticOption(display_value="Sick Leave", raw_value="sick"),
                StaticOption(display_value="Personal Time", raw_value="personal")
            ]
        )
    )
    
    # Sample context
    context = ConversationContext(
        user_id="test_user",
        session_id="test_session",
        thread_id="test_thread",
        user_attributes={"department": "Engineering", "role": "Developer"}
    )
    
    # Test utterances
    test_utterances = [
        "What's my vacation balance?",
        "How many sick days do I have?",
        "Check my PTO balance",
        "I need time off"
    ]
    
    inference_engine = MoveworksSlotInference()
    
    print("Testing Slot Inference:")
    print("=" * 50)
    
    for utterance in test_utterances:
        result = await inference_engine.infer_slot_value(
            pto_type_slot, utterance, context
        )
        
        print(f"Utterance: '{utterance}'")
        print(f"Inferred Value: {result.inferred_value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Needs Clarification: {result.needs_clarification}")
        if result.clarification_options:
            print(f"Options: {result.clarification_options}")
        print("-" * 30)


if __name__ == "__main__":
    asyncio.run(test_slot_inference())

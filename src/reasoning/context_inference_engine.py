"""
Moveworks Context Inference Engine

This is the missing piece that makes Moveworks intelligent!
Instead of asking humans for every slot, the AI analyzes conversation context
to extract slot values using LLM-powered inference.

This is what separates proper Moveworks architecture from basic chatbots.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from llm.llm_factory import LLMFactory
from .moveworks_slot_system import (
    MoveworksSlot, SlotResolutionResult, MoveworksConversationContext, ConversationMessage
)

logger = logging.getLogger(__name__)


class MoveworksContextInferenceEngine:
    """
    LLM-powered context inference engine for Moveworks slot resolution.
    
    This is the core intelligence that makes Moveworks smart:
    - Analyzes conversation history to extract slot values
    - Understands context, implications, and user intent
    - Reduces need for human-in-the-loop to confirmations only
    """
    
    def __init__(self):
        self.llm_factory = LLMFactory()
        self.llm = None  # Will be initialized when needed
        
    async def initialize(self):
        """Initialize the LLM for inference."""
        try:
            # Initialize LLM using factory
            self.llm = self.llm_factory.get_default_llm()
            logger.info(f"✅ Context inference engine initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize context inference engine: {e}")
            raise
    
    async def infer_slot_from_context(self, slot: MoveworksSlot, 
                                    context: MoveworksConversationContext) -> Optional[Any]:
        """
        Core method: Infer slot value from conversation context using LLM.
        
        This is what makes Moveworks intelligent - analyzing conversation
        to extract information instead of asking humans for everything.
        """
        if not self.llm:
            await self.initialize()
        
        try:
            # Build conversation context
            conversation_text = self._build_conversation_context(context)
            
            # Create extraction prompt
            prompt = self._create_extraction_prompt(slot, conversation_text, context)
            
            # Use LLM to analyze and extract
            response = await self.llm.ainvoke(prompt)
            
            # Parse and validate response
            extracted_value = self._parse_llm_response(response.content, slot)
            
            if extracted_value is not None:
                logger.info(f"✅ Inferred {slot.name}: {extracted_value}")
                return extracted_value
            else:
                logger.debug(f"🔍 Could not infer {slot.name} from context")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error inferring slot {slot.name}: {e}")
            return None
    
    def _build_conversation_context(self, context: MoveworksConversationContext) -> str:
        """Build conversation context for LLM analysis."""
        
        # Include user info
        context_parts = [
            f"User: {context.user_id}",
            f"Domain: {context.primary_domain}",
            f"Conversation started: {context.created_at.strftime('%Y-%m-%d %H:%M')}"
        ]
        
        # Include conversation history
        if context.message_history:
            context_parts.append("\nConversation History:")
            for msg in context.message_history[-10:]:  # Last 10 messages for context
                timestamp = msg.timestamp.strftime('%H:%M')
                context_parts.append(f"[{timestamp}] {msg.actor}: {msg.content}")
        
        # Include already resolved slots for context
        if context.resolved_slots:
            context_parts.append("\nAlready Known:")
            for slot_name, result in context.resolved_slots.items():
                context_parts.append(f"- {slot_name}: {result.resolved_value}")
        
        return "\n".join(context_parts)
    
    def _create_extraction_prompt(self, slot: MoveworksSlot, conversation_text: str, 
                                context: MoveworksConversationContext) -> str:
        """Create LLM prompt for slot extraction."""
        
        current_time = datetime.now()
        
        prompt = f"""You are a Moveworks AI assistant analyzing a conversation to extract specific information.

TASK: Extract the value for slot '{slot.name}' from the conversation context.

SLOT DETAILS:
- Name: {slot.name}
- Type: {slot.data_type}
- Description: {slot.description}

CONTEXT:
{conversation_text}

CURRENT TIME: {current_time.strftime('%Y-%m-%d %H:%M:%S')} (Use this for relative dates)

EXTRACTION RULES:
1. Look for explicit mentions of the slot value
2. Consider implied information and context
3. Use common sense for date/time references
4. If the slot is mentioned in any form, extract it
5. If not mentioned at all, return NULL

EXAMPLES:
- "vacation time" → pto_type: "vacation"
- "next week" → start_date: "{(current_time + timedelta(days=7)).strftime('%Y-%m-%d')}"
- "5 days" → duration: 5
- "John Smith" → employee_name: "John Smith"
- "my name is Sarah" → employee_name: "Sarah"

RESPONSE FORMAT:
Return ONLY the extracted value or NULL if not found.
For dates, use YYYY-MM-DD format.
For text, return the exact relevant text.
For numbers, return just the number.

EXTRACTED VALUE:"""

        return prompt
    
    def _parse_llm_response(self, response: str, slot: MoveworksSlot) -> Optional[Any]:
        """Parse LLM response and convert to appropriate type."""
        
        response = response.strip()
        
        # Handle NULL responses
        if response.upper() in ["NULL", "NONE", "NOT FOUND", "N/A", ""]:
            return None
        
        try:
            # Type conversion based on slot data type
            if slot.data_type == "string":
                return response
            elif slot.data_type == "number":
                # Extract number from response
                import re
                numbers = re.findall(r'-?\d+\.?\d*', response)
                if numbers:
                    return float(numbers[0])
                return None
            elif slot.data_type == "integer":
                import re
                numbers = re.findall(r'-?\d+', response)
                if numbers:
                    return int(numbers[0])
                return None
            elif slot.data_type == "boolean":
                return response.lower() in ["true", "yes", "1", "on", "enabled"]
            elif slot.data_type == "User":
                # For User type, create user object
                return {
                    "display_name": response,
                    "email": f"{response.lower().replace(' ', '.')}@company.com",
                    "employee_id": f"EMP_{response.upper().replace(' ', '')}"
                }
            else:
                # For custom types, return as string for now
                return response
                
        except Exception as e:
            logger.error(f"Error parsing LLM response '{response}' for slot {slot.name}: {e}")
            return None
    
    async def infer_multiple_slots(self, slots: List[MoveworksSlot], 
                                 context: MoveworksConversationContext) -> Dict[str, Any]:
        """Infer multiple slots in a single LLM call for efficiency."""
        
        if not self.llm:
            await self.initialize()
        
        try:
            conversation_text = self._build_conversation_context(context)
            
            # Create multi-slot extraction prompt
            slot_descriptions = []
            for slot in slots:
                slot_descriptions.append(f"- {slot.name} ({slot.data_type}): {slot.description}")
            
            prompt = f"""Analyze this conversation and extract values for multiple slots:

SLOTS TO EXTRACT:
{chr(10).join(slot_descriptions)}

CONTEXT:
{conversation_text}

Return a JSON object with slot names as keys and extracted values as values.
Use null for slots that cannot be determined from the context.

Example response:
{{
    "pto_type": "vacation",
    "employee_name": "John Smith", 
    "start_date": "2024-12-23",
    "duration": 5
}}

RESPONSE:"""

            response = await self.llm.ainvoke(prompt)
            
            # Parse JSON response
            try:
                extracted_data = json.loads(response.content.strip())
                
                # Validate and convert types
                results = {}
                for slot in slots:
                    if slot.name in extracted_data and extracted_data[slot.name] is not None:
                        parsed_value = self._parse_llm_response(str(extracted_data[slot.name]), slot)
                        if parsed_value is not None:
                            results[slot.name] = parsed_value
                
                return results
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {response.content}")
                return {}
                
        except Exception as e:
            logger.error(f"Error in multi-slot inference: {e}")
            return {}


# Global inference engine instance
context_inference_engine = MoveworksContextInferenceEngine()

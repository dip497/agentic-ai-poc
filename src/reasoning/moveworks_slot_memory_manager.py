"""
Moveworks Slot-Based Memory Manager

This module implements the Moveworks slot-based memory management system:
1. Slot Resolution: Convert user input to typed slot values
2. Conversational Process State: Track process execution through slots
3. Context Management: Maintain conversation context and history
4. Validation: Apply DSL rules to validate slot values

This follows the official Moveworks architecture patterns.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

from .moveworks_slot_system import (
    MoveworksSlot, SlotInferencePolicy, SlotResolutionResult,
    MoveworksConversationContext, ConversationalProcessState, ConversationMessage
)
from .moveworks_data_types import data_type_registry
from .moveworks_dsl_engine import dsl_validator
from .moveworks_resolver_engine import resolver_engine
from .context_inference_engine import context_inference_engine

logger = logging.getLogger(__name__)


class MoveworksSlotBasedMemoryManager:
    """
    Moveworks slot-based memory manager following official architecture.
    Replaces the old 4-type memory system with slot-centric approach.
    """
    
    def __init__(self):
        self.active_conversations: Dict[str, MoveworksConversationContext] = {}
        self.active_processes: Dict[str, ConversationalProcessState] = {}
        self.slot_definitions: Dict[str, MoveworksSlot] = {}
    
    async def initialize(self):
        """Initialize the memory manager and context inference engine."""
        try:
            # Initialize context inference engine for smart slot resolution
            await context_inference_engine.initialize()
            logger.info("✅ Moveworks slot-based memory manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory manager: {e}")
            raise
    
    def register_slot(self, slot: MoveworksSlot):
        """Register a slot definition."""
        self.slot_definitions[slot.name] = slot
    
    async def create_conversation_context(self, user_id: str, primary_domain: str = "IT") -> MoveworksConversationContext:
        """Create new conversation context."""
        conversation_id = str(uuid.uuid4())
        
        context = MoveworksConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            primary_domain=primary_domain
        )
        
        self.active_conversations[conversation_id] = context
        return context
    
    async def get_conversation_context(self, conversation_id: str) -> Optional[MoveworksConversationContext]:
        """Get conversation context."""
        return self.active_conversations.get(conversation_id)
    
    async def add_message(self, conversation_id: str, content: str, actor: str, message_type: str = "text"):
        """Add message to conversation."""
        context = await self.get_conversation_context(conversation_id)
        if not context:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        context.add_message(content, actor, message_type)
    
    async def resolve_slot(self, conversation_id: str, slot: MoveworksSlot,
                          user_context: Dict[str, Any] = None) -> SlotResolutionResult:
        """
        Proper Moveworks slot resolution with AI-powered context inference.

        This implements the correct Moveworks flow:
        1. Try context inference using LLM (PRIMARY - This is what makes Moveworks smart!)
        2. Use resolver strategies for complex types
        3. Apply DSL validation
        4. Only use HITL for confirmations/disambiguation (NOT data collection)
        """
        context = await self.get_conversation_context(conversation_id)
        if not context:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Check if slot already resolved
        if context.is_slot_resolved(slot.name):
            return context.resolved_slots[slot.name]

        # Step 1: Try AI-powered context inference (MOVEWORKS INTELLIGENCE)
        if slot.inference_policy == SlotInferencePolicy.INFER_IF_AVAILABLE:
            inferred_value = await context_inference_engine.infer_slot_from_context(slot, context)

            if inferred_value is not None:
                # Validate inferred value
                if slot.validation_policy:
                    validation_result = dsl_validator.validate_slot_value(
                        slot.name, inferred_value, slot.validation_policy, user_context
                    )

                    if validation_result.is_valid:
                        result = SlotResolutionResult(
                            slot_name=slot.name,
                            resolved_value=inferred_value,
                            raw_value="inferred_from_context",
                            confidence=0.9,
                            resolution_method="context_inference",
                            validation_passed=True
                        )

                        # Store successful inference
                        context.resolve_slot(slot.name, result)
                        logger.info(f"✅ AI inferred {slot.name}: {inferred_value}")
                        return result
                else:
                    # No validation needed, accept inferred value
                    result = SlotResolutionResult(
                        slot_name=slot.name,
                        resolved_value=inferred_value,
                        raw_value="inferred_from_context",
                        confidence=0.9,
                        resolution_method="context_inference",
                        validation_passed=True
                    )

                    context.resolve_slot(slot.name, result)
                    logger.info(f"✅ AI inferred {slot.name}: {inferred_value}")
                    return result

        # Step 2: Try resolver strategies for complex types (if inference failed)
        if slot.resolver_strategy:
            # Use the last user message for resolver input
            last_message = context.message_history[-1].content if context.message_history else ""

            result = await resolver_engine.resolve_slot_value(
                slot.name, last_message, slot.data_type, slot.resolver_strategy, context
            )

            if result.resolved_value is not None and result.validation_passed:
                context.resolve_slot(slot.name, result)
                return result

        # Step 3: Return unresolved (let process handle missing slots appropriately)
        # NOTE: We DON'T automatically ask humans here - that's not proper Moveworks!
        result = SlotResolutionResult(
            slot_name=slot.name,
            resolved_value=None,
            raw_value="unresolved",
            confidence=0.0,
            resolution_method="unresolved",
            validation_passed=False,
            validation_errors=["Could not infer from context and no resolver available"]
        )

        logger.debug(f"🔍 Could not resolve {slot.name} - may need clarification")
        return result

    async def smart_resolve_all_slots(self, conversation_id: str, required_slots: List[str]) -> Dict[str, SlotResolutionResult]:
        """
        Smart resolution of all required slots using AI context inference.
        This is the proper Moveworks approach - analyze conversation once for all slots.
        """
        context = await self.get_conversation_context(conversation_id)
        if not context:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Get slot definitions
        slots_to_resolve = []
        for slot_name in required_slots:
            if slot_name in self.slot_definitions and not context.is_slot_resolved(slot_name):
                slots_to_resolve.append(self.slot_definitions[slot_name])

        if not slots_to_resolve:
            return {}

        # Use multi-slot inference for efficiency
        inferred_values = await context_inference_engine.infer_multiple_slots(slots_to_resolve, context)

        results = {}
        for slot in slots_to_resolve:
            if slot.name in inferred_values:
                # Validate and store inferred value
                inferred_value = inferred_values[slot.name]

                if slot.validation_policy:
                    validation_result = dsl_validator.validate_slot_value(
                        slot.name, inferred_value, slot.validation_policy
                    )

                    if validation_result.is_valid:
                        result = SlotResolutionResult(
                            slot_name=slot.name,
                            resolved_value=inferred_value,
                            raw_value="multi_slot_inference",
                            confidence=0.9,
                            resolution_method="multi_slot_context_inference",
                            validation_passed=True
                        )

                        context.resolve_slot(slot.name, result)
                        results[slot.name] = result
                        logger.info(f"✅ AI inferred {slot.name}: {inferred_value}")
                else:
                    result = SlotResolutionResult(
                        slot_name=slot.name,
                        resolved_value=inferred_value,
                        raw_value="multi_slot_inference",
                        confidence=0.9,
                        resolution_method="multi_slot_context_inference",
                        validation_passed=True
                    )

                    context.resolve_slot(slot.name, result)
                    results[slot.name] = result
                    logger.info(f"✅ AI inferred {slot.name}: {inferred_value}")

        return results
    
    async def create_process_state(self, process_name: str, conversation_id: str, 
                                 required_slots: List[str], optional_slots: List[str] = None) -> ConversationalProcessState:
        """Create new conversational process state."""
        process_id = str(uuid.uuid4())
        
        process_state = ConversationalProcessState(
            process_id=process_id,
            process_name=process_name,
            conversation_id=conversation_id,
            required_slots=required_slots,
            optional_slots=optional_slots or []
        )
        
        self.active_processes[process_id] = process_state
        
        # Update conversation context
        context = await self.get_conversation_context(conversation_id)
        if context:
            context.current_process_id = process_id
        
        return process_state
    
    async def get_process_state(self, process_id: str) -> Optional[ConversationalProcessState]:
        """Get process state."""
        return self.active_processes.get(process_id)
    
    async def update_process_activity(self, process_id: str, activity_name: str, result: Any):
        """Update process with activity result."""
        process_state = await self.get_process_state(process_id)
        if process_state:
            process_state.complete_activity(activity_name, result)
    
    async def complete_process(self, process_id: str, final_output: Dict[str, Any]):
        """Complete a conversational process."""
        process_state = await self.get_process_state(process_id)
        if process_state:
            process_state.complete_process(final_output)
    
    async def get_next_required_slot(self, conversation_id: str) -> Optional[MoveworksSlot]:
        """Get next slot that needs resolution for active process."""
        context = await self.get_conversation_context(conversation_id)
        if not context or not context.current_process_id:
            return None
        
        process_state = await self.get_process_state(context.current_process_id)
        if not process_state:
            return None
        
        next_slot_name = process_state.get_next_pending_slot(context)
        if next_slot_name and next_slot_name in self.slot_definitions:
            return self.slot_definitions[next_slot_name]
        
        return None
    
    async def are_all_required_slots_resolved(self, conversation_id: str) -> bool:
        """Check if all required slots are resolved for active process."""
        context = await self.get_conversation_context(conversation_id)
        if not context or not context.current_process_id:
            return False
        
        process_state = await self.get_process_state(context.current_process_id)
        if not process_state:
            return False
        
        return process_state.all_required_slots_resolved(context)
    
    def create_sample_slots(self) -> Dict[str, MoveworksSlot]:
        """Create sample slots for testing."""
        from .moveworks_slot_system import ResolverStrategy, ResolverMethodType, StaticResolverOption
        
        # PTO Type slot with static resolver
        pto_resolver = ResolverStrategy(
            method_name="choose_pto_type",
            method_type=ResolverMethodType.STATIC,
            output_data_type="string",
            static_options=[
                StaticResolverOption("Vacation", "vacation"),
                StaticResolverOption("Sick", "sick"),
                StaticResolverOption("Personal", "personal")
            ]
        )
        
        pto_type_slot = MoveworksSlot(
            name="pto_type",
            data_type="string",
            description="The type of PTO balance the user is requesting",
            resolver_strategy=pto_resolver,
            validation_policy="value IN ['vacation', 'sick', 'personal']"
        )
        
        # Employee slot with User type
        employee_slot = MoveworksSlot(
            name="employee",
            data_type="User",
            description="Employee to look up",
            inference_policy=SlotInferencePolicy.INFER_IF_AVAILABLE
        )
        
        # Quantity slot with validation
        quantity_slot = MoveworksSlot(
            name="quantity",
            data_type="number",
            description="Quantity of items",
            validation_policy="value > 0",
            validation_description="Quantity must be greater than zero"
        )
        
        # Due date slot with time validation
        due_date_slot = MoveworksSlot(
            name="due_date",
            data_type="string",
            description="Due date for the task",
            validation_policy="$PARSE_TIME(value) > $TIME()",
            validation_description="Due date must be in the future"
        )
        
        slots = {
            "pto_type": pto_type_slot,
            "employee": employee_slot,
            "quantity": quantity_slot,
            "due_date": due_date_slot
        }
        
        # Register slots
        for slot in slots.values():
            self.register_slot(slot)
        
        return slots


# Global memory manager instance
moveworks_slot_memory_manager = MoveworksSlotBasedMemoryManager()

"""
Moveworks Resolver Strategy Engine

Implements resolver strategies for converting user input to system data types:
1. Static Resolvers: Predefined option mappings (display -> raw value)
2. Dynamic Resolvers: API-based resolution with search
3. Built-in Resolvers: User lookup, etc.

Each resolver strategy is bound to exactly one data type following Moveworks patterns.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import re

from .moveworks_slot_system import (
    ResolverStrategy, ResolverMethodType, StaticResolverOption,
    SlotResolutionResult, MoveworksConversationContext
)
from .moveworks_data_types import data_type_registry


@dataclass
class ResolverExecutionContext:
    """Context for resolver execution."""
    user_input: str
    conversation_context: MoveworksConversationContext
    slot_name: str
    additional_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolverResult:
    """Result from resolver execution."""
    success: bool
    resolved_value: Any = None
    raw_value: Any = None
    confidence: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class StaticResolver:
    """Static resolver for predefined option mappings."""
    
    def __init__(self, strategy: ResolverStrategy):
        self.strategy = strategy
        # Create mapping for both display values and raw values
        self.options_map = {}
        self.reverse_map = {}

        for opt in strategy.static_options:
            # Map display value (case insensitive) to raw value
            self.options_map[opt.display_value.lower()] = opt.raw_value
            # Map raw value to display value
            self.reverse_map[opt.raw_value] = opt.display_value
            # Also map raw value to itself for direct matches
            self.options_map[opt.raw_value.lower()] = opt.raw_value
    
    async def resolve(self, context: ResolverExecutionContext) -> ResolverResult:
        """Resolve user input to static option."""
        user_input = context.user_input.strip()
        user_input_lower = user_input.lower()

        # Exact match (case insensitive)
        if user_input_lower in self.options_map:
            return ResolverResult(
                success=True,
                resolved_value=self.options_map[user_input_lower],
                raw_value=context.user_input,
                confidence=1.0,
                metadata={"match_type": "exact", "display_value": self._get_display_value(self.options_map[user_input_lower])}
            )

        # Try exact match with display values from strategy options
        for opt in self.strategy.static_options:
            if user_input.lower() == opt.display_value.lower():
                return ResolverResult(
                    success=True,
                    resolved_value=opt.raw_value,
                    raw_value=context.user_input,
                    confidence=1.0,
                    metadata={"match_type": "exact_display", "display_value": opt.display_value}
                )
        
        # Fuzzy match
        best_match = None
        best_score = 0.0

        # Check if any option is contained in user input
        for opt in self.strategy.static_options:
            display_lower = opt.display_value.lower()
            raw_lower = opt.raw_value.lower()

            # Check if display value is in user input
            if display_lower in user_input_lower:
                score = len(display_lower) / len(user_input_lower)  # Longer matches get higher score
                if score > best_score:
                    best_score = score
                    best_match = opt.raw_value

            # Check if raw value is in user input
            elif raw_lower in user_input_lower:
                score = len(raw_lower) / len(user_input_lower)
                if score > best_score:
                    best_score = score
                    best_match = opt.raw_value

        if best_match and best_score > 0.3:  # Lower threshold for partial matches
            return ResolverResult(
                success=True,
                resolved_value=best_match,
                raw_value=context.user_input,
                confidence=best_score,
                metadata={"match_type": "partial", "display_value": self._get_display_value(best_match)}
            )
        
        return ResolverResult(
            success=False,
            error_message=f"No matching option found for '{context.user_input}'. Available options: {list(self.reverse_map.values())}"
        )
    
    def _calculate_similarity(self, input_text: str, option_text: str) -> float:
        """Calculate similarity between input and option (simplified)."""
        if input_text == option_text:
            return 1.0
        if input_text in option_text or option_text in input_text:
            return 0.8
        # Could implement more sophisticated similarity (Levenshtein, etc.)
        return 0.0
    
    def _get_display_value(self, raw_value: str) -> str:
        """Get display value for raw value."""
        return self.reverse_map.get(raw_value, raw_value)


class DynamicResolver:
    """Dynamic resolver using API calls."""
    
    def __init__(self, strategy: ResolverStrategy):
        self.strategy = strategy
    
    async def resolve(self, context: ResolverExecutionContext) -> ResolverResult:
        """Resolve user input via API call."""
        try:
            # Get data type info
            data_type = data_type_registry.get_data_type(self.strategy.output_data_type)
            if not data_type:
                return ResolverResult(
                    success=False,
                    error_message=f"Unknown data type: {self.strategy.output_data_type}"
                )
            
            # Perform API search
            search_results = await self._perform_api_search(context, data_type)
            
            if not search_results:
                return ResolverResult(
                    success=False,
                    error_message=f"No results found for '{context.user_input}'"
                )
            
            # Return best match
            best_match = search_results[0]  # Assume sorted by relevance
            
            return ResolverResult(
                success=True,
                resolved_value=best_match,
                raw_value=context.user_input,
                confidence=0.9,  # Could be calculated based on search score
                metadata={
                    "search_results_count": len(search_results),
                    "data_type": self.strategy.output_data_type
                }
            )
            
        except Exception as e:
            return ResolverResult(
                success=False,
                error_message=f"Dynamic resolver error: {str(e)}"
            )
    
    async def _perform_api_search(self, context: ResolverExecutionContext, data_type) -> List[Dict[str, Any]]:
        """Perform API search for data type."""
        # This would integrate with actual APIs
        # For now, return mock data based on data type
        
        mock_results = {
            "u_JiraIssue": [
                {
                    "id": "10001",
                    "key": "BUG-123",
                    "summary": f"Issue matching '{context.user_input}'",
                    "status": "Open",
                    "priority": "High"
                }
            ],
            "u_ServiceNowTicket": [
                {
                    "sys_id": "abc123",
                    "number": "INC0000123",
                    "short_description": f"Ticket for '{context.user_input}'",
                    "state": "New"
                }
            ],
            "u_Employee": [
                {
                    "employee_id": "EMP001",
                    "email": f"{context.user_input.lower().replace(' ', '.')}@company.com",
                    "first_name": context.user_input.split()[0] if ' ' in context.user_input else context.user_input,
                    "last_name": context.user_input.split()[-1] if ' ' in context.user_input else "Doe",
                    "display_name": context.user_input
                }
            ]
        }
        
        return mock_results.get(self.strategy.output_data_type, [])


class BuiltInUserResolver:
    """Built-in resolver for User data type."""
    
    async def resolve(self, context: ResolverExecutionContext) -> ResolverResult:
        """Resolve user input to User object."""
        user_input = context.user_input.strip()
        
        # Try email format
        if '@' in user_input:
            user_data = {
                "employee_id": f"EMP_{user_input.split('@')[0].upper()}",
                "email": user_input,
                "display_name": user_input.split('@')[0].replace('.', ' ').title(),
                "status": "active"
            }
            
            return ResolverResult(
                success=True,
                resolved_value=user_data,
                raw_value=user_input,
                confidence=0.9,
                metadata={"resolution_method": "email_format"}
            )
        
        # Try name format
        if ' ' in user_input:
            parts = user_input.split()
            email = f"{parts[0].lower()}.{parts[-1].lower()}@company.com"
            user_data = {
                "employee_id": f"EMP_{parts[0].upper()}{parts[-1].upper()}",
                "email": email,
                "first_name": parts[0],
                "last_name": parts[-1],
                "display_name": user_input,
                "status": "active"
            }
            
            return ResolverResult(
                success=True,
                resolved_value=user_data,
                raw_value=user_input,
                confidence=0.8,
                metadata={"resolution_method": "name_format"}
            )
        
        return ResolverResult(
            success=False,
            error_message=f"Cannot resolve user: '{user_input}'. Expected email or 'First Last' format."
        )


class MoveworksResolverEngine:
    """Main resolver engine for Moveworks slot resolution."""
    
    def __init__(self):
        self.resolvers: Dict[str, Any] = {}
        self.built_in_resolvers = {
            "User": BuiltInUserResolver()
        }
    
    def register_resolver(self, strategy: ResolverStrategy):
        """Register a resolver strategy."""
        if strategy.method_type == ResolverMethodType.STATIC:
            resolver = StaticResolver(strategy)
        elif strategy.method_type == ResolverMethodType.DYNAMIC:
            resolver = DynamicResolver(strategy)
        else:
            raise ValueError(f"Unsupported resolver type: {strategy.method_type}")
        
        self.resolvers[strategy.method_name] = resolver
    
    async def resolve_slot_value(self, slot_name: str, user_input: str, 
                                data_type: str, resolver_strategy: Optional[ResolverStrategy],
                                conversation_context: MoveworksConversationContext) -> SlotResolutionResult:
        """
        Resolve slot value using appropriate resolver strategy.
        
        Args:
            slot_name: Name of the slot being resolved
            user_input: User's natural language input
            data_type: Expected data type (string, number, boolean, User, u_CustomType)
            resolver_strategy: Resolver strategy to use (if any)
            conversation_context: Current conversation context
        
        Returns:
            SlotResolutionResult with resolution outcome
        """
        context = ResolverExecutionContext(
            user_input=user_input,
            conversation_context=conversation_context,
            slot_name=slot_name
        )
        
        try:
            # Handle built-in types
            if data_type in ["string", "number", "integer", "boolean"]:
                return await self._resolve_builtin_type(data_type, context)
            
            # Handle User type
            if data_type == "User":
                resolver = self.built_in_resolvers["User"]
                result = await resolver.resolve(context)
                return self._create_slot_result(slot_name, result, "built_in_user_resolver")
            
            # Handle custom types with resolver strategies
            if data_type.startswith("u_") and resolver_strategy:
                if resolver_strategy.method_name not in self.resolvers:
                    self.register_resolver(resolver_strategy)
                
                resolver = self.resolvers[resolver_strategy.method_name]
                result = await resolver.resolve(context)
                return self._create_slot_result(slot_name, result, resolver_strategy.method_name)
            
            # Fallback: treat as string
            return SlotResolutionResult(
                slot_name=slot_name,
                resolved_value=user_input,
                raw_value=user_input,
                confidence=0.5,
                resolution_method="fallback_string"
            )
            
        except Exception as e:
            return SlotResolutionResult(
                slot_name=slot_name,
                resolved_value=None,
                raw_value=user_input,
                confidence=0.0,
                validation_passed=False,
                validation_errors=[f"Resolver error: {str(e)}"],
                resolution_method="error"
            )
    
    async def _resolve_builtin_type(self, data_type: str, context: ResolverExecutionContext) -> SlotResolutionResult:
        """Resolve built-in data types."""
        user_input = context.user_input.strip()

        try:
            if data_type == "string":
                resolved_value = user_input
            elif data_type == "number":
                # Extract number from text
                import re
                numbers = re.findall(r'-?\d+\.?\d*', user_input)
                if numbers:
                    resolved_value = float(numbers[0])
                else:
                    raise ValueError(f"No number found in '{user_input}'")
            elif data_type == "integer":
                # Extract integer from text
                import re
                numbers = re.findall(r'-?\d+', user_input)
                if numbers:
                    resolved_value = int(numbers[0])
                else:
                    raise ValueError(f"No integer found in '{user_input}'")
            elif data_type == "boolean":
                resolved_value = user_input.lower() in ["true", "yes", "1", "on", "enabled"]
            else:
                raise ValueError(f"Unknown built-in type: {data_type}")

            return SlotResolutionResult(
                slot_name=context.slot_name,
                resolved_value=resolved_value,
                raw_value=user_input,
                confidence=1.0,
                resolution_method=f"builtin_{data_type}"
            )

        except (ValueError, TypeError) as e:
            return SlotResolutionResult(
                slot_name=context.slot_name,
                resolved_value=None,
                raw_value=user_input,
                confidence=0.0,
                validation_passed=False,
                validation_errors=[f"Type conversion error: {str(e)}"],
                resolution_method=f"builtin_{data_type}_error"
            )
    
    def _create_slot_result(self, slot_name: str, resolver_result: ResolverResult, 
                           resolver_name: str) -> SlotResolutionResult:
        """Convert ResolverResult to SlotResolutionResult."""
        return SlotResolutionResult(
            slot_name=slot_name,
            resolved_value=resolver_result.resolved_value if resolver_result.success else None,
            raw_value=resolver_result.raw_value,
            confidence=resolver_result.confidence,
            validation_passed=resolver_result.success,
            validation_errors=[resolver_result.error_message] if not resolver_result.success else [],
            resolver_used=resolver_name
        )


# Global resolver engine instance
resolver_engine = MoveworksResolverEngine()

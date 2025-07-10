"""
Moveworks-style Slot Resolver Engine.
Implements Static, API, and Inline resolver strategies following Moveworks patterns.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from ..models.moveworks import (
    Slot, ResolverStrategy, ResolverMethod, ResolverMethodType,
    ConversationContext, StaticOption
)
from ..llm.llm_factory import LLMFactory
from ..actions.http_actions import HTTPActionExecutor

logger = logging.getLogger(__name__)


class ResolverType(Enum):
    """Types of slot resolvers."""
    STATIC = "static"
    API = "api"
    INLINE = "inline"


@dataclass
class ResolverResult:
    """Result of slot resolution."""
    success: bool
    resolved_value: Any = None
    display_value: str = ""
    raw_value: Any = None
    confidence: float = 0.0
    reasoning: str = ""
    options: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class StaticResolverConfig:
    """Configuration for static resolver."""
    options: List[StaticOption]
    allow_custom: bool = False
    case_sensitive: bool = False


@dataclass
class APIResolverConfig:
    """Configuration for API resolver."""
    endpoint: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body_template: Optional[str] = None
    response_path: str = ""
    value_field: str = "value"
    display_field: str = "display"


@dataclass
class InlineResolverConfig:
    """Configuration for inline resolver."""
    resolver_function: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    llm_prompt: Optional[str] = None


class MoveworksSlotResolverEngine:
    """
    Moveworks-style Slot Resolver Engine implementing the three resolver strategies:
    1. Static Resolvers - Predefined options
    2. API Resolvers - Dynamic data from APIs
    3. Inline Resolvers - Custom logic and LLM-powered resolution
    """
    
    def __init__(self):
        """Initialize the slot resolver engine."""
        self.llm = None
        self.http_executor = HTTPActionExecutor()
        self.static_resolvers: Dict[str, StaticResolverConfig] = {}
        self.api_resolvers: Dict[str, APIResolverConfig] = {}
        self.inline_resolvers: Dict[str, InlineResolverConfig] = {}
        self.resolution_cache: Dict[str, ResolverResult] = {}
        
    async def initialize(self):
        """Initialize the resolver engine."""
        self.llm = LLMFactory.get_fast_llm()
        await self.http_executor.initialize()
        logger.info("Slot resolver engine initialized")
    
    async def resolve_slot(
        self,
        slot: Slot,
        user_input: str,
        context: ConversationContext,
        process_data: Dict[str, Any]
    ) -> ResolverResult:
        """
        Resolve a slot value using the appropriate resolver strategy.
        
        Args:
            slot: Slot definition with resolver strategy
            user_input: User's natural language input
            context: Conversation context
            process_data: Current process data
            
        Returns:
            ResolverResult with resolved value and metadata
        """
        if not slot.resolver_strategy:
            return ResolverResult(
                success=False,
                error_message="No resolver strategy defined for slot"
            )
        
        # Check cache first
        cache_key = f"{slot.name}:{user_input}:{hash(str(context.user_attributes))}"
        if cache_key in self.resolution_cache:
            logger.debug(f"Using cached resolution for slot: {slot.name}")
            return self.resolution_cache[cache_key]
        
        try:
            # Determine resolver type and execute
            resolver_strategy = slot.resolver_strategy
            
            if not resolver_strategy.methods:
                return ResolverResult(
                    success=False,
                    error_message="No resolver methods defined"
                )
            
            # For now, use the first method (can be enhanced with method selection)
            method = resolver_strategy.methods[0]
            
            if method.method_type == ResolverMethodType.STATIC:
                result = await self._resolve_static(slot, user_input, method, context)
            elif method.method_type == ResolverMethodType.API:
                result = await self._resolve_api(slot, user_input, method, context, process_data)
            elif method.method_type == ResolverMethodType.INLINE:
                result = await self._resolve_inline(slot, user_input, method, context, process_data)
            else:
                result = ResolverResult(
                    success=False,
                    error_message=f"Unknown resolver method type: {method.method_type}"
                )
            
            # Cache successful results
            if result.success:
                self.resolution_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error resolving slot {slot.name}: {e}")
            return ResolverResult(
                success=False,
                error_message=f"Slot resolution failed: {str(e)}"
            )
    
    async def _resolve_static(
        self,
        slot: Slot,
        user_input: str,
        method: ResolverMethod,
        context: ConversationContext
    ) -> ResolverResult:
        """Resolve slot using static options."""
        
        if not method.static_options:
            return ResolverResult(
                success=False,
                error_message="No static options defined"
            )
        
        user_input_lower = user_input.lower().strip()
        best_match = None
        best_score = 0.0
        
        # Find best matching option
        for option in method.static_options:
            display_lower = option.display_value.lower()
            raw_lower = str(option.raw_value).lower()
            
            # Exact match gets highest score
            if user_input_lower == display_lower or user_input_lower == raw_lower:
                best_match = option
                best_score = 1.0
                break
            
            # Partial match scoring
            if user_input_lower in display_lower or display_lower in user_input_lower:
                score = len(user_input_lower) / max(len(display_lower), len(user_input_lower))
                if score > best_score:
                    best_match = option
                    best_score = score
            
            # Check if raw value matches
            if user_input_lower in raw_lower or raw_lower in user_input_lower:
                score = len(user_input_lower) / max(len(raw_lower), len(user_input_lower))
                if score > best_score:
                    best_match = option
                    best_score = score
        
        if best_match and best_score >= 0.5:  # Minimum threshold
            return ResolverResult(
                success=True,
                resolved_value=best_match.raw_value,
                display_value=best_match.display_value,
                raw_value=best_match.raw_value,
                confidence=best_score,
                reasoning=f"Matched static option: {best_match.display_value}",
                options=[{
                    "display": opt.display_value,
                    "value": opt.raw_value
                } for opt in method.static_options]
            )
        else:
            # Return available options for user to choose from
            return ResolverResult(
                success=False,
                error_message=f"No matching option found. Available options: {', '.join([opt.display_value for opt in method.static_options])}",
                options=[{
                    "display": opt.display_value,
                    "value": opt.raw_value
                } for opt in method.static_options]
            )
    
    async def _resolve_api(
        self,
        slot: Slot,
        user_input: str,
        method: ResolverMethod,
        context: ConversationContext,
        process_data: Dict[str, Any]
    ) -> ResolverResult:
        """Resolve slot using API call."""
        
        if not method.action_name:
            return ResolverResult(
                success=False,
                error_message="No action defined for API resolver"
            )
        
        try:
            # Prepare input arguments for API call
            input_args = {}
            
            # Add user input
            input_args["query"] = user_input
            input_args["user_input"] = user_input
            
            # Add user context
            if context.user_attributes:
                input_args.update(context.user_attributes)
            
            # Add process data
            input_args.update(process_data)
            
            # Execute API action
            api_result = await self.http_executor.execute_action(
                method.action_name,
                input_args
            )
            
            if not api_result.get("success", False):
                return ResolverResult(
                    success=False,
                    error_message=f"API call failed: {api_result.get('error', 'Unknown error')}"
                )
            
            # Extract data from API response
            response_data = api_result.get("data", {})
            
            # If response is a list, use it as options
            if isinstance(response_data, list):
                options = []
                for item in response_data:
                    if isinstance(item, dict):
                        display = item.get("display", item.get("name", str(item.get("id", item))))
                        value = item.get("value", item.get("id", item))
                        options.append({"display": display, "value": value})
                    else:
                        options.append({"display": str(item), "value": item})
                
                # Try to find best match
                if user_input and options:
                    best_match = await self._find_best_api_match(user_input, options)
                    if best_match:
                        return ResolverResult(
                            success=True,
                            resolved_value=best_match["value"],
                            display_value=best_match["display"],
                            raw_value=best_match["value"],
                            confidence=0.8,
                            reasoning=f"Matched API result: {best_match['display']}",
                            options=options
                        )
                
                return ResolverResult(
                    success=False,
                    error_message="Multiple options available, please specify",
                    options=options
                )
            
            # Single result
            else:
                display = response_data.get("display", response_data.get("name", str(response_data)))
                value = response_data.get("value", response_data.get("id", response_data))
                
                return ResolverResult(
                    success=True,
                    resolved_value=value,
                    display_value=str(display),
                    raw_value=value,
                    confidence=0.9,
                    reasoning="API resolved single result"
                )
                
        except Exception as e:
            logger.error(f"Error in API resolver: {e}")
            return ResolverResult(
                success=False,
                error_message=f"API resolver error: {str(e)}"
            )
    
    async def _resolve_inline(
        self,
        slot: Slot,
        user_input: str,
        method: ResolverMethod,
        context: ConversationContext,
        process_data: Dict[str, Any]
    ) -> ResolverResult:
        """Resolve slot using inline/custom logic."""
        
        # For now, implement LLM-powered inline resolution
        # This can be extended with custom function execution
        
        try:
            # Build context for LLM
            context_parts = [
                f"Slot: {slot.name}",
                f"Description: {slot.slot_description}",
                f"User Input: {user_input}",
                f"Data Type: {slot.data_type}"
            ]
            
            if context.user_attributes:
                context_parts.append(f"User Context: {json.dumps(context.user_attributes)}")
            
            if process_data:
                context_parts.append(f"Process Data: {json.dumps(process_data)}")
            
            context_text = "\n".join(context_parts)
            
            # Create resolution prompt
            prompt = f"""Resolve the slot value from user input using intelligent reasoning.

{context_text}

Extract and resolve the appropriate value for this slot. Consider:
1. The slot's purpose and data type
2. User's natural language input
3. Context and available data

Respond with JSON:
{{
    "resolved_value": "the extracted/resolved value",
    "display_value": "human-readable display value",
    "confidence": 0.85,
    "reasoning": "explanation of resolution"
}}

If you cannot resolve the value, set resolved_value to null and explain why."""
            
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse LLM response
            resolution_data = json.loads(response_text)
            
            resolved_value = resolution_data.get("resolved_value")
            if resolved_value is not None:
                return ResolverResult(
                    success=True,
                    resolved_value=resolved_value,
                    display_value=resolution_data.get("display_value", str(resolved_value)),
                    raw_value=resolved_value,
                    confidence=resolution_data.get("confidence", 0.7),
                    reasoning=resolution_data.get("reasoning", "LLM-powered resolution")
                )
            else:
                return ResolverResult(
                    success=False,
                    error_message=resolution_data.get("reasoning", "Could not resolve slot value"),
                    confidence=resolution_data.get("confidence", 0.0)
                )
                
        except Exception as e:
            logger.error(f"Error in inline resolver: {e}")
            return ResolverResult(
                success=False,
                error_message=f"Inline resolver error: {str(e)}"
            )

    async def _find_best_api_match(
        self,
        user_input: str,
        options: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find the best matching option from API results."""

        user_input_lower = user_input.lower().strip()
        best_match = None
        best_score = 0.0

        for option in options:
            display = str(option.get("display", "")).lower()
            value = str(option.get("value", "")).lower()

            # Exact match
            if user_input_lower == display or user_input_lower == value:
                return option

            # Partial match scoring
            if user_input_lower in display or display in user_input_lower:
                score = len(user_input_lower) / max(len(display), len(user_input_lower))
                if score > best_score:
                    best_match = option
                    best_score = score

            if user_input_lower in value or value in user_input_lower:
                score = len(user_input_lower) / max(len(value), len(user_input_lower))
                if score > best_score:
                    best_match = option
                    best_score = score

        # Return match if confidence is high enough
        return best_match if best_score >= 0.6 else None

    def clear_cache(self):
        """Clear resolution cache."""
        self.resolution_cache.clear()
        logger.info("Slot resolver cache cleared")

    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics data for slot resolution."""
        return {
            "cache_size": len(self.resolution_cache),
            "total_resolutions": len(self.resolution_cache)
        }

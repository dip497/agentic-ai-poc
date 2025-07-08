"""
Moveworks AI-Powered Plugin Selection Engine.
Implements autonomous plugin selection based on descriptions, examples, and domain compatibility.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from models.moveworks import Plugin, ConversationalProcess
from config.loader import MoveworksConfigLoader
from llm.llm_factory import LLMFactory
from .memory_constructs import MemorySnapshot

logger = logging.getLogger(__name__)


class PluginSelectionResult:
    """Result of plugin selection with confidence and reasoning."""
    
    def __init__(
        self,
        plugin: Plugin,
        confidence: float,
        reasoning: str,
        matching_process: Optional[ConversationalProcess] = None,
        selection_method: str = "ai_powered"
    ):
        self.plugin = plugin
        self.confidence = confidence
        self.reasoning = reasoning
        self.matching_process = matching_process
        self.selection_method = selection_method
        self.timestamp = datetime.now()


class MoveworksPluginSelector:
    """
    AI-powered plugin selection engine following Moveworks patterns.
    Selects plugins based on descriptions, examples, domain compatibility, and user context.
    """
    
    def __init__(self, config_loader: MoveworksConfigLoader):
        self.config_loader = config_loader
        self.llm = None
        self.plugins: Dict[str, Plugin] = {}
        self.selection_cache: Dict[str, List[PluginSelectionResult]] = {}

    async def initialize(self):
        """Initialize the plugin selector with LLM and load plugins."""
        # Initialize LLM using centralized configuration
        self.llm = LLMFactory.get_fast_llm()

        # Load plugins from configuration
        await self._load_plugins()

        logger.info(f"Plugin selector initialized with {len(self.plugins)} plugins")
    
    async def _load_plugins(self):
        """Load plugins from configuration and enhance with metadata."""
        # Load plugins using the config loader
        plugins_list = self.config_loader.load_plugins()

        for plugin in plugins_list:
            # Enhance plugin with derived metadata if not provided
            if not plugin.capabilities:
                plugin.capabilities = await self._derive_capabilities(plugin)

            if not plugin.domain_compatibility:
                plugin.domain_compatibility = await self._derive_domain_compatibility(plugin)

            if not plugin.positive_examples:
                plugin.positive_examples = self._extract_trigger_utterances(plugin)

            self.plugins[plugin.name] = plugin
    
    async def select_plugins(
        self,
        user_query: str,
        user_context: Dict[str, Any],
        memory_snapshot: Optional[MemorySnapshot] = None,
        max_plugins: int = 3
    ) -> List[PluginSelectionResult]:
        """
        Select the most appropriate plugins for a user query.
        
        Args:
            user_query: The user's input query
            user_context: User attributes and context
            memory_snapshot: Memory context from conversation
            max_plugins: Maximum number of plugins to return
            
        Returns:
            List of PluginSelectionResult objects ranked by confidence
        """
        # Check cache first
        cache_key = f"{user_query}:{hash(str(user_context))}"
        if cache_key in self.selection_cache:
            logger.debug(f"Using cached plugin selection for query: {user_query[:50]}...")
            return self.selection_cache[cache_key][:max_plugins]
        
        # Get plugin candidates
        candidates = await self._get_plugin_candidates(user_query, user_context, memory_snapshot)
        
        # Score and rank candidates
        scored_candidates = []
        for plugin in candidates:
            confidence, reasoning = await self._score_plugin(user_query, plugin, user_context, memory_snapshot)
            
            if confidence >= plugin.confidence_threshold:
                # Find best matching process
                matching_process = await self._find_best_process(user_query, plugin, user_context)
                
                result = PluginSelectionResult(
                    plugin=plugin,
                    confidence=confidence,
                    reasoning=reasoning,
                    matching_process=matching_process,
                    selection_method="ai_powered"
                )
                scored_candidates.append(result)
        
        # Sort by confidence and apply business rules
        scored_candidates.sort(key=lambda x: x.confidence, reverse=True)
        final_results = await self._apply_selection_rules(scored_candidates, user_context, max_plugins)
        
        # Cache results
        self.selection_cache[cache_key] = final_results
        
        # Log selection for analytics
        await self._log_selection(user_query, final_results, user_context)
        
        return final_results[:max_plugins]
    
    async def _get_plugin_candidates(
        self,
        user_query: str,
        user_context: Dict[str, Any],
        memory_snapshot: Optional[MemorySnapshot]
    ) -> List[Plugin]:
        """Get initial plugin candidates based on basic filtering."""
        candidates = []
        
        for plugin in self.plugins.values():
            # Check access permissions
            if not await self._check_plugin_access(plugin, user_context):
                continue
            
            # Check domain compatibility if user has domain context
            user_domain = user_context.get("domain") or user_context.get("department", "").lower()
            if user_domain and plugin.domain_compatibility:
                if not any(domain.lower() in user_domain or user_domain in domain.lower() 
                          for domain in plugin.domain_compatibility):
                    continue
            
            # Basic keyword matching for initial filtering
            if await self._has_keyword_match(user_query, plugin):
                candidates.append(plugin)
        
        # If no candidates found, include all accessible plugins for AI evaluation
        if not candidates:
            candidates = [p for p in self.plugins.values() 
                         if await self._check_plugin_access(p, user_context)]
        
        return candidates
    
    async def _score_plugin(
        self,
        user_query: str,
        plugin: Plugin,
        user_context: Dict[str, Any],
        memory_snapshot: Optional[MemorySnapshot]
    ) -> Tuple[float, str]:
        """Score a plugin's relevance to the user query using LLM."""

        # Build context for LLM
        context_parts = [
            f"User Query: {user_query}",
            f"Plugin Name: {plugin.name}",
            f"Plugin Description: {plugin.description}",
            f"Plugin Capabilities: {', '.join(plugin.capabilities)}",
            f"Domain Compatibility: {', '.join(plugin.domain_compatibility)}"
        ]

        if plugin.positive_examples:
            context_parts.append(f"Positive Examples: {'; '.join(plugin.positive_examples[:3])}")

        if plugin.negative_examples:
            context_parts.append(f"Negative Examples: {'; '.join(plugin.negative_examples[:3])}")

        # Add user context
        if user_context.get("department"):
            context_parts.append(f"User Department: {user_context['department']}")

        if user_context.get("role"):
            context_parts.append(f"User Role: {user_context['role']}")

        context = "\n".join(context_parts)

        # Create scoring prompt
        prompt = f"""Analyze if this plugin is relevant for the user query.

{context}

Rate relevance from 0.0 to 1.0 and provide brief reasoning.
Respond with only: confidence_score|reasoning

Example: 0.85|Plugin capabilities match user intent for PTO management"""

        try:
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse response - must be in format "confidence|reasoning"
            if "|" not in response_text:
                raise ValueError(f"LLM response not in expected format 'confidence|reasoning': {response_text}")

            confidence_str, reasoning = response_text.split("|", 1)
            confidence = float(confidence_str.strip())
            reasoning = reasoning.strip()

            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence score {confidence} not in valid range 0.0-1.0")

            # Apply historical success rate adjustment
            if plugin.success_rate > 0:
                confidence = confidence * (0.7 + 0.3 * plugin.success_rate)

            return min(confidence, 1.0), reasoning

        except Exception as e:
            logger.error(f"Error scoring plugin {plugin.name}: {e}")
            # Don't fallback - if LLM fails, the system should fail
            raise RuntimeError(f"Plugin scoring failed for {plugin.name}: {e}") from e
    
    async def _derive_capabilities(self, plugin: Plugin) -> List[str]:
        """Derive plugin capabilities from its processes and description."""
        capabilities = set()
        
        # Extract from process descriptions
        for process in plugin.conversational_processes:
            # Add process title as capability
            capabilities.add(process.title.replace("_", " ").replace("process", "").strip())
            
            # Extract action types
            for activity in process.activities:
                if activity.activity_type == "action":
                    action_name = activity.action_name.replace("_", " ").replace("action", "").strip()
                    capabilities.add(action_name)
        
        return list(capabilities)
    
    async def _derive_domain_compatibility(self, plugin: Plugin) -> List[str]:
        """Derive domain compatibility from plugin name and description."""
        domains = []
        
        text = f"{plugin.name} {plugin.description}".lower()
        
        # Common domain keywords
        domain_keywords = {
            "hr": ["pto", "vacation", "leave", "employee", "hr", "human resources"],
            "it": ["ticket", "support", "technical", "system", "password", "access"],
            "finance": ["purchase", "procurement", "expense", "budget", "cost"],
            "project": ["feature", "request", "development", "project", "task"],
            "sales": ["lead", "opportunity", "customer", "account", "crm"],
            "marketing": ["campaign", "content", "social", "analytics"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                domains.append(domain)
        
        return domains if domains else ["general"]
    
    def _extract_trigger_utterances(self, plugin: Plugin) -> List[str]:
        """Extract trigger utterances as positive examples."""
        examples = []
        for process in plugin.conversational_processes:
            examples.extend(process.trigger_utterances)
        return examples

    async def _check_plugin_access(self, plugin: Plugin, user_context: Dict[str, Any]) -> bool:
        """Check if user has access to this plugin."""
        if not plugin.access_policies:
            return True

        # For now, implement basic access control
        # TODO: Integrate with actual access policy system
        user_role = user_context.get("role", "").lower()
        user_dept = user_context.get("department", "").lower()

        # Basic role-based access
        if "admin" in user_role or "manager" in user_role:
            return True

        # Department-based access for specific plugins
        if "pto" in plugin.name.lower() and user_dept in ["hr", "engineering", "all"]:
            return True

        if "procurement" in plugin.name.lower() and user_dept in ["finance", "procurement", "all"]:
            return True

        if "feature" in plugin.name.lower() and user_dept in ["engineering", "product", "all"]:
            return True

        return True  # Default allow for now

    async def _has_keyword_match(self, user_query: str, plugin: Plugin) -> bool:
        """Check for basic keyword matches."""
        query_lower = user_query.lower()

        # Check plugin name and description
        if plugin.name.lower() in query_lower or any(
            word in query_lower for word in plugin.description.lower().split()
        ):
            return True

        # Check capabilities
        if any(cap.lower() in query_lower for cap in plugin.capabilities):
            return True

        # Check trigger utterances
        for process in plugin.conversational_processes:
            for trigger in process.trigger_utterances:
                if any(word in query_lower for word in trigger.lower().split() if len(word) > 3):
                    return True

        return False

    async def _find_best_process(
        self,
        user_query: str,
        plugin: Plugin,
        user_context: Dict[str, Any]
    ) -> Optional[ConversationalProcess]:
        """Find the best matching process within a plugin."""
        if len(plugin.conversational_processes) == 1:
            return plugin.conversational_processes[0]

        best_process = None
        best_score = 0.0

        for process in plugin.conversational_processes:
            score = 0.0

            # Score based on trigger utterances
            for trigger in process.trigger_utterances:
                if trigger.lower() in user_query.lower():
                    score += len(trigger) / len(user_query)  # Longer matches get higher scores

            # Score based on description similarity
            if process.description and user_query:
                # Simple word overlap scoring
                process_words = set(process.description.lower().split())
                query_words = set(user_query.lower().split())
                overlap = len(process_words.intersection(query_words))
                score += overlap / max(len(process_words), len(query_words))

            if score > best_score:
                best_score = score
                best_process = process

        return best_process

    async def _apply_selection_rules(
        self,
        candidates: List[PluginSelectionResult],
        user_context: Dict[str, Any],
        max_plugins: int
    ) -> List[PluginSelectionResult]:
        """Apply business rules to final plugin selection."""
        if not candidates:
            return []

        # Rule 1: If there's a high-confidence match (>0.9), prefer it
        high_confidence = [c for c in candidates if c.confidence > 0.9]
        if high_confidence:
            return high_confidence[:max_plugins]

        # Rule 2: Prefer plugins from user's department
        user_dept = user_context.get("department", "").lower()
        if user_dept:
            dept_matches = [
                c for c in candidates
                if user_dept in [d.lower() for d in c.plugin.domain_compatibility]
            ]
            if dept_matches:
                return dept_matches[:max_plugins]

        # Rule 3: Return top candidates by confidence
        return candidates[:max_plugins]

    async def _log_selection(
        self,
        user_query: str,
        results: List[PluginSelectionResult],
        user_context: Dict[str, Any]
    ):
        """Log plugin selection for analytics and improvement."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "user_context": {
                "department": user_context.get("department"),
                "role": user_context.get("role")
            },
            "selected_plugins": [
                {
                    "name": result.plugin.name,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "process": result.matching_process.title if result.matching_process else None
                }
                for result in results
            ]
        }

        logger.info(f"Plugin selection: {json.dumps(log_data, indent=2)}")

    async def update_plugin_performance(self, plugin_name: str, success: bool):
        """Update plugin performance metrics based on execution results."""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]

            # Update usage stats
            if "total_uses" not in plugin.usage_stats:
                plugin.usage_stats["total_uses"] = 0
                plugin.usage_stats["successful_uses"] = 0

            plugin.usage_stats["total_uses"] += 1
            if success:
                plugin.usage_stats["successful_uses"] += 1

            # Update success rate
            plugin.success_rate = (
                plugin.usage_stats["successful_uses"] / plugin.usage_stats["total_uses"]
            )

            # Clear cache to force re-evaluation
            self.selection_cache.clear()

            logger.debug(f"Updated performance for {plugin_name}: success_rate={plugin.success_rate:.2f}")

    def get_plugin_analytics(self) -> Dict[str, Any]:
        """Get analytics data for all plugins."""
        analytics = {
            "total_plugins": len(self.plugins),
            "plugins": {}
        }

        for name, plugin in self.plugins.items():
            analytics["plugins"][name] = {
                "capabilities": plugin.capabilities,
                "domain_compatibility": plugin.domain_compatibility,
                "confidence_threshold": plugin.confidence_threshold,
                "success_rate": plugin.success_rate,
                "usage_stats": plugin.usage_stats,
                "processes_count": len(plugin.conversational_processes)
            }

        return analytics

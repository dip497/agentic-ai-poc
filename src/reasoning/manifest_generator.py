"""
Moveworks-style Manifest Generator for Autonomous Plugin Selection.
Implements sophisticated reasoning for plugin selection, capability matching, and multi-plugin coordination.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from models.moveworks import Plugin, ConversationalProcess, ConversationContext
from llm.llm_factory import LLMFactory
from reasoning.memory_constructs import MemorySnapshot
from reasoning.plugin_selection_engine import PluginSelectionResult
from vector_store.plugin_similarity_search import MoveworksPluginSimilaritySearch, PluginSimilarityResult

logger = logging.getLogger(__name__)


@dataclass
class IntentAnalysis:
    """Result of user intent analysis."""
    primary_intent: str
    secondary_intents: List[str] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    urgency_level: str = "normal"  # low, normal, high, critical
    complexity: str = "simple"  # simple, moderate, complex
    domain: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class CapabilityMatch:
    """Result of capability matching."""
    plugin: Plugin
    matched_capabilities: List[str]
    match_score: float
    reasoning: str
    process: Optional[ConversationalProcess] = None


@dataclass
class ExecutionPlan:
    """Multi-plugin execution plan."""
    primary_plugin: Plugin
    supporting_plugins: List[Plugin] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    coordination_strategy: str = "sequential"  # sequential, parallel, conditional
    fallback_plugins: List[Plugin] = field(default_factory=list)
    estimated_duration: float = 0.0
    confidence: float = 0.0


@dataclass
class ManifestResult:
    """Complete manifest generation result."""
    intent_analysis: IntentAnalysis
    capability_matches: List[CapabilityMatch]
    execution_plan: ExecutionPlan
    selected_plugins: List[PluginSelectionResult]
    reasoning_trace: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class MoveworksManifestGenerator:
    """
    Moveworks-style Manifest Generator implementing autonomous plugin selection
    with sophisticated reasoning, capability matching, and multi-plugin coordination.
    """
    
    def __init__(self, plugin_selector=None, plugin_similarity_search=None):
        """Initialize the manifest generator."""
        self.llm = None
        self.plugin_selector = plugin_selector
        self.plugin_similarity_search = plugin_similarity_search
        self.intent_cache: Dict[str, IntentAnalysis] = {}
        self.capability_cache: Dict[str, List[CapabilityMatch]] = {}
        
    async def initialize(self):
        """Initialize the manifest generator with LLM and plugin similarity search."""
        self.llm = LLMFactory.get_reasoning_llm()

        # Initialize plugin similarity search if not provided
        if not self.plugin_similarity_search:
            self.plugin_similarity_search = MoveworksPluginSimilaritySearch()
            await self.plugin_similarity_search.initialize()

        logger.info("Manifest generator initialized")
    
    async def generate_manifest(
        self,
        user_intent: str,
        context: ConversationContext,
        available_plugins: List[Plugin],
        memory_snapshot: Optional[MemorySnapshot] = None,
        max_plugins: int = 3
    ) -> ManifestResult:
        """
        Generate a complete execution manifest for user intent.
        
        Args:
            user_intent: User's natural language input
            context: Conversation context
            available_plugins: Available plugins to choose from
            memory_snapshot: Memory context from conversation
            max_plugins: Maximum number of plugins to select
            
        Returns:
            ManifestResult with complete execution plan
        """
        reasoning_trace = []
        
        try:
            # Step 1: Analyze user intent
            reasoning_trace.append("Starting intent analysis...")
            intent_analysis = await self._analyze_user_intent(user_intent, context, memory_snapshot)
            reasoning_trace.append(f"Intent identified: {intent_analysis.primary_intent}")
            
            # Step 2: Match capabilities
            reasoning_trace.append("Matching capabilities to available plugins...")
            capability_matches = await self._match_capabilities(intent_analysis, available_plugins)
            reasoning_trace.append(f"Found {len(capability_matches)} capability matches")
            
            # Step 3: Resolve plugin competition
            reasoning_trace.append("Resolving plugin competition...")
            selected_plugins = await self._resolve_plugin_competition(
                capability_matches, context, max_plugins
            )
            reasoning_trace.append(f"Selected {len(selected_plugins)} plugins after competition resolution")
            
            # Step 4: Plan multi-plugin execution
            reasoning_trace.append("Planning multi-plugin execution...")
            execution_plan = await self._plan_multi_plugin_execution(
                selected_plugins, intent_analysis, context
            )
            reasoning_trace.append(f"Execution plan: {execution_plan.coordination_strategy}")
            
            return ManifestResult(
                intent_analysis=intent_analysis,
                capability_matches=capability_matches,
                execution_plan=execution_plan,
                selected_plugins=selected_plugins,
                reasoning_trace=reasoning_trace
            )
            
        except Exception as e:
            logger.error(f"Error generating manifest: {e}")
            reasoning_trace.append(f"Error: {str(e)}")
            
            # Return fallback result
            return ManifestResult(
                intent_analysis=IntentAnalysis(primary_intent="unknown", reasoning="Error in analysis"),
                capability_matches=[],
                execution_plan=ExecutionPlan(
                    primary_plugin=available_plugins[0] if available_plugins else None,
                    confidence=0.0
                ),
                selected_plugins=[],
                reasoning_trace=reasoning_trace
            )
    
    async def _analyze_user_intent(
        self,
        user_input: str,
        context: ConversationContext,
        memory_snapshot: Optional[MemorySnapshot]
    ) -> IntentAnalysis:
        """Analyze user intent using sophisticated LLM reasoning."""
        
        # Check cache first
        cache_key = f"{user_input}:{hash(str(context.user_attributes))}"
        if cache_key in self.intent_cache:
            return self.intent_cache[cache_key]
        
        # Build context for LLM
        context_parts = [
            f"User Input: {user_input}",
            f"User Department: {context.user_attributes.get('department', 'unknown')}",
            f"User Role: {context.user_attributes.get('role', 'unknown')}"
        ]
        
        if memory_snapshot:
            recent_context = memory_snapshot.get_recent_context(limit=3)
            if recent_context:
                context_parts.append(f"Recent Context: {recent_context}")
        
        context_text = "\n".join(context_parts)
        
        # Create intent analysis prompt
        prompt = f"""Analyze the user's intent from their input. Provide a detailed analysis.

{context_text}

Analyze and respond with JSON in this exact format:
{{
    "primary_intent": "main goal or action the user wants",
    "secondary_intents": ["any additional goals"],
    "entities": {{"entity_type": "entity_value"}},
    "urgency_level": "low|normal|high|critical",
    "complexity": "simple|moderate|complex",
    "domain": "hr|it|finance|project|sales|marketing|general",
    "confidence": 0.85,
    "reasoning": "explanation of the analysis"
}}

Focus on understanding what the user actually wants to accomplish."""
        
        try:
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            analysis_data = json.loads(response_text)
            
            intent_analysis = IntentAnalysis(
                primary_intent=analysis_data["primary_intent"],
                secondary_intents=analysis_data.get("secondary_intents", []),
                entities=analysis_data.get("entities", {}),
                urgency_level=analysis_data.get("urgency_level", "normal"),
                complexity=analysis_data.get("complexity", "simple"),
                domain=analysis_data.get("domain"),
                confidence=analysis_data.get("confidence", 0.0),
                reasoning=analysis_data.get("reasoning", "")
            )
            
            # Cache the result
            self.intent_cache[cache_key] = intent_analysis
            
            return intent_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            # Fallback to basic analysis
            return IntentAnalysis(
                primary_intent=user_input[:50],
                confidence=0.3,
                reasoning=f"Fallback analysis due to error: {e}"
            )
    
    async def _match_capabilities(
        self,
        intent_analysis: IntentAnalysis,
        available_plugins: List[Plugin]
    ) -> List[CapabilityMatch]:
        """Match user intent to plugin capabilities using similarity search."""

        cache_key = f"{intent_analysis.primary_intent}:{len(available_plugins)}"
        if cache_key in self.capability_cache:
            return self.capability_cache[cache_key]

        matches = []

        # Use plugin similarity search if available
        if self.plugin_similarity_search:
            try:
                # First register all available plugins in similarity search
                for plugin in available_plugins:
                    await self.plugin_similarity_search.register_plugin(plugin)

                # Search for similar plugins
                similarity_results = await self.plugin_similarity_search.search_similar_plugins(
                    query=intent_analysis.primary_intent,
                    domain=intent_analysis.domain,
                    max_results=len(available_plugins),
                    similarity_threshold=0.3
                )

                # Convert similarity results to capability matches
                for sim_result in similarity_results:
                    matches.append(CapabilityMatch(
                        plugin=sim_result.plugin,
                        matched_capabilities=sim_result.capability_matches,
                        match_score=sim_result.similarity_score,
                        reasoning=sim_result.match_reasoning,
                        process=sim_result.matching_process
                    ))

            except Exception as e:
                logger.error(f"Error using plugin similarity search: {e}")
                # Fallback to traditional matching
                matches = await self._traditional_capability_matching(intent_analysis, available_plugins)
        else:
            # Fallback to traditional matching
            matches = await self._traditional_capability_matching(intent_analysis, available_plugins)

        # Cache the result
        self.capability_cache[cache_key] = matches

        return matches

    async def _traditional_capability_matching(
        self,
        intent_analysis: IntentAnalysis,
        available_plugins: List[Plugin]
    ) -> List[CapabilityMatch]:
        """Traditional LLM-based capability matching as fallback."""

        matches = []

        for plugin in available_plugins:
            # Build plugin capability description
            capabilities_text = ", ".join(plugin.capabilities) if plugin.capabilities else "general assistance"
            processes_text = "; ".join([p.description for p in plugin.conversational_processes])

            # Create capability matching prompt
            prompt = f"""Evaluate if this plugin can handle the user's intent.

User Intent: {intent_analysis.primary_intent}
User Domain: {intent_analysis.domain or 'unknown'}
Intent Complexity: {intent_analysis.complexity}

Plugin: {plugin.name}
Description: {plugin.description}
Capabilities: {capabilities_text}
Processes: {processes_text}
Domain Compatibility: {', '.join(plugin.domain_compatibility) if plugin.domain_compatibility else 'general'}

Rate the match from 0.0 to 1.0 and explain why.
Respond with JSON: {{"match_score": 0.85, "matched_capabilities": ["cap1", "cap2"], "reasoning": "explanation"}}"""

            try:
                response = await self.llm.ainvoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)

                match_data = json.loads(response_text)
                match_score = match_data.get("match_score", 0.0)

                if match_score >= 0.3:  # Threshold for consideration
                    # Find best matching process
                    best_process = None
                    if plugin.conversational_processes:
                        best_process = plugin.conversational_processes[0]  # Simplified for now

                    matches.append(CapabilityMatch(
                        plugin=plugin,
                        matched_capabilities=match_data.get("matched_capabilities", []),
                        match_score=match_score,
                        reasoning=match_data.get("reasoning", ""),
                        process=best_process
                    ))

            except Exception as e:
                logger.error(f"Error matching capabilities for {plugin.name}: {e}")
                continue

        # Sort by match score
        matches.sort(key=lambda x: x.match_score, reverse=True)

        return matches

    async def _resolve_plugin_competition(
        self,
        capability_matches: List[CapabilityMatch],
        context: ConversationContext,
        max_plugins: int
    ) -> List[PluginSelectionResult]:
        """Resolve competition between plugins that could handle the request."""

        if not capability_matches:
            return []

        # If only one match or within limit, no competition
        if len(capability_matches) <= max_plugins:
            return [
                PluginSelectionResult(
                    plugin=match.plugin,
                    confidence=match.match_score,
                    reasoning=match.reasoning,
                    matching_process=match.process,
                    selection_method="capability_match"
                )
                for match in capability_matches
            ]

        # Resolve competition using sophisticated reasoning
        competition_prompt = f"""Multiple plugins can handle this request. Select the best {max_plugins} plugins.

User Context:
- Department: {context.user_attributes.get('department', 'unknown')}
- Role: {context.user_attributes.get('role', 'unknown')}

Available Plugins:
"""

        for i, match in enumerate(capability_matches[:10]):  # why we have 2 reasing delete ? delete one Limit to top 10 for LLM
            competition_prompt += f"""
{i+1}. {match.plugin.name}
   - Match Score: {match.match_score:.2f}
   - Capabilities: {', '.join(match.matched_capabilities)}
   - Reasoning: {match.reasoning}
   - Domain: {', '.join(match.plugin.domain_compatibility) if match.plugin.domain_compatibility else 'general'}
"""

        competition_prompt += f"""
Select the best {max_plugins} plugins considering:
1. Match score and relevance
2. User's department/role alignment
3. Plugin specialization vs generalization
4. Potential for collaboration

Respond with JSON: {{"selected_plugins": [1, 3, 5], "reasoning": "explanation of selection"}}"""

        try:
            response = await self.llm.ainvoke(competition_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            selection_data = json.loads(response_text)
            selected_indices = selection_data.get("selected_plugins", [])
            selection_reasoning = selection_data.get("reasoning", "")

            selected_plugins = []
            for idx in selected_indices:
                if 1 <= idx <= len(capability_matches):
                    match = capability_matches[idx - 1]  # Convert to 0-based index
                    selected_plugins.append(
                        PluginSelectionResult(
                            plugin=match.plugin,
                            confidence=match.match_score,
                            reasoning=f"{match.reasoning} | Competition: {selection_reasoning}",
                            matching_process=match.process,
                            selection_method="competition_resolved"
                        )
                    )

            return selected_plugins[:max_plugins]

        except Exception as e:
            logger.error(f"Error resolving plugin competition: {e}")
            # Fallback to top matches by score
            return [
                PluginSelectionResult(
                    plugin=match.plugin,
                    confidence=match.match_score,
                    reasoning=f"{match.reasoning} | Fallback selection",
                    matching_process=match.process,
                    selection_method="fallback"
                )
                for match in capability_matches[:max_plugins]
            ]

    async def _plan_multi_plugin_execution(
        self,
        selected_plugins: List[PluginSelectionResult],
        intent_analysis: IntentAnalysis,
        context: ConversationContext
    ) -> ExecutionPlan:
        """Plan execution strategy for multiple plugins."""

        if not selected_plugins:
            return ExecutionPlan(primary_plugin=None, confidence=0.0)

        if len(selected_plugins) == 1:
            # Single plugin execution
            return ExecutionPlan(
                primary_plugin=selected_plugins[0].plugin,
                execution_order=[selected_plugins[0].plugin.name],
                coordination_strategy="single",
                confidence=selected_plugins[0].confidence
            )

        # Multi-plugin coordination planning
        plugins_info = []
        for result in selected_plugins:
            plugins_info.append({
                "name": result.plugin.name,
                "description": result.plugin.description,
                "confidence": result.confidence,
                "capabilities": result.plugin.capabilities
            })

        planning_prompt = f"""Plan execution for multiple plugins to handle this user intent.

User Intent: {intent_analysis.primary_intent}
Intent Complexity: {intent_analysis.complexity}
Urgency: {intent_analysis.urgency_level}

Selected Plugins:
{json.dumps(plugins_info, indent=2)}

Determine:
1. Primary plugin (main handler)
2. Supporting plugins (if any)
3. Execution order
4. Coordination strategy: sequential, parallel, or conditional

Respond with JSON:
{{
    "primary_plugin": "plugin_name",
    "supporting_plugins": ["plugin2", "plugin3"],
    "execution_order": ["plugin1", "plugin2", "plugin3"],
    "coordination_strategy": "sequential|parallel|conditional",
    "estimated_duration": 30.0,
    "reasoning": "explanation"
}}"""

        try:
            response = await self.llm.ainvoke(planning_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            plan_data = json.loads(response_text)

            # Find primary plugin
            primary_plugin = None
            supporting_plugins = []

            primary_name = plan_data.get("primary_plugin")
            for result in selected_plugins:
                if result.plugin.name == primary_name:
                    primary_plugin = result.plugin
                else:
                    supporting_plugins.append(result.plugin)

            if not primary_plugin:
                primary_plugin = selected_plugins[0].plugin  # Fallback

            return ExecutionPlan(
                primary_plugin=primary_plugin,
                supporting_plugins=supporting_plugins,
                execution_order=plan_data.get("execution_order", [primary_plugin.name]),
                coordination_strategy=plan_data.get("coordination_strategy", "sequential"),
                estimated_duration=plan_data.get("estimated_duration", 30.0),
                confidence=sum(r.confidence for r in selected_plugins) / len(selected_plugins)
            )

        except Exception as e:
            logger.error(f"Error planning multi-plugin execution: {e}")
            # Fallback to simple sequential execution
            return ExecutionPlan(
                primary_plugin=selected_plugins[0].plugin,
                supporting_plugins=[r.plugin for r in selected_plugins[1:]],
                execution_order=[r.plugin.name for r in selected_plugins],
                coordination_strategy="sequential",
                confidence=selected_plugins[0].confidence
            )

    def clear_cache(self):
        """Clear all caches."""
        self.intent_cache.clear()
        self.capability_cache.clear()
        logger.info("Manifest generator caches cleared")

    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics data for the manifest generator."""
        return {
            "intent_cache_size": len(self.intent_cache),
            "capability_cache_size": len(self.capability_cache),
            "total_cache_entries": len(self.intent_cache) + len(self.capability_cache)
        }

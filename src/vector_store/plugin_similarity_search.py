"""
Moveworks-style Plugin Similarity Search for 1000+ Plugin Scaling.
Uses LangChain's PGVector integration for efficient embedding-based plugin selection.
"""

import os
import json
import uuid
from typing import Dict, List, Any, Optional, Union, cast
from datetime import datetime
import logging

from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from models.moveworks import Plugin, ConversationalProcess
from llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class PluginSimilarityResult:
    """Result of plugin similarity search."""
    
    def __init__(
        self,
        plugin: Plugin,
        similarity_score: float,
        match_reasoning: str,
        capability_matches: List[str],
        confidence: float,
        matching_process: Optional[ConversationalProcess] = None
    ):
        self.plugin = plugin
        self.similarity_score = similarity_score
        self.match_reasoning = match_reasoning
        self.capability_matches = capability_matches
        self.confidence = confidence
        self.matching_process = matching_process
        self.timestamp = datetime.now()


class MoveworksPluginSimilaritySearch:
    """
    Moveworks-style plugin similarity search engine for 1000+ plugin scaling.
    Uses LangChain's PGVector for efficient embedding-based plugin selection.
    """

    def __init__(self, connection_string: Optional[str] = None, collection_name: str = "moveworks_plugins"):
        """Initialize the plugin similarity search engine."""
        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/moveworks_plugins"
        )
        self.collection_name = collection_name
        self.vector_store: Optional[PGVector] = None
        self.embeddings: Optional[Embeddings] = None
        self.plugin_cache: Dict[str, Plugin] = {}
        self.search_cache: Dict[str, List[PluginSimilarityResult]] = {}

    async def initialize(self):
        """Initialize LangChain PGVector and embedding model."""
        try:
            # Initialize embeddings using Together AI (configured via TOGETHER_API_KEY)
            self.embeddings = LLMFactory.get_embedding_model(provider="together")

            # Initialize PGVector store using LangChain's recommended approach
            self.vector_store = PGVector(
                connection=self.connection_string,
                embeddings=self.embeddings,
                collection_name=self.collection_name
            )

            logger.info("Plugin similarity search initialized with LangChain PGVector")

        except Exception as e:
            logger.error(f"Failed to initialize plugin similarity search: {e}")
            raise
    
    def _create_plugin_document(self, plugin: Plugin) -> Document:
        """Create a LangChain Document from a Plugin for vector storage."""
        # Create comprehensive text representation for embedding
        text_parts = [
            plugin.description,
            " ".join(plugin.capabilities),
            " ".join(plugin.domain_compatibility),
            " ".join(plugin.positive_examples)
        ]

        # Add conversational process descriptions
        for process in plugin.conversational_processes:
            text_parts.append(process.description)
            text_parts.extend(process.trigger_utterances)

        content = " ".join(filter(None, text_parts))

        # Create metadata for filtering and retrieval
        metadata = {
            "plugin_name": plugin.name,
            "description": plugin.description,
            "capabilities": plugin.capabilities,
            "domain_compatibility": plugin.domain_compatibility,
            "confidence_threshold": plugin.confidence_threshold,
            "success_rate": plugin.success_rate,
            "positive_examples": plugin.positive_examples,
            "negative_examples": plugin.negative_examples,
            "plugin_data": plugin.model_dump()  # Store full plugin data
        }

        return Document(
            page_content=content,
            metadata=metadata
        )
    
    async def register_plugin(self, plugin: Plugin) -> str:
        """Register a new plugin in the vector store."""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Call initialize() first.")

            # Create document from plugin
            document = self._create_plugin_document(plugin)

            # Add document to vector store
            # LangChain PGVector handles embedding generation automatically
            ids = self.vector_store.add_documents([document])
            plugin_id = ids[0] if ids else str(uuid.uuid4())

            # Add to cache
            self.plugin_cache[plugin_id] = plugin

            logger.info(f"Registered plugin: {plugin.name} ({plugin_id})")
            return plugin_id

        except Exception as e:
            logger.error(f"Error registering plugin {plugin.name}: {e}")
            raise
    
    async def search_similar_plugins(
        self,
        query: str,
        domain: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.7,
        exclude_plugins: Optional[List[str]] = None
    ) -> List[PluginSimilarityResult]:
        """
        Search for plugins similar to the query using LangChain PGVector.

        Args:
            query: Natural language query describing the task
            domain: Optional domain filter
            required_capabilities: Optional list of required capabilities
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold
            exclude_plugins: Optional list of plugin names to exclude

        Returns:
            List of PluginSimilarityResult objects ranked by similarity
        """
        # Check cache first
        cache_key = f"{query}:{domain}:{required_capabilities}:{max_results}:{similarity_threshold}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Call initialize() first.")

            # Build metadata filter
            filter_dict: Dict[str, Any] = {}

            if domain:
                filter_dict["domain_compatibility"] = {"$in": [domain]}

            if required_capabilities:
                # Filter for plugins that have at least one of the required capabilities
                filter_dict["capabilities"] = {"$in": required_capabilities}

            if exclude_plugins:
                filter_dict["plugin_name"] = {"$nin": exclude_plugins}

            # Perform similarity search using LangChain PGVector
            if filter_dict:
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=max_results,
                    filter=filter_dict
                )
            else:
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=max_results
                )

            results = []
            for doc, score in docs_with_scores:
                # Convert score to similarity (PGVector returns distance, we want similarity)
                similarity_score = 1.0 - score if score <= 1.0 else 1.0 / (1.0 + score)

                # Skip results below threshold
                if similarity_score < similarity_threshold:
                    continue

                try:
                    # Reconstruct Plugin from metadata
                    plugin_data_raw = doc.metadata.get("plugin_data", {})
                    if isinstance(plugin_data_raw, str):
                        plugin_data = cast(Dict[str, Any], json.loads(plugin_data_raw))
                    else:
                        plugin_data = cast(Dict[str, Any], plugin_data_raw)

                    plugin = Plugin(**plugin_data)

                    # Find capability matches
                    capability_matches = self._find_capability_matches(query, plugin.capabilities)

                    # Generate match reasoning
                    match_reasoning = await self._generate_match_reasoning(query, plugin)

                    # Find best matching process
                    matching_process = self._find_best_matching_process(query, plugin.conversational_processes)

                    result = PluginSimilarityResult(
                        plugin=plugin,
                        similarity_score=similarity_score,
                        match_reasoning=match_reasoning,
                        capability_matches=capability_matches,
                        confidence=min(similarity_score * plugin.success_rate, 1.0),
                        matching_process=matching_process
                    )
                    results.append(result)

                except Exception as e:
                    logger.error(f"Error processing search result: {e}")
                    continue

            # Cache results
            self.search_cache[cache_key] = results

            return results

        except Exception as e:
            logger.error(f"Error searching similar plugins: {e}")
            return []



    def _find_capability_matches(self, query: str, capabilities: List[str]) -> List[str]:
        """Find capability matches between query and plugin capabilities."""
        query_lower = query.lower()
        matches = []

        for capability in capabilities:
            # Simple keyword matching - can be enhanced with semantic matching
            if any(word in query_lower for word in capability.lower().split()):
                matches.append(capability)

        return matches

    def _find_best_matching_process(
        self,
        query: str,
        processes: List[ConversationalProcess]
    ) -> Optional[ConversationalProcess]:
        """Find the best matching conversational process for the query."""
        if not processes:
            return None

        query_lower = query.lower()
        best_match = None
        best_score = 0

        for process in processes:
            score = 0

            # Check trigger utterances
            for utterance in process.trigger_utterances:
                if utterance.lower() in query_lower or query_lower in utterance.lower():
                    score += 2
                elif any(word in query_lower for word in utterance.lower().split()):
                    score += 1

            # Check description
            if any(word in query_lower for word in process.description.lower().split()):
                score += 1

            if score > best_score:
                best_score = score
                best_match = process

        return best_match

    async def _generate_match_reasoning(self, query: str, plugin: Plugin) -> str:
        """Generate reasoning for why this plugin matches the query."""
        try:
            reasoning_prompt = f"""Explain why this plugin is a good match for the query.

Query: {query}

Plugin Profile:
- Name: {plugin.name}
- Description: {plugin.description}
- Capabilities: {', '.join(plugin.capabilities)}
- Domain Compatibility: {', '.join(plugin.domain_compatibility)}
- Success Rate: {plugin.success_rate}

Provide a brief explanation (1-2 sentences) of why this plugin matches the query."""

            llm = LLMFactory.get_fast_llm()
            response = await llm.ainvoke(reasoning_prompt)

            # Handle different response types
            if hasattr(response, 'content'):
                content = response.content
                if isinstance(content, str):
                    reasoning = content
                elif isinstance(content, list):
                    # Handle list of content parts
                    reasoning = " ".join(str(part) for part in content)
                else:
                    reasoning = str(content)
            else:
                reasoning = str(response)

            return reasoning.strip()

        except Exception as e:
            logger.error(f"Error generating match reasoning: {e}")
            return f"Plugin matches based on capabilities: {', '.join(plugin.capabilities[:3])}"

    async def update_plugin_performance(
        self,
        plugin_name: str,
        success: bool,
        response_time: float
    ):
        """Update plugin performance metrics in cache."""
        try:
            # Update cache if plugin is cached
            for plugin in self.plugin_cache.values():
                if plugin.name == plugin_name:
                    # Simple performance tracking in memory
                    # In production, you might want to persist this to a separate analytics table
                    current_rate = plugin.success_rate
                    new_rate = (current_rate + (1.0 if success else 0.0)) / 2.0
                    plugin.success_rate = new_rate
                    break

        except Exception as e:
            logger.error(f"Error updating plugin performance: {e}")

    async def get_plugin_analytics(self) -> Dict[str, Any]:
        """Get analytics data for plugin performance and search patterns."""
        try:
            # Simple analytics from cache and vector store
            total_plugins = len(self.plugin_cache)
            avg_success_rate = sum(p.success_rate for p in self.plugin_cache.values()) / total_plugins if total_plugins > 0 else 0.0

            return {
                "total_plugins": total_plugins,
                "avg_success_rate": avg_success_rate,
                "cache_size": len(self.plugin_cache),
                "search_cache_size": len(self.search_cache)
            }

        except Exception as e:
            logger.error(f"Error getting plugin analytics: {e}")
            return {}

    async def clear_cache(self):
        """Clear all caches."""
        self.plugin_cache.clear()
        self.search_cache.clear()
        logger.info("Plugin similarity search caches cleared")

    async def close(self):
        """Close vector store connections."""
        # LangChain PGVector handles connection management
        logger.info("Plugin similarity search closed")

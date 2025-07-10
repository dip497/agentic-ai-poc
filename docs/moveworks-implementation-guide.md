# Moveworks-Style Implementation Guide

## 🎯 Overview

This guide covers the newly implemented Moveworks-style components that bring autonomous plugin selection, 1000+ agent scaling, and sophisticated slot resolution to the system.

## 🚀 Key Components Implemented

### 1. Manifest Generator (`src/reasoning/manifest_generator.py`)

**Purpose:** Autonomous plugin selection with sophisticated reasoning, following exact Moveworks patterns.

**Key Features:**
- **Intent Analysis:** LLM-powered understanding of user intent
- **Capability Matching:** Semantic matching between intent and plugin capabilities
- **Competition Resolution:** AI-powered selection when multiple plugins match
- **Multi-Plugin Coordination:** Planning execution strategies for complex workflows

**Usage:**
```python
from src.reasoning.manifest_generator import MoveworksManifestGenerator

# Initialize
manifest_generator = MoveworksManifestGenerator()
await manifest_generator.initialize()

# Generate manifest
manifest = await manifest_generator.generate_manifest(
    user_intent="What's my vacation balance?",
    context=conversation_context,
    available_plugins=plugins,
    max_plugins=3
)

print(f"Selected: {manifest.execution_plan.primary_plugin.name}")
```

### 2. Agent Similarity Search (`src/vector_store/agent_similarity_search.py`)

**Purpose:** Scale to 1000+ agents using embedding-based similarity search with PostgreSQL + pgvector.

**Key Features:**
- **Embedding-Based Search:** Semantic agent matching using vector similarity
- **Performance Tracking:** Agent performance metrics and analytics
- **Dynamic Registration:** Runtime agent registration and management
- **Optimized Scaling:** PostgreSQL with pgvector for efficient search

**Usage:**
```python
from src.vector_store.agent_similarity_search import MoveworksAgentSimilaritySearch, AgentProfile

# Initialize
similarity_search = MoveworksAgentSimilaritySearch()
await similarity_search.initialize()

# Register agents
agent = AgentProfile(
    name="PTO Specialist",
    description="Handles PTO requests and balance inquiries",
    capabilities=["pto_management", "balance_inquiry"],
    domain_expertise=["hr", "employee_services"]
)
agent_id = await similarity_search.register_agent(agent)

# Search for similar agents
results = await similarity_search.search_similar_agents(
    query="I need help with vacation time",
    max_results=5,
    similarity_threshold=0.7
)
```

### 3. Slot Resolver Engine (`src/reasoning/slot_resolver_engine.py`)

**Purpose:** Implement Moveworks-style slot resolution with Static, API, and Inline resolvers.

**Key Features:**
- **Static Resolvers:** Predefined options with intelligent matching
- **API Resolvers:** Dynamic data resolution from external APIs
- **Inline Resolvers:** LLM-powered custom logic resolution
- **Confidence Scoring:** Reasoning and confidence for all resolutions

**Usage:**
```python
from src.reasoning.slot_resolver_engine import MoveworksSlotResolverEngine

# Initialize
slot_resolver = MoveworksSlotResolverEngine()
await slot_resolver.initialize()

# Resolve slot
result = await slot_resolver.resolve_slot(
    slot=slot_definition,
    user_input="vacation",
    context=conversation_context,
    process_data={}
)

if result.success:
    print(f"Resolved: {result.display_value} -> {result.raw_value}")
```

## 🔧 Setup Instructions

### 1. Database Setup (for Agent Similarity Search)

```bash
# Install PostgreSQL with pgvector
sudo apt-get install postgresql postgresql-contrib
sudo -u postgres psql -c "CREATE EXTENSION vector;"

# Set environment variable
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/moveworks_agents"
```

### 2. LLM Configuration

Configure your LLM provider in `config/moveworks_config.yml`:

```yaml
llm:
  default:
    provider: "gemini"  # or openai, anthropic
    model: "gemini-1.5-flash"
    temperature: 0.1
    max_tokens: 2000
```

### 3. Environment Variables

```bash
# Required for LLM providers
export GEMINI_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"  # if using OpenAI

# Database for agent similarity search
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/moveworks_agents"
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_moveworks_implementation.py
```

This will test:
- ✅ Manifest Generator functionality
- ✅ Agent Similarity Search with 1000+ scaling
- ✅ Slot Resolver Engine with all three types
- ✅ Component integration

## 📊 Performance & Scaling

### Agent Similarity Search Performance
- **Database:** PostgreSQL with pgvector extension
- **Indexing:** IVFFlat index for vector similarity
- **Caching:** Multi-level caching for frequently accessed agents
- **Pool Size:** Configurable connection pool (default: 5-50 connections)

### Manifest Generator Performance
- **Caching:** Intent and capability matching results cached
- **LLM Optimization:** Uses fast LLM for quick responses
- **Parallel Processing:** Capability matching can be parallelized

### Slot Resolver Performance
- **Resolution Caching:** Successful resolutions cached by context
- **API Optimization:** HTTP connection pooling for API resolvers
- **Fallback Strategies:** Graceful degradation on failures

## 🔗 Integration with Existing System

The new components integrate seamlessly with the existing reasoning engine:

```python
# In MoveworksReasoningEngine
self.manifest_generator = MoveworksManifestGenerator(self.plugin_selector)
await self.manifest_generator.initialize()

# Manifest generation replaces basic plugin selection
manifest = await self.manifest_generator.generate_manifest(
    user_intent=user_query,
    context=context,
    available_plugins=available_plugins
)
```

## 📈 Analytics & Monitoring

All components provide analytics:

```python
# Manifest Generator Analytics
analytics = manifest_generator.get_analytics()
print(f"Cache entries: {analytics['total_cache_entries']}")

# Agent Similarity Search Analytics
analytics = await similarity_search.get_agent_analytics()
print(f"Total agents: {analytics['agent_statistics']['total_agents']}")

# Slot Resolver Analytics
analytics = slot_resolver.get_analytics()
print(f"Resolutions cached: {analytics['cache_size']}")
```

## 🎯 Next Steps

1. **Configure your LLM provider** (Gemini recommended for cost-effectiveness)
2. **Set up PostgreSQL with pgvector** for agent similarity search
3. **Load your actual plugins and agents** into the system
4. **Test with real user queries** to validate performance
5. **Monitor analytics** to optimize performance

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Ensure PostgreSQL is running
   - Check DATABASE_URL environment variable
   - Verify pgvector extension is installed

2. **LLM API Errors**
   - Check API key environment variables
   - Verify API quotas and limits
   - Test with simple queries first

3. **Performance Issues**
   - Monitor cache hit rates
   - Check database connection pool size
   - Review LLM response times

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 References

- [Moveworks Agentic Reasoning Engine Documentation](https://help.moveworks.com/docs/agentic-reasoning-engine)
- [Moveworks Plugin Architecture](https://help.moveworks.com/docs/plugin-architecture)
- [PostgreSQL pgvector Extension](https://github.com/pgvector/pgvector)

---

**Implementation Status:** ✅ Core components complete, ready for production testing
**Last Updated:** 2025-01-08
**Next Phase:** Policy Validator Engine and enhanced orchestration

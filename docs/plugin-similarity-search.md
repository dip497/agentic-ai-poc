# Moveworks Plugin Similarity Search

## Overview

The Moveworks Plugin Similarity Search system enables efficient scaling to 1000+ plugins using **LangChain's PGVector integration**. This system allows the Moveworks reasoning engine to quickly find the most relevant plugins for any given user request using embedding-based similarity search.

## Architecture

### Core Components

1. **MoveworksPluginSimilaritySearch**: Main class that manages plugin registration and similarity search
2. **LangChain PGVector**: Vector store backend for efficient similarity search
3. **Plugin Documents**: LangChain Document objects created from Plugin models
4. **Similarity Results**: Structured results with reasoning and confidence scores

### Key Features

- **LangChain Integration**: Uses LangChain's PGVector for robust vector operations
- **Automatic Embedding**: LangChain handles embedding generation automatically
- **Metadata Filtering**: Filter by domain, capabilities, and other criteria
- **Performance Tracking**: Monitor plugin success rates and usage
- **Caching**: In-memory caching for frequently accessed plugins and searches

## Implementation

### Initialization

```python
from src.vector_store.plugin_similarity_search import MoveworksPluginSimilaritySearch

# Initialize with PostgreSQL connection
search_engine = MoveworksPluginSimilaritySearch(
    connection_string="postgresql+psycopg://postgres:postgres@localhost:5432/moveworks_plugins",
    collection_name="moveworks_plugins"
)

await search_engine.initialize()
```

### Plugin Registration

```python
from src.models.moveworks import Plugin, ConversationalProcess

# Create a plugin
plugin = Plugin(
    name="Knowledge Base",
    description="Search and retrieve information from company knowledge base",
    conversational_processes=[...],
    capabilities=["search", "information_retrieval"],
    domain_compatibility=["general", "hr", "it"],
    positive_examples=["Find information about vacation policy"],
    confidence_threshold=0.8,
    success_rate=0.92
)

# Register in vector store
plugin_id = await search_engine.register_plugin(plugin)
```

### Similarity Search

```python
# Basic similarity search
results = await search_engine.search_similar_plugins(
    query="I need to find information about company policies",
    max_results=5,
    similarity_threshold=0.7
)

# Advanced search with filters
results = await search_engine.search_similar_plugins(
    query="Create a support ticket",
    domain="it",
    required_capabilities=["ticket_creation"],
    max_results=3,
    exclude_plugins=["Legacy Ticket System"]
)

# Process results
for result in results:
    print(f"Plugin: {result.plugin.name}")
    print(f"Similarity: {result.similarity_score:.3f}")
    print(f"Reasoning: {result.match_reasoning}")
    print(f"Capabilities: {result.capability_matches}")
```

## Data Flow

### Plugin Registration Flow

1. **Plugin Creation**: Create Plugin model with comprehensive metadata
2. **Document Generation**: Convert Plugin to LangChain Document with searchable content
3. **Vector Storage**: LangChain PGVector automatically generates embeddings and stores
4. **Cache Update**: Add plugin to in-memory cache for fast access

### Search Flow

1. **Query Processing**: Receive natural language query with optional filters
2. **Cache Check**: Check if results are already cached
3. **Vector Search**: Use LangChain PGVector similarity search with metadata filters
4. **Result Processing**: Convert vector results back to Plugin objects
5. **Reasoning Generation**: Generate match reasoning using LLM
6. **Result Ranking**: Sort by similarity score and confidence
7. **Cache Storage**: Store results for future queries

## Configuration

### Database Setup

```sql
-- PostgreSQL with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- LangChain PGVector handles table creation automatically
```

### Environment Variables

```bash
# Database connection
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/moveworks_plugins

# LLM configuration (for reasoning generation)
OPENAI_API_KEY=your_openai_key
```

## Integration with Manifest Generator

The Plugin Similarity Search integrates seamlessly with the Manifest Generator:

```python
from src.reasoning.manifest_generator import MoveworksManifestGenerator
from src.vector_store.plugin_similarity_search import MoveworksPluginSimilaritySearch

# Initialize both components
similarity_search = MoveworksPluginSimilaritySearch()
await similarity_search.initialize()

manifest_generator = MoveworksManifestGenerator(
    plugin_similarity_search=similarity_search
)
await manifest_generator.initialize()

# The manifest generator will automatically use similarity search
# for enhanced capability matching
```

## Performance Considerations

### Scaling to 1000+ Plugins

1. **Vector Indexing**: LangChain PGVector automatically creates efficient indexes
2. **Metadata Filtering**: Use domain and capability filters to reduce search space
3. **Caching Strategy**: Frequently accessed plugins and searches are cached
4. **Connection Pooling**: LangChain handles database connection management

### Optimization Tips

1. **Batch Registration**: Register multiple plugins in batches when possible
2. **Smart Caching**: Clear cache periodically to prevent memory bloat
3. **Filter Early**: Use metadata filters to reduce vector search scope
4. **Monitor Performance**: Track search times and plugin success rates

## Error Handling

The system includes comprehensive error handling:

- **Connection Failures**: Graceful degradation when database is unavailable
- **Embedding Errors**: Fallback mechanisms for embedding generation
- **Plugin Reconstruction**: Safe handling of malformed plugin data
- **Search Timeouts**: Configurable timeouts for long-running searches

## Testing

Run the test suite to verify functionality:

```bash
python test_plugin_similarity_search.py
```

The test covers:
- Plugin registration and retrieval
- Similarity search with various filters
- Performance tracking and analytics
- Error handling scenarios

## Future Enhancements

1. **Advanced Filtering**: Support for complex metadata queries
2. **Hybrid Search**: Combine vector similarity with keyword search
3. **Real-time Updates**: Live plugin updates without restart
4. **Analytics Dashboard**: Web interface for monitoring plugin performance
5. **A/B Testing**: Compare different similarity algorithms

## Troubleshooting

### Common Issues

1. **Connection Errors**: Verify PostgreSQL is running with pgvector extension
2. **Embedding Failures**: Check LLM factory configuration and API keys
3. **Slow Searches**: Consider adding more specific metadata filters
4. **Memory Usage**: Monitor cache size and clear periodically

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger("src.vector_store.plugin_similarity_search").setLevel(logging.DEBUG)
```

## Related Documentation

- [Plugin Selection Engine](plugin-selection.md)
- [Moveworks Implementation Guide](moveworks-implementation-guide.md)
- [Data Types](data-types/README.md)

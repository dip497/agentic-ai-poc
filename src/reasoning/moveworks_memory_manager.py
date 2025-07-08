"""
Moveworks Memory Manager Implementation.
Manages all four memory constructs with PostgreSQL + pgvector integration.
Supports dynamic domains and 1000+ agent scaling.
"""

import os
import asyncio
import asyncpg
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
import logging

from .memory_constructs import (
    MemoryType, DomainDefinition, SemanticMemoryEntry, EpisodicMemoryEntry,
    ConversationContext, ProcedureMemoryEntry, WorkingMemoryEntry, MemorySnapshot
)
from config.reasoning_config import get_reasoning_config

logger = logging.getLogger(__name__)


class MoveworksMemoryManager:
    """
    Moveworks Memory Manager implementing all four memory constructs:
    - Semantic Memory: Entity knowledge and domain awareness
    - Episodic Memory: Conversation context and history
    - Procedure Memory: Plugin capabilities and business processes  
    - Working Memory: Process state and variable tracking
    
    Features:
    - Dynamic domain management (no hardcoded domains)
    - Vector search with pgvector for 1000+ agent scaling
    - Variable tracking framework for business object integrity
    - Moveworks 6-20 message window management
    """
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/agent_studio"
        )
        self.embedding_dimension = 1536  # OpenAI embedding size
        self._domain_cache: Dict[str, DomainDefinition] = {}
    
    async def initialize(self):
        """Initialize memory manager with database connection and tables."""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Create memory tables
            await self._create_memory_tables()
            
            # Initialize sample memory data
            await self._initialize_sample_memory()
            
            # Load domain cache
            await self._load_domain_cache()
            
            logger.info("Moveworks Memory Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            raise
    
    async def _create_memory_tables(self):
        """Create all memory construct tables with pgvector support."""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
            
            # Domain Definitions table (dynamic domains)
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS domain_definitions (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name VARCHAR(255) UNIQUE NOT NULL,
                    display_name VARCHAR(255) NOT NULL,
                    description TEXT,
                    parent_domain VARCHAR(255),
                    keywords JSONB DEFAULT '[]',
                    trigger_phrases JSONB DEFAULT '[]',
                    confidence_threshold FLOAT DEFAULT 0.7,
                    embedding vector({self.embedding_dimension}),
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Semantic Memory table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    entity_type VARCHAR(255) NOT NULL,
                    entity_name VARCHAR(255) NOT NULL,
                    entity_description TEXT,
                    domain VARCHAR(255) NOT NULL,
                    properties JSONB DEFAULT '{{}}',
                    synonyms JSONB DEFAULT '[]',
                    embedding vector({self.embedding_dimension}),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Episodic Memory table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    conversation_id UUID NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    message_type VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    intent VARCHAR(255),
                    entities_extracted JSONB DEFAULT '{}',
                    slot_values JSONB DEFAULT '{}',
                    plugin_calls JSONB DEFAULT '[]',
                    timestamp TIMESTAMP DEFAULT NOW(),
                    sequence_number INTEGER DEFAULT 0
                )
            """)
            
            # Conversation Context table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_contexts (
                    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id VARCHAR(255) NOT NULL,
                    domain VARCHAR(255),
                    route VARCHAR(50),
                    persistent_slots JSONB DEFAULT '{}',
                    active_plugins JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_updated TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Procedure Memory table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS procedure_memory (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    plugin_id VARCHAR(255) NOT NULL,
                    plugin_name VARCHAR(255) NOT NULL,
                    description TEXT,
                    capabilities JSONB DEFAULT '[]',
                    trigger_utterances JSONB DEFAULT '[]',
                    domain_compatibility JSONB DEFAULT '[]',
                    required_slots JSONB DEFAULT '[]',
                    output_types JSONB DEFAULT '[]',
                    business_rules JSONB DEFAULT '{{}}',
                    embedding vector({self.embedding_dimension}),
                    confidence_threshold FLOAT DEFAULT 0.7,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Working Memory table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS working_memory (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    conversation_id UUID NOT NULL,
                    process_id VARCHAR(255) NOT NULL,
                    process_name VARCHAR(255),
                    current_step VARCHAR(255),
                    status VARCHAR(50) DEFAULT 'pending',
                    variables JSONB DEFAULT '{}',
                    business_objects JSONB DEFAULT '{}',
                    step_history JSONB DEFAULT '[]',
                    reference_links JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_domain_definitions_embedding ON domain_definitions USING ivfflat (embedding vector_cosine_ops);
                CREATE INDEX IF NOT EXISTS idx_semantic_memory_embedding ON semantic_memory USING ivfflat (embedding vector_cosine_ops);
                CREATE INDEX IF NOT EXISTS idx_semantic_memory_domain ON semantic_memory(domain);
                CREATE INDEX IF NOT EXISTS idx_episodic_memory_conversation ON episodic_memory(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_episodic_memory_timestamp ON episodic_memory(timestamp);
                CREATE INDEX IF NOT EXISTS idx_procedure_memory_embedding ON procedure_memory USING ivfflat (embedding vector_cosine_ops);
                CREATE INDEX IF NOT EXISTS idx_procedure_memory_domain ON procedure_memory USING gin(domain_compatibility);
                CREATE INDEX IF NOT EXISTS idx_working_memory_conversation ON working_memory(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_working_memory_status ON working_memory(status);
            """)
            
            logger.info("Memory construct tables created successfully")
    
    async def _initialize_sample_memory(self):
        """Initialize with sample memory data including dynamic domains."""
        async with self.pool.acquire() as conn:
            # Check if we already have domain data
            domain_count = await conn.fetchval("SELECT COUNT(*) FROM domain_definitions")
            
            if domain_count == 0:
                # Insert sample domains
                sample_domains = [
                    {
                        "name": "IT_DOMAIN",
                        "display_name": "Information Technology",
                        "description": "IT support, infrastructure, software, and technical issues",
                        "keywords": ["IT", "technical", "software", "hardware", "system", "network"],
                        "trigger_phrases": ["technical issue", "software problem", "system down", "network issue"]
                    },
                    {
                        "name": "HR_DOMAIN", 
                        "display_name": "Human Resources",
                        "description": "HR policies, benefits, payroll, and employee relations",
                        "keywords": ["HR", "human resources", "benefits", "payroll", "employee", "policy"],
                        "trigger_phrases": ["HR question", "benefits inquiry", "payroll issue", "employee handbook"]
                    },
                    {
                        "name": "SALES_DOMAIN",
                        "display_name": "Sales",
                        "description": "Sales processes, CRM, accounts, and customer management",
                        "keywords": ["sales", "CRM", "customer", "account", "lead", "opportunity"],
                        "trigger_phrases": ["sales question", "customer account", "CRM issue", "lead management"]
                    },
                    {
                        "name": "FINANCE_DOMAIN",
                        "display_name": "Finance",
                        "description": "Financial processes, accounting, expenses, and budgets",
                        "keywords": ["finance", "accounting", "expense", "budget", "invoice", "payment"],
                        "trigger_phrases": ["expense report", "budget question", "invoice issue", "financial data"]
                    },
                    {
                        "name": "GENERAL_DOMAIN",
                        "display_name": "General",
                        "description": "General inquiries and cross-domain questions",
                        "keywords": ["general", "help", "question", "information"],
                        "trigger_phrases": ["general question", "need help", "information request"]
                    }
                ]
                
                for domain in sample_domains:
                    await conn.execute("""
                        INSERT INTO domain_definitions (
                            name, display_name, description, keywords, trigger_phrases
                        ) VALUES ($1, $2, $3, $4, $5)
                    """,
                        domain["name"],
                        domain["display_name"],
                        domain["description"],
                        json.dumps(domain["keywords"]),
                        json.dumps(domain["trigger_phrases"])
                    )
                
                logger.info("Sample domain definitions inserted")
            
            # Check if we already have semantic memory data
            semantic_count = await conn.fetchval("SELECT COUNT(*) FROM semantic_memory")
            
            if semantic_count == 0:
                # Insert sample semantic memory entries
                sample_entities = [
                    {
                        "entity_type": "u_JiraIssue",
                        "entity_name": "Jira Issue",
                        "entity_description": "A Jira issue or ticket for tracking work items",
                        "domain": "IT_DOMAIN",
                        "properties": {"fields": ["key", "summary", "description", "status", "assignee"]},
                        "synonyms": ["jira ticket", "issue", "bug", "task", "story"]
                    },
                    {
                        "entity_type": "u_ServiceNowTicket",
                        "entity_name": "ServiceNow Ticket",
                        "entity_description": "A ServiceNow incident or request ticket",
                        "domain": "IT_DOMAIN", 
                        "properties": {"fields": ["number", "short_description", "state", "assigned_to"]},
                        "synonyms": ["snow ticket", "incident", "request", "service ticket"]
                    },
                    {
                        "entity_type": "u_SalesforceAccount",
                        "entity_name": "Salesforce Account",
                        "entity_description": "A Salesforce customer account record",
                        "domain": "SALES_DOMAIN",
                        "properties": {"fields": ["name", "type", "industry", "owner"]},
                        "synonyms": ["account", "customer", "client", "prospect"]
                    }
                ]
                
                for entity in sample_entities:
                    await conn.execute("""
                        INSERT INTO semantic_memory (
                            entity_type, entity_name, entity_description, domain, properties, synonyms
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                        entity["entity_type"],
                        entity["entity_name"], 
                        entity["entity_description"],
                        entity["domain"],
                        json.dumps(entity["properties"]),
                        json.dumps(entity["synonyms"])
                    )
                
                logger.info("Sample semantic memory data inserted")
    
    async def _load_domain_cache(self):
        """Load domain definitions into cache for fast access."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT name, display_name, description, parent_domain, keywords, 
                       trigger_phrases, confidence_threshold, is_active, created_at, updated_at
                FROM domain_definitions 
                WHERE is_active = TRUE
            """)
            
            for row in rows:
                domain = DomainDefinition(
                    id=str(uuid.uuid4()),  # We don't store the UUID in cache
                    name=row["name"],
                    display_name=row["display_name"],
                    description=row["description"] or "",
                    parent_domain=row["parent_domain"],
                    keywords=json.loads(row["keywords"]) if row["keywords"] else [],
                    trigger_phrases=json.loads(row["trigger_phrases"]) if row["trigger_phrases"] else [],
                    confidence_threshold=row["confidence_threshold"],
                    is_active=row["is_active"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                self._domain_cache[domain.name] = domain
            
            logger.info(f"Loaded {len(self._domain_cache)} domains into cache")
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
    
    # ========== DOMAIN MANAGEMENT ==========
    
    async def create_domain(self, domain: DomainDefinition) -> str:
        """Create a new dynamic domain definition."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO domain_definitions (
                    name, display_name, description, parent_domain, keywords, 
                    trigger_phrases, confidence_threshold, is_active
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                domain.name,
                domain.display_name,
                domain.description,
                domain.parent_domain,
                json.dumps(domain.keywords),
                json.dumps(domain.trigger_phrases),
                domain.confidence_threshold,
                domain.is_active
            )
            
            # Update cache
            self._domain_cache[domain.name] = domain
            
            logger.info(f"Created domain: {domain.name}")
            return domain.name
    
    async def get_domain(self, domain_name: str) -> Optional[DomainDefinition]:
        """Get domain definition by name."""
        return self._domain_cache.get(domain_name)
    
    async def list_domains(self) -> List[DomainDefinition]:
        """List all active domains."""
        return [domain for domain in self._domain_cache.values() if domain.is_active]
    
    async def classify_domain(self, text: str, threshold: float = 0.7) -> Optional[str]:
        """
        Classify text into a domain using keyword matching.
        In production, this would use vector similarity with embeddings.
        """
        text_lower = text.lower()
        best_match = None
        best_score = 0.0
        
        for domain in self._domain_cache.values():
            if not domain.is_active:
                continue
                
            score = 0.0
            
            # Check trigger phrases (higher weight)
            for phrase in domain.trigger_phrases:
                if phrase.lower() in text_lower:
                    score += 2.0
            
            # Check keywords
            for keyword in domain.keywords:
                if keyword.lower() in text_lower:
                    score += 1.0
            
            # Normalize by total possible score
            max_score = len(domain.trigger_phrases) * 2.0 + len(domain.keywords) * 1.0
            if max_score > 0:
                score = score / max_score
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = domain.name
        
        return best_match

    # ========== SEMANTIC MEMORY ==========

    async def add_semantic_entry(self, entry: SemanticMemoryEntry) -> str:
        """Add a semantic memory entry."""
        async with self.pool.acquire() as conn:
            entry_id = await conn.fetchval("""
                INSERT INTO semantic_memory (
                    entity_type, entity_name, entity_description, domain, properties, synonyms
                ) VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """,
                entry.entity_type,
                entry.entity_name,
                entry.entity_description,
                entry.domain,
                json.dumps(entry.properties),
                json.dumps(entry.synonyms)
            )
            return str(entry_id)

    async def search_semantic_memory(self, query: str, domain: Optional[str] = None, limit: int = 10) -> List[SemanticMemoryEntry]:
        """Search semantic memory by text similarity."""
        async with self.pool.acquire() as conn:
            # For now, use text search. In production, use vector similarity
            where_clause = "WHERE (entity_name ILIKE $1 OR entity_description ILIKE $1 OR synonyms::text ILIKE $1)"
            params = [f"%{query}%"]

            if domain:
                where_clause += " AND domain = $2"
                params.append(domain)

            query_sql = f"""
                SELECT id, entity_type, entity_name, entity_description, domain, properties, synonyms, created_at, updated_at
                FROM semantic_memory
                {where_clause}
                ORDER BY created_at DESC
                LIMIT {limit}
            """

            rows = await conn.fetch(query_sql, *params)

            entries = []
            for row in rows:
                entry = SemanticMemoryEntry(
                    id=str(row["id"]),
                    entity_type=row["entity_type"],
                    entity_name=row["entity_name"],
                    entity_description=row["entity_description"] or "",
                    domain=row["domain"],
                    properties=json.loads(row["properties"]) if row["properties"] else {},
                    synonyms=json.loads(row["synonyms"]) if row["synonyms"] else [],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                entries.append(entry)

            return entries

    # ========== EPISODIC MEMORY ==========

    async def get_or_create_conversation(self, user_id: str, conversation_id: Optional[str] = None) -> ConversationContext:
        """Get existing conversation or create new one."""
        if conversation_id:
            context = await self.get_conversation_context(conversation_id)
            if context:
                return context

        # Create new conversation
        new_conversation_id = str(uuid.uuid4())

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO conversation_contexts (conversation_id, user_id)
                VALUES ($1, $2)
            """, uuid.UUID(new_conversation_id), user_id)

        return ConversationContext(
            conversation_id=new_conversation_id,
            user_id=user_id
        )

    async def get_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context with recent messages."""
        # Convert conversation_id to UUID if it's not already
        try:
            conversation_uuid = uuid.UUID(conversation_id)
        except ValueError:
            # If it's not a valid UUID, treat it as a string-based conversation ID
            # For testing purposes, we'll return None and let the system create a minimal context
            logger.debug(f"Invalid UUID format for conversation_id: {conversation_id}, returning None")
            return None

        async with self.pool.acquire() as conn:
            # Get conversation metadata
            context_row = await conn.fetchrow("""
                SELECT conversation_id, user_id, domain, route, persistent_slots, active_plugins, created_at, last_updated
                FROM conversation_contexts
                WHERE conversation_id = $1
            """, conversation_uuid)

            if not context_row:
                return None

            # Get recent messages (last 20 as per Moveworks standard)
            message_rows = await conn.fetch("""
                SELECT id, conversation_id, user_id, message_type, content, intent,
                       entities_extracted, slot_values, plugin_calls, timestamp, sequence_number
                FROM episodic_memory
                WHERE conversation_id = $1
                ORDER BY sequence_number DESC
                LIMIT 20
            """, conversation_uuid)

            # Build conversation context
            messages = []
            for row in message_rows:
                message = EpisodicMemoryEntry(
                    id=str(row["id"]),
                    conversation_id=str(row["conversation_id"]),
                    user_id=row["user_id"],
                    message_type=row["message_type"],
                    content=row["content"],
                    intent=row["intent"],
                    entities_extracted=json.loads(row["entities_extracted"]) if row["entities_extracted"] else {},
                    slot_values=json.loads(row["slot_values"]) if row["slot_values"] else {},
                    plugin_calls=json.loads(row["plugin_calls"]) if row["plugin_calls"] else [],
                    timestamp=row["timestamp"],
                    sequence_number=row["sequence_number"]
                )
                messages.append(message)

            # Reverse to get chronological order
            messages.reverse()

            context = ConversationContext(
                conversation_id=str(context_row["conversation_id"]),
                user_id=context_row["user_id"],
                domain=context_row["domain"],
                route=context_row["route"] or "",
                messages=messages,
                persistent_slots=json.loads(context_row["persistent_slots"]) if context_row["persistent_slots"] else {},
                active_plugins=json.loads(context_row["active_plugins"]) if context_row["active_plugins"] else [],
                created_at=context_row["created_at"],
                last_updated=context_row["last_updated"]
            )

            return context

    async def add_message(self, conversation_id: str, message: EpisodicMemoryEntry):
        """Add message to episodic memory with window management."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Get current message count
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM episodic_memory WHERE conversation_id = $1
                """, uuid.UUID(conversation_id))

                message.sequence_number = count

                # Insert new message
                await conn.execute("""
                    INSERT INTO episodic_memory (
                        conversation_id, user_id, message_type, content, intent,
                        entities_extracted, slot_values, plugin_calls, sequence_number
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    uuid.UUID(conversation_id),
                    message.user_id,
                    message.message_type,
                    message.content,
                    message.intent,
                    json.dumps(message.entities_extracted),
                    json.dumps(message.slot_values),
                    json.dumps(message.plugin_calls),
                    message.sequence_number
                )

                # Maintain window size (keep last N messages - configurable)
                config = get_reasoning_config()
                await conn.execute(f"""
                    DELETE FROM episodic_memory
                    WHERE conversation_id = $1
                    AND sequence_number < (
                        SELECT MAX(sequence_number) - {config.memory_window_size - 1}
                        FROM episodic_memory
                        WHERE conversation_id = $1
                    )
                """, uuid.UUID(conversation_id))

                # Update conversation last_updated
                await conn.execute("""
                    UPDATE conversation_contexts
                    SET last_updated = NOW()
                    WHERE conversation_id = $1
                """, uuid.UUID(conversation_id))

    # ========== PROCEDURE MEMORY ==========

    async def add_procedure_entry(self, entry: ProcedureMemoryEntry) -> str:
        """Add a procedure memory entry for plugin capabilities."""
        async with self.pool.acquire() as conn:
            entry_id = await conn.fetchval("""
                INSERT INTO procedure_memory (
                    plugin_id, plugin_name, description, capabilities, trigger_utterances,
                    domain_compatibility, required_slots, output_types, business_rules, confidence_threshold
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
            """,
                entry.plugin_id,
                entry.plugin_name,
                entry.description,
                json.dumps(entry.capabilities),
                json.dumps(entry.trigger_utterances),
                json.dumps(entry.domain_compatibility),
                json.dumps(entry.required_slots),
                json.dumps(entry.output_types),
                json.dumps(entry.business_rules),
                entry.confidence_threshold
            )
            return str(entry_id)

    async def search_procedures(self, query: str, domain: Optional[str] = None, limit: int = 10) -> List[ProcedureMemoryEntry]:
        """Search procedure memory for matching capabilities."""
        async with self.pool.acquire() as conn:
            # Text-based search for capabilities and triggers
            where_clause = """
                WHERE (plugin_name ILIKE $1 OR description ILIKE $1 OR
                       capabilities::text ILIKE $1 OR trigger_utterances::text ILIKE $1)
            """
            params = [f"%{query}%"]

            if domain:
                where_clause += " AND domain_compatibility::jsonb ? $2"
                params.append(domain)

            query_sql = f"""
                SELECT id, plugin_id, plugin_name, description, capabilities, trigger_utterances,
                       domain_compatibility, required_slots, output_types, business_rules,
                       confidence_threshold, created_at, updated_at
                FROM procedure_memory
                {where_clause}
                ORDER BY created_at DESC
                LIMIT {limit}
            """

            rows = await conn.fetch(query_sql, *params)

            entries = []
            for row in rows:
                entry = ProcedureMemoryEntry(
                    id=str(row["id"]),
                    plugin_id=row["plugin_id"],
                    plugin_name=row["plugin_name"],
                    description=row["description"] or "",
                    capabilities=json.loads(row["capabilities"]) if row["capabilities"] else [],
                    trigger_utterances=json.loads(row["trigger_utterances"]) if row["trigger_utterances"] else [],
                    domain_compatibility=json.loads(row["domain_compatibility"]) if row["domain_compatibility"] else [],
                    required_slots=json.loads(row["required_slots"]) if row["required_slots"] else [],
                    output_types=json.loads(row["output_types"]) if row["output_types"] else [],
                    business_rules=json.loads(row["business_rules"]) if row["business_rules"] else {},
                    confidence_threshold=row["confidence_threshold"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                entries.append(entry)

            return entries

    # ========== WORKING MEMORY ==========

    async def create_working_memory(self, entry: WorkingMemoryEntry) -> str:
        """Create a new working memory entry for process tracking."""
        async with self.pool.acquire() as conn:
            entry_id = await conn.fetchval("""
                INSERT INTO working_memory (
                    conversation_id, process_id, process_name, current_step, status,
                    variables, business_objects, step_history, reference_links
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """,
                uuid.UUID(entry.conversation_id),
                entry.process_id,
                entry.process_name,
                entry.current_step,
                entry.status,
                json.dumps(entry.variables),
                json.dumps(entry.business_objects),
                json.dumps(entry.step_history),
                json.dumps(entry.references)
            )
            return str(entry_id)

    async def update_working_memory(self, entry_id: str, entry: WorkingMemoryEntry) -> bool:
        """Update working memory entry with new state."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE working_memory
                SET current_step = $2, status = $3, variables = $4, business_objects = $5,
                    step_history = $6, reference_links = $7, updated_at = NOW()
                WHERE id = $1
            """,
                uuid.UUID(entry_id),
                entry.current_step,
                entry.status,
                json.dumps(entry.variables),
                json.dumps(entry.business_objects),
                json.dumps(entry.step_history),
                json.dumps(entry.references)
            )
            return result == "UPDATE 1"

    async def get_active_working_memory(self, conversation_id: str) -> List[WorkingMemoryEntry]:
        """Get active working memory entries for a conversation."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, conversation_id, process_id, process_name, current_step, status,
                       variables, business_objects, step_history, reference_links, created_at, updated_at
                FROM working_memory
                WHERE conversation_id = $1 AND status IN ('pending', 'in_progress')
                ORDER BY created_at DESC
            """, uuid.UUID(conversation_id))

            entries = []
            for row in rows:
                entry = WorkingMemoryEntry(
                    id=str(row["id"]),
                    conversation_id=str(row["conversation_id"]),
                    process_id=row["process_id"],
                    process_name=row["process_name"] or "",
                    current_step=row["current_step"] or "",
                    status=row["status"],
                    variables=json.loads(row["variables"]) if row["variables"] else {},
                    business_objects=json.loads(row["business_objects"]) if row["business_objects"] else {},
                    step_history=json.loads(row["step_history"]) if row["step_history"] else [],
                    references=json.loads(row["reference_links"]) if row["reference_links"] else [],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                entries.append(entry)

            return entries

    # ========== MEMORY SNAPSHOT ==========

    async def get_memory_snapshot(self, conversation_id: str, user_context: Dict[str, Any]) -> Optional[MemorySnapshot]:
        """Get memory snapshot for reasoning engine - alias for create_memory_snapshot."""
        try:
            return await self.create_memory_snapshot(conversation_id, user_context.get("query"))
        except Exception as e:
            logger.warning(f"Failed to get memory snapshot: {e}")
            return None

    async def create_memory_snapshot(self, conversation_id: str, query: Optional[str] = None) -> MemorySnapshot:
        """Create a complete memory snapshot for reasoning engine."""
        # Get conversation context
        context = await self.get_conversation_context(conversation_id)
        if not context:
            # Create minimal context if not found
            context = ConversationContext(conversation_id=conversation_id, user_id="unknown")

        # Get relevant semantic entries
        semantic_entries = []
        if query:
            semantic_entries = await self.search_semantic_memory(query, context.domain, limit=5)

        # Get available procedures
        procedure_entries = []
        if query:
            procedure_entries = await self.search_procedures(query, context.domain, limit=10)

        # Get active working memory
        working_memory = await self.get_active_working_memory(conversation_id)

        return MemorySnapshot(
            conversation_context=context,
            relevant_semantic_entries=semantic_entries,
            available_procedures=procedure_entries,
            active_working_memory=working_memory
        )


# Global memory manager instance
moveworks_memory_manager = MoveworksMemoryManager()

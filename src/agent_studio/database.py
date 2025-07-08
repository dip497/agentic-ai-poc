"""
PostgreSQL database setup for Agent Studio.
Handles conversational processes, connectors, and test results.
"""

import os
import asyncio
import asyncpg
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


class AgentStudioDatabase:
    """PostgreSQL database manager for Agent Studio."""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/agent_studio"
        )
    
    async def initialize(self):
        """Initialize database connection and create tables."""
        try:
            # Create database if it doesn't exist
            await self._create_database_if_not_exists()
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Create tables
            await self._create_tables()
            
            # Insert sample data
            await self._insert_sample_data()
            
            logger.info("Agent Studio database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def _create_database_if_not_exists(self):
        """Create the agent_studio database if it doesn't exist."""
        try:
            # Connect to default postgres database to create agent_studio database
            default_url = self.database_url.replace("/agent_studio", "/postgres")
            conn = await asyncpg.connect(default_url)
            
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = 'agent_studio'"
            )
            
            if not exists:
                await conn.execute("CREATE DATABASE agent_studio")
                logger.info("Created agent_studio database")
            
            await conn.close()
            
        except Exception as e:
            logger.warning(f"Could not create database (may already exist): {e}")
    
    async def _create_tables(self):
        """Create all necessary tables."""
        async with self.pool.acquire() as conn:
            # Enable UUID extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
            
            # Conversational Processes table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversational_processes (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    version VARCHAR(50) DEFAULT '1.0.0',
                    status VARCHAR(50) DEFAULT 'draft',
                    triggers JSONB DEFAULT '[]',
                    keywords JSONB DEFAULT '[]',
                    activities JSONB DEFAULT '[]',
                    slots JSONB DEFAULT '[]',
                    required_connectors JSONB DEFAULT '[]',
                    permissions JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    created_by VARCHAR(255) DEFAULT 'system'
                )
            """)
            
            # Connectors table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS connectors (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    type VARCHAR(50) NOT NULL,
                    base_url VARCHAR(500),
                    auth_type VARCHAR(50) DEFAULT 'none',
                    auth_config JSONB DEFAULT '{}',
                    headers JSONB DEFAULT '{}',
                    available_actions JSONB DEFAULT '[]',
                    status VARCHAR(50) DEFAULT 'testing',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Test Results table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    process_id UUID REFERENCES conversational_processes(id) ON DELETE CASCADE,
                    test_input TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    response TEXT,
                    execution_time FLOAT,
                    steps_executed JSONB DEFAULT '[]',
                    errors JSONB DEFAULT '[]',
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Deployments table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS deployments (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    process_id UUID REFERENCES conversational_processes(id) ON DELETE CASCADE,
                    environment VARCHAR(50) NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    priority INTEGER DEFAULT 1,
                    user_groups JSONB DEFAULT '[]',
                    rate_limit INTEGER,
                    deployed_at TIMESTAMP DEFAULT NOW(),
                    deployed_by VARCHAR(255)
                )
            """)

            # Custom Data Types table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS custom_data_types (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    schema JSONB NOT NULL,
                    default_resolver_strategy VARCHAR(255),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    created_by VARCHAR(255) DEFAULT 'system'
                )
            """)

            # Resolver Strategies table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS resolver_strategies (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name VARCHAR(255) NOT NULL UNIQUE,
                    data_type VARCHAR(255) NOT NULL,
                    description TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Resolver Methods table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS resolver_methods (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    strategy_id UUID REFERENCES resolver_strategies(id) ON DELETE CASCADE,
                    name VARCHAR(255) NOT NULL,
                    method_type VARCHAR(50) NOT NULL CHECK (method_type IN ('Static', 'Dynamic')),
                    description TEXT,
                    static_options JSONB,
                    action_name VARCHAR(255),
                    input_arguments JSONB,
                    output_mapping TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy_id, name)
                )
            """)
            
            # Create indexes for better performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_processes_status ON conversational_processes(status);
                CREATE INDEX IF NOT EXISTS idx_processes_created_by ON conversational_processes(created_by);
                CREATE INDEX IF NOT EXISTS idx_connectors_type ON connectors(type);
                CREATE INDEX IF NOT EXISTS idx_connectors_status ON connectors(status);
                CREATE INDEX IF NOT EXISTS idx_test_results_process_id ON test_results(process_id);
                CREATE INDEX IF NOT EXISTS idx_deployments_process_id ON deployments(process_id);
                CREATE INDEX IF NOT EXISTS idx_deployments_environment ON deployments(environment);
            """)
            
            logger.info("Database tables created successfully")
    
    async def _insert_sample_data(self):
        """Insert sample data if tables are empty."""
        async with self.pool.acquire() as conn:
            # Check if we already have data
            process_count = await conn.fetchval("SELECT COUNT(*) FROM conversational_processes")
            connector_count = await conn.fetchval("SELECT COUNT(*) FROM connectors")
            
            if process_count == 0 and connector_count == 0:
                # Insert sample connector
                connector_id = await conn.fetchval("""
                    INSERT INTO connectors (name, description, type, base_url, auth_type, available_actions)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """, 
                    "Sample REST API",
                    "Sample REST API connector for testing",
                    "http",
                    "https://jsonplaceholder.typicode.com",
                    "none",
                    json.dumps([
                        {
                            "name": "get_user",
                            "description": "Get user by ID",
                            "endpoint": "/users/{user_id}",
                            "method": "GET",
                            "parameters": {"user_id": "string"}
                        },
                        {
                            "name": "create_post",
                            "description": "Create a new post",
                            "endpoint": "/posts",
                            "method": "POST",
                            "parameters": {"title": "string", "body": "string", "userId": "number"}
                        }
                    ])
                )
                
                # Insert sample process
                await conn.execute("""
                    INSERT INTO conversational_processes (
                        name, description, triggers, keywords, activities, slots, required_connectors, permissions
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    "User Lookup",
                    "Look up user information by ID",
                    json.dumps(["find user", "lookup user", "get user info"]),
                    json.dumps(["user", "lookup", "find", "information"]),
                    json.dumps([
                        {
                            "name": "collect_user_id",
                            "description": "Collect the user ID from the user",
                            "type": "slot_collection",
                            "slot_name": "user_id"
                        },
                        {
                            "name": "lookup_user",
                            "description": "Look up the user in the system",
                            "type": "http_action",
                            "connector_name": "Sample REST API",
                            "endpoint": "/users/{user_id}",
                            "method": "GET",
                            "parameters": {"user_id": "{user_id}"}
                        },
                        {
                            "name": "respond_with_user_info",
                            "description": "Respond with the user information",
                            "type": "content_action",
                            "content_template": "Found user: {name} ({email}). They are located in {address.city}."
                        }
                    ]),
                    json.dumps([
                        {
                            "name": "user_id",
                            "description": "The ID of the user to look up",
                            "type": "number",
                            "required": True,
                            "prompt_text": "What is the user ID you want to look up?"
                        }
                    ]),
                    json.dumps(["Sample REST API"]),
                    json.dumps({"all_users": True})
                )
                
                logger.info("Sample data inserted successfully")
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
    
    # ========== CONVERSATIONAL PROCESSES ==========
    
    async def create_process(self, process_data: Dict[str, Any]) -> str:
        """Create a new conversational process."""
        async with self.pool.acquire() as conn:
            process_id = await conn.fetchval("""
                INSERT INTO conversational_processes (
                    name, description, triggers, keywords, activities, slots, 
                    required_connectors, permissions, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """,
                process_data.get("name", ""),
                process_data.get("description", ""),
                json.dumps(process_data.get("triggers", [])),
                json.dumps(process_data.get("keywords", [])),
                json.dumps(process_data.get("activities", [])),
                json.dumps(process_data.get("slots", [])),
                json.dumps(process_data.get("required_connectors", [])),
                json.dumps(process_data.get("permissions", {})),
                process_data.get("created_by", "system")
            )
            return str(process_id)
    
    async def get_process(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversational process by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM conversational_processes WHERE id = $1
            """, uuid.UUID(process_id))

            if row:
                process_dict = dict(row)
                # Parse JSONB fields that are returned as strings
                jsonb_fields = ['triggers', 'keywords', 'activities', 'slots', 'required_connectors', 'permissions']
                for field in jsonb_fields:
                    if field in process_dict and isinstance(process_dict[field], str):
                        try:
                            process_dict[field] = json.loads(process_dict[field])
                        except (json.JSONDecodeError, TypeError):
                            # If parsing fails, keep as empty list/dict
                            if field == 'permissions':
                                process_dict[field] = {}
                            else:
                                process_dict[field] = []
                return process_dict
            return None
    
    async def list_processes(self) -> List[Dict[str, Any]]:
        """List all conversational processes."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM conversational_processes ORDER BY created_at DESC
            """)

            processes = []
            for row in rows:
                process_dict = dict(row)
                # Parse JSONB fields that are returned as strings
                jsonb_fields = ['triggers', 'keywords', 'activities', 'slots', 'required_connectors', 'permissions']
                for field in jsonb_fields:
                    if field in process_dict and isinstance(process_dict[field], str):
                        try:
                            process_dict[field] = json.loads(process_dict[field])
                        except (json.JSONDecodeError, TypeError):
                            # If parsing fails, keep as empty list/dict
                            if field == 'permissions':
                                process_dict[field] = {}
                            else:
                                process_dict[field] = []
                processes.append(process_dict)

            return processes
    
    async def update_process(self, process_id: str, process_data: Dict[str, Any]) -> bool:
        """Update a conversational process."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE conversational_processes
                SET name = $2, description = $3, triggers = $4, keywords = $5,
                    activities = $6, slots = $7, required_connectors = $8,
                    permissions = $9, updated_at = NOW()
                WHERE id = $1
            """,
                uuid.UUID(process_id),
                process_data.get("name"),
                process_data.get("description"),
                json.dumps(process_data.get("triggers", [])),
                json.dumps(process_data.get("keywords", [])),
                json.dumps(process_data.get("activities", [])),
                json.dumps(process_data.get("slots", [])),
                json.dumps(process_data.get("required_connectors", [])),
                json.dumps(process_data.get("permissions", {}))
            )
            return result == "UPDATE 1"
    
    async def delete_process(self, process_id: str) -> bool:
        """Delete a conversational process."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM conversational_processes WHERE id = $1
            """, uuid.UUID(process_id))
            return result == "DELETE 1"
    
    # ========== CONNECTORS ==========
    
    async def create_connector(self, connector_data: Dict[str, Any]) -> str:
        """Create a new connector."""
        async with self.pool.acquire() as conn:
            connector_id = await conn.fetchval("""
                INSERT INTO connectors (
                    name, description, type, base_url, auth_type, 
                    auth_config, headers, available_actions
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """,
                connector_data.get("name", ""),
                connector_data.get("description", ""),
                connector_data.get("type", "http"),
                connector_data.get("base_url", ""),
                connector_data.get("auth_type", "none"),
                json.dumps(connector_data.get("auth_config", {})),
                json.dumps(connector_data.get("headers", {})),
                json.dumps(connector_data.get("available_actions", []))
            )
            return str(connector_id)
    
    async def get_connector(self, connector_id: str) -> Optional[Dict[str, Any]]:
        """Get a connector by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM connectors WHERE id = $1
            """, uuid.UUID(connector_id))
            
            if row:
                return dict(row)
            return None
    
    async def list_connectors(self) -> List[Dict[str, Any]]:
        """List all connectors."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM connectors ORDER BY created_at DESC
            """)
            return [dict(row) for row in rows]
    
    async def delete_connector(self, connector_id: str) -> bool:
        """Delete a connector."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM connectors WHERE id = $1
            """, uuid.UUID(connector_id))
            return result == "DELETE 1"
    
    # ========== TEST RESULTS ==========
    
    async def add_test_result(self, result_data: Dict[str, Any]):
        """Add a test result."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO test_results (
                    process_id, test_input, success, response, execution_time,
                    steps_executed, errors
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                uuid.UUID(result_data["process_id"]),
                result_data["test_input"],
                result_data["success"],
                result_data.get("response", ""),
                result_data.get("execution_time", 0.0),
                json.dumps(result_data.get("steps_executed", [])),
                json.dumps(result_data.get("errors", []))
            )
    
    async def get_test_results(self, process_id: str) -> List[Dict[str, Any]]:
        """Get test results for a process."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM test_results
                WHERE process_id = $1
                ORDER BY timestamp DESC
            """, uuid.UUID(process_id))

            # Convert rows to dict and ensure JSONB fields are properly parsed
            results = []
            for row in rows:
                row_dict = dict(row)
                # Parse JSONB fields if they're strings
                if isinstance(row_dict.get('steps_executed'), str):
                    try:
                        row_dict['steps_executed'] = json.loads(row_dict['steps_executed'])
                    except (json.JSONDecodeError, TypeError):
                        row_dict['steps_executed'] = []

                if isinstance(row_dict.get('errors'), str):
                    try:
                        row_dict['errors'] = json.loads(row_dict['errors'])
                    except (json.JSONDecodeError, TypeError):
                        row_dict['errors'] = []

                results.append(row_dict)

            return results
    
    # ========== DEPLOYMENTS ==========
    
    async def create_deployment(self, deployment_data: Dict[str, Any]) -> bool:
        """Create a deployment configuration."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO deployments (
                    process_id, environment, enabled, user_groups, deployed_by
                ) VALUES ($1, $2, $3, $4, $5)
            """,
                uuid.UUID(deployment_data["process_id"]),
                deployment_data["environment"],
                deployment_data.get("enabled", True),
                json.dumps(deployment_data.get("user_groups", [])),
                deployment_data.get("deployed_by", "system")
            )
            return True
    
    async def get_deployment(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment configuration for a process."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM deployments WHERE process_id = $1
            """, uuid.UUID(process_id))
            
            if row:
                return dict(row)
            return None

    # ========== CUSTOM DATA TYPES ==========

    async def list_custom_data_types(self) -> List[Dict[str, Any]]:
        """List all custom data types."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id::text as id, name, description, schema, default_resolver_strategy,
                       created_at::text as created_at, updated_at::text as updated_at
                FROM custom_data_types
                ORDER BY name
            """)
            return [dict(row) for row in rows]

    async def create_custom_data_type(self, name: str, description: str,
                                    schema: Dict[str, Any],
                                    default_resolver_strategy: Optional[str] = None) -> str:
        """Create a new custom data type."""
        async with self.pool.acquire() as conn:
            data_type_id = await conn.fetchval("""
                INSERT INTO custom_data_types (name, description, schema, default_resolver_strategy)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, name, description, json.dumps(schema), default_resolver_strategy)
            return str(data_type_id)

    async def get_custom_data_type(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific custom data type by name."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT name, description, schema, default_resolver_strategy,
                       created_at, updated_at
                FROM custom_data_types
                WHERE name = $1
            """, name)
            if row:
                result = dict(row)
                result['schema'] = json.loads(result['schema'])
                return result
            return None

    async def update_custom_data_type(self, current_name: str, name: Optional[str] = None,
                                    description: Optional[str] = None,
                                    schema: Optional[Dict[str, Any]] = None,
                                    default_resolver_strategy: Optional[str] = None) -> bool:
        """Update a custom data type."""
        async with self.pool.acquire() as conn:
            # Build dynamic update query
            updates = []
            params = []
            param_count = 1

            if name is not None:
                updates.append(f"name = ${param_count}")
                params.append(name)
                param_count += 1

            if description is not None:
                updates.append(f"description = ${param_count}")
                params.append(description)
                param_count += 1

            if schema is not None:
                updates.append(f"schema = ${param_count}")
                params.append(json.dumps(schema))
                param_count += 1

            if default_resolver_strategy is not None:
                updates.append(f"default_resolver_strategy = ${param_count}")
                params.append(default_resolver_strategy)
                param_count += 1

            if not updates:
                return True  # No updates to make

            updates.append(f"updated_at = NOW()")
            params.append(current_name)

            query = f"""
                UPDATE custom_data_types
                SET {', '.join(updates)}
                WHERE name = ${param_count}
            """

            result = await conn.execute(query, *params)
            return result != "UPDATE 0"

    async def delete_custom_data_type(self, name: str) -> bool:
        """Delete a custom data type."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM custom_data_types WHERE name = $1
            """, name)
            return result != "DELETE 0"

    # ========== RESOLVER STRATEGIES ==========

    async def list_resolver_strategies(self) -> List[Dict[str, Any]]:
        """List all resolver strategies with their methods."""
        async with self.pool.acquire() as conn:
            strategies = await conn.fetch("""
                SELECT rs.id::text as id,
                       rs.name,
                       rs.data_type,
                       rs.description,
                       rs.created_at::text as created_at,
                       rs.updated_at::text as updated_at,
                       rs.metadata,
                       COALESCE(
                           json_agg(
                               json_build_object(
                                   'id', rm.id::text,
                                   'name', rm.name,
                                   'method_type', rm.method_type,
                                   'description', rm.description,
                                   'static_options', rm.static_options,
                                   'action_name', rm.action_name,
                                   'input_arguments', rm.input_arguments,
                                   'output_mapping', rm.output_mapping,
                                   'created_at', rm.created_at::text,
                                   'updated_at', rm.updated_at::text
                               ) ORDER BY rm.created_at
                           ) FILTER (WHERE rm.id IS NOT NULL),
                           '[]'::json
                       ) as methods
                FROM resolver_strategies rs
                LEFT JOIN resolver_methods rm ON rs.id = rm.strategy_id
                GROUP BY rs.id, rs.name, rs.data_type, rs.description, rs.created_at, rs.updated_at, rs.metadata
                ORDER BY rs.created_at DESC
            """)

            return [dict(strategy) for strategy in strategies]

    async def create_resolver_strategy(self, name: str, data_type: str, description: str, methods: List[Dict[str, Any]]) -> str:
        """Create a new resolver strategy with methods."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Create strategy
                strategy_id = await conn.fetchval("""
                    INSERT INTO resolver_strategies (name, data_type, description)
                    VALUES ($1, $2, $3)
                    RETURNING id
                """, name, data_type, description)

                # Create methods
                for method in methods:
                    await conn.execute("""
                        INSERT INTO resolver_methods (
                            strategy_id, name, method_type, description,
                            static_options, action_name, input_arguments, output_mapping
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                        strategy_id,
                        method["name"],
                        method["method_type"],
                        method.get("description"),
                        json.dumps(method.get("static_options")) if method.get("static_options") else None,
                        method.get("action_name"),
                        json.dumps(method.get("input_arguments")) if method.get("input_arguments") else None,
                        method.get("output_mapping")
                    )

                return str(strategy_id)

    async def get_resolver_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a resolver strategy by name with its methods."""
        async with self.pool.acquire() as conn:
            strategy = await conn.fetchrow("""
                SELECT rs.*,
                       COALESCE(
                           json_agg(
                               json_build_object(
                                   'id', rm.id,
                                   'name', rm.name,
                                   'method_type', rm.method_type,
                                   'description', rm.description,
                                   'static_options', rm.static_options,
                                   'action_name', rm.action_name,
                                   'input_arguments', rm.input_arguments,
                                   'output_mapping', rm.output_mapping,
                                   'created_at', rm.created_at,
                                   'updated_at', rm.updated_at
                               ) ORDER BY rm.created_at
                           ) FILTER (WHERE rm.id IS NOT NULL),
                           '[]'::json
                       ) as methods
                FROM resolver_strategies rs
                LEFT JOIN resolver_methods rm ON rs.id = rm.strategy_id
                WHERE rs.name = $1
                GROUP BY rs.id, rs.name, rs.data_type, rs.description, rs.created_at, rs.updated_at, rs.metadata
            """, name)

            return dict(strategy) if strategy else None

    async def update_resolver_strategy(self, current_name: str, name: Optional[str] = None,
                                     data_type: Optional[str] = None, description: Optional[str] = None,
                                     methods: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Update a resolver strategy."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Update strategy
                updates = []
                params = []
                param_count = 2

                if name is not None:
                    updates.append(f"name = ${param_count}")
                    params.append(name)
                    param_count += 1

                if data_type is not None:
                    updates.append(f"data_type = ${param_count}")
                    params.append(data_type)
                    param_count += 1

                if description is not None:
                    updates.append(f"description = ${param_count}")
                    params.append(description)
                    param_count += 1

                if updates:
                    updates.append(f"updated_at = CURRENT_TIMESTAMP")
                    query = f"""
                        UPDATE resolver_strategies
                        SET {', '.join(updates)}
                        WHERE name = $1
                    """
                    result = await conn.execute(query, current_name, *params)
                    if result == "UPDATE 0":
                        return False

                # Update methods if provided
                if methods is not None:
                    # Get strategy ID
                    strategy_id = await conn.fetchval("""
                        SELECT id FROM resolver_strategies WHERE name = $1
                    """, name or current_name)

                    if strategy_id:
                        # Delete existing methods
                        await conn.execute("""
                            DELETE FROM resolver_methods WHERE strategy_id = $1
                        """, strategy_id)

                        # Insert new methods
                        for method in methods:
                            await conn.execute("""
                                INSERT INTO resolver_methods (
                                    strategy_id, name, method_type, description,
                                    static_options, action_name, input_arguments, output_mapping
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            """,
                                strategy_id,
                                method["name"],
                                method["method_type"],
                                method.get("description"),
                                json.dumps(method.get("static_options")) if method.get("static_options") else None,
                                method.get("action_name"),
                                json.dumps(method.get("input_arguments")) if method.get("input_arguments") else None,
                                method.get("output_mapping")
                            )

                return True

    async def delete_resolver_strategy(self, name: str) -> bool:
        """Delete a resolver strategy and its methods."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM resolver_strategies WHERE name = $1
            """, name)
            return result != "DELETE 0"


# Global database instance
agent_studio_db = AgentStudioDatabase()

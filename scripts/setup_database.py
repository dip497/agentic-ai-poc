#!/usr/bin/env python3
"""
Setup script for Agent Studio PostgreSQL database.
Creates the database and initializes tables.
"""

import asyncio
import os
import sys
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent_studio.database import agent_studio_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_database():
    """Setup the Agent Studio database."""
    try:
        logger.info("Setting up Agent Studio database...")
        
        # Initialize the database
        await agent_studio_db.initialize()
        
        logger.info("Database setup completed successfully!")
        
        # Test the database by listing processes
        processes = await agent_studio_db.list_processes()
        logger.info(f"Found {len(processes)} sample processes in database")
        
        connectors = await agent_studio_db.list_connectors()
        logger.info(f"Found {len(connectors)} sample connectors in database")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise
    finally:
        await agent_studio_db.close()


if __name__ == "__main__":
    # Set default database URL if not provided
    if not os.getenv("DATABASE_URL"):
        os.environ["DATABASE_URL"] = "postgresql://postgres:password@localhost:5432/agent_studio"
        logger.info("Using default database URL: postgresql://postgres:password@localhost:5432/agent_studio")
    
    asyncio.run(setup_database())

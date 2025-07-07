-- Initialize Agent Studio database
-- This script is run automatically when PostgreSQL container starts

-- Create the agent_studio database (if not exists)
SELECT 'CREATE DATABASE agent_studio'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'agent_studio');

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE agent_studio TO postgres;

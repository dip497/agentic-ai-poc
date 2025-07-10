"""
Base connector classes following Moveworks patterns.
Supports HTTP connectors, system connectors, and various authentication methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import aiohttp
import asyncio
from datetime import datetime


class AuthType(Enum):
    """Authentication types supported by connectors."""
    NO_AUTH = "no_auth"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"
    OAUTH2_PASSWORD_GRANT = "oauth2_password_grant"
    OAUTH2_JWT_GRANT = "oauth2_jwt_grant"


@dataclass
class ConnectorConfig:
    """Configuration for a connector."""
    name: str
    description: str
    base_url: str
    auth_type: AuthType
    auth_config: Dict[str, Any]
    headers: Optional[Dict[str, str]] = None
    timeout: int = 30
    retry_count: int = 3


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    execution_time: Optional[float] = None


class BaseConnector(ABC):
    """Base class for all connectors following Moveworks patterns."""
    
    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self):
        """Initialize the connector."""
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._get_default_headers()
        )
    
    async def close(self):
        """Close the connector and cleanup resources."""
        if self.session:
            await self.session.close()
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        headers = {
            "User-Agent": "Moveworks-Agent/1.0",
            "Content-Type": "application/json"
        }
        
        if self.config.headers:
            headers.update(self.config.headers)
        
        # Add authentication headers
        auth_headers = self._get_auth_headers()
        if auth_headers:
            headers.update(auth_headers)
        
        return headers
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on auth type."""
        if self.config.auth_type == AuthType.NO_AUTH:
            return {}
        
        elif self.config.auth_type == AuthType.API_KEY:
            header_name = self.config.auth_config.get("header_name", "Authorization")
            api_key = self.config.auth_config.get("api_key")
            pattern = self.config.auth_config.get("pattern", "Bearer %s")
            
            if api_key:
                return {header_name: pattern % api_key}
        
        elif self.config.auth_type == AuthType.BEARER_TOKEN:
            token = self.config.auth_config.get("token")
            if token:
                return {"Authorization": f"Bearer {token}"}
        
        elif self.config.auth_type == AuthType.BASIC_AUTH:
            import base64
            username = self.config.auth_config.get("username")
            password = self.config.auth_config.get("password")
            
            if username and password:
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                return {"Authorization": f"Basic {credentials}"}
        
        return {}
    
    @abstractmethod
    async def execute_action(self, action_name: str, input_args: Dict[str, Any]) -> ActionResult:
        """Execute an action using this connector."""
        pass
    
    async def test_connection(self) -> ActionResult:
        """Test the connector connection."""
        try:
            if not self.session:
                await self.initialize()
            
            # Simple health check
            async with self.session.get(self.config.base_url) as response:
                return ActionResult(
                    success=response.status < 400,
                    status_code=response.status,
                    data={"message": "Connection test successful"}
                )
        except Exception as e:
            return ActionResult(
                success=False,
                error=str(e),
                data={"message": "Connection test failed"}
            )


class HTTPConnector(BaseConnector):
    """HTTP connector for making REST API calls."""
    
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self.actions: Dict[str, Dict[str, Any]] = {}
    
    def register_action(self, action_name: str, action_config: Dict[str, Any]):
        """Register an HTTP action with this connector."""
        self.actions[action_name] = action_config
    
    async def execute_action(self, action_name: str, input_args: Dict[str, Any]) -> ActionResult:
        """Execute an HTTP action."""
        if action_name not in self.actions:
            return ActionResult(
                success=False,
                error=f"Action '{action_name}' not found in connector '{self.config.name}'"
            )
        
        action_config = self.actions[action_name]
        
        try:
            start_time = datetime.now()
            
            # Build URL
            endpoint = action_config.get("endpoint", "")
            url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            
            # Replace URL parameters
            for key, value in input_args.items():
                url = url.replace(f"{{{{{key}}}}}", str(value))
            
            # Prepare request
            method = action_config.get("method", "GET").upper()
            params = action_config.get("params", {})
            data = action_config.get("data", {})
            
            # Replace parameters with input args
            for key, value in input_args.items():
                if key in params:
                    params[key] = value
                if key in data:
                    data[key] = value
            
            if not self.session:
                await self.initialize()
            
            # Make request
            async with self.session.request(
                method=method,
                url=url,
                params=params if method == "GET" else None,
                json=data if method != "GET" and data else None
            ) as response:
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                if response.status < 400:
                    try:
                        response_data = await response.json()
                    except:
                        response_data = await response.text()
                    
                    return ActionResult(
                        success=True,
                        data=response_data,
                        status_code=response.status,
                        execution_time=execution_time
                    )
                else:
                    error_text = await response.text()
                    return ActionResult(
                        success=False,
                        error=f"HTTP {response.status}: {error_text}",
                        status_code=response.status,
                        execution_time=execution_time
                    )
        
        except Exception as e:
            return ActionResult(
                success=False,
                error=str(e)
            )

    async def make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None, data: Dict[str, Any] = None) -> ActionResult:
        """Make a direct HTTP request."""
        try:
            if not self.session:
                await self.initialize()

            # Build URL
            url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

            # Make request
            async with self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=data if method.upper() != "GET" and data else None
            ) as response:

                if response.status < 400:
                    try:
                        response_data = await response.json()
                    except:
                        response_data = await response.text()

                    return ActionResult(
                        success=True,
                        data=response_data,
                        status_code=response.status
                    )
                else:
                    error_text = await response.text()
                    return ActionResult(
                        success=False,
                        error=f"HTTP {response.status}: {error_text}",
                        status_code=response.status
                    )

        except Exception as e:
            return ActionResult(
                success=False,
                error=str(e)
            )


class SystemConnector(BaseConnector):
    """System connector for integrating with enterprise systems."""
    
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self.system_type = config.auth_config.get("system_type", "generic")
    
    async def execute_action(self, action_name: str, input_args: Dict[str, Any]) -> ActionResult:
        """Execute a system-specific action."""
        # This would be implemented based on the specific system
        # For now, return a mock result
        return ActionResult(
            success=True,
            data={
                "message": f"System action '{action_name}' executed",
                "system_type": self.system_type,
                "input_args": input_args
            }
        )


class ConnectorFactory:
    """Factory for creating connectors from database configuration."""

    @staticmethod
    async def create_connector_from_config(connector_config: Dict[str, Any]) -> BaseConnector:
        """Create a connector instance from database configuration."""
        config = ConnectorConfig(
            name=connector_config['name'],
            description=connector_config['description'],
            base_url=connector_config['base_url'],
            auth_type=AuthType(connector_config['auth_type']),
            auth_config=connector_config['auth_config'],
            headers=connector_config.get('headers', {}),
            timeout=connector_config.get('timeout', 30),
            retry_count=connector_config.get('retry_count', 3)
        )

        # Create appropriate connector type based on configuration
        connector_type = connector_config.get('type', 'http')

        if connector_type == 'http':
            connector = HTTPConnector(config)

            # Register available actions from configuration
            available_actions = connector_config.get('available_actions', [])
            for action in available_actions:
                if isinstance(action, dict) and 'name' in action:
                    action_config = {
                        'endpoint': action.get('endpoint', ''),
                        'method': action.get('method', 'GET'),
                        'params': action.get('params', {}),
                        'data': action.get('data', {})
                    }
                    connector.register_action(action['name'], action_config)

            return connector

        elif connector_type == 'system':
            return SystemConnector(config)

        else:
            # Default to HTTP connector
            return HTTPConnector(config)


class ConnectorManager:
    """Manages all connectors in the system - now database-driven."""

    def __init__(self):
        self.connectors: Dict[str, BaseConnector] = {}
        self.connector_configs: Dict[str, Dict[str, Any]] = {}

    async def load_connectors_from_database(self):
        """Load all connectors from database configuration."""
        try:
            from src.agent_studio.database import agent_studio_db

            # Ensure database is initialized
            if not hasattr(agent_studio_db, 'pool') or agent_studio_db.pool is None:
                await agent_studio_db.initialize()

            # Load connector configurations
            connector_configs = await agent_studio_db.list_connectors()

            for config in connector_configs:
                try:
                    connector = await ConnectorFactory.create_connector_from_config(config)
                    await connector.initialize()

                    self.connectors[config['name']] = connector
                    self.connector_configs[config['name']] = config

                    print(f"Loaded connector: {config['name']} -> {config['base_url']}")

                except Exception as e:
                    print(f"Failed to load connector {config['name']}: {e}")
                    continue

            print(f"Successfully loaded {len(self.connectors)} connectors from database")

        except Exception as e:
            print(f"Failed to load connectors from database: {e}")

    def register_connector(self, connector: BaseConnector):
        """Register a connector manually."""
        self.connectors[connector.config.name] = connector

    def get_connector(self, name: str) -> Optional[BaseConnector]:
        """Get a connector by name."""
        return self.connectors.get(name)

    async def execute_action(self, connector_name: str, action_name: str, input_args: Dict[str, Any]) -> ActionResult:
        """Execute an action using a specific connector."""
        connector = self.get_connector(connector_name)
        if not connector:
            return ActionResult(
                success=False,
                error=f"Connector '{connector_name}' not found. Available: {list(self.connectors.keys())}"
            )

        return await connector.execute_action(action_name, input_args)

    async def test_all_connections(self) -> Dict[str, ActionResult]:
        """Test all connector connections."""
        results = {}
        for name, connector in self.connectors.items():
            results[name] = await connector.test_connection()
        return results

    async def close_all(self):
        """Close all connectors."""
        for connector in self.connectors.values():
            await connector.close()


# Global connector manager instance
connector_manager = ConnectorManager()

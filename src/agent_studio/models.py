"""
Agent Studio data models for building conversational processes.
Following Moveworks patterns for plugins, activities, and slots.
"""

from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class SlotDefinition:
    """Defines a slot (parameter) that needs to be collected."""
    name: str
    description: str
    type: Literal["string", "number", "boolean", "email", "date", "choice"]
    required: bool = True
    default_value: Optional[Any] = None
    validation_pattern: Optional[str] = None
    choices: Optional[List[str]] = None
    prompt_text: Optional[str] = None


@dataclass
class ActionDefinition:
    """Defines an action that can be executed."""
    name: str
    description: str
    type: Literal["http_action", "content_action", "slot_collection", "conditional"]
    
    # HTTP Action fields
    connector_name: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = "GET"
    headers: Optional[Dict[str, str]] = None
    parameters: Optional[Dict[str, Any]] = None
    
    # Content Action fields
    content_template: Optional[str] = None
    
    # Slot Collection fields
    slot_name: Optional[str] = None
    
    # Conditional fields
    condition: Optional[str] = None
    true_action: Optional[str] = None
    false_action: Optional[str] = None


@dataclass
class ConversationalProcessDefinition:
    """Defines a complete conversational process (plugin)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    # Triggers
    triggers: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Process flow
    activities: List[ActionDefinition] = field(default_factory=list)
    slots: List[SlotDefinition] = field(default_factory=list)
    
    # Configuration
    required_connectors: List[str] = field(default_factory=list)
    permissions: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    status: Literal["draft", "testing", "published", "archived"] = "draft"


@dataclass
class ConnectorDefinition:
    """Defines a connector configuration."""
    type: Literal["http", "system", "database", "api"]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Connection details
    base_url: str = ""
    auth_type: Literal["none", "api_key", "bearer", "basic", "oauth2"] = "none"
    auth_config: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Available actions
    available_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: Literal["active", "inactive", "testing"] = "testing"


@dataclass
class TestResult:
    """Result of testing a conversational process."""
    process_id: str
    test_input: str
    success: bool
    response: str
    execution_time: float
    steps_executed: List[Dict[str, Any]]
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DeploymentConfig:
    """Configuration for deploying a conversational process."""
    process_id: str
    environment: Literal["development", "staging", "production"]
    enabled: bool = True
    priority: int = 1
    user_groups: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None
    deployed_at: Optional[datetime] = None
    deployed_by: Optional[str] = None


class AgentStudioRepository:
    """In-memory repository for Agent Studio data."""
    
    def __init__(self):
        self.processes: Dict[str, ConversationalProcessDefinition] = {}
        self.connectors: Dict[str, ConnectorDefinition] = {}
        self.test_results: Dict[str, List[TestResult]] = {}
        self.deployments: Dict[str, DeploymentConfig] = {}
        
        # Initialize with some sample data
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample processes and connectors."""
        
        # Sample HTTP Connector
        sample_connector = ConnectorDefinition(
            name="Sample REST API",
            description="Sample REST API connector for testing",
            type="http",
            base_url="https://jsonplaceholder.typicode.com",
            auth_type="none",
            available_actions=[
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
            ]
        )
        self.connectors[sample_connector.id] = sample_connector
        
        # Sample Conversational Process
        sample_process = ConversationalProcessDefinition(
            name="User Lookup",
            description="Look up user information by ID",
            triggers=["find user", "lookup user", "get user info"],
            keywords=["user", "lookup", "find", "information"],
            slots=[
                SlotDefinition(
                    name="user_id",
                    description="The ID of the user to look up",
                    type="number",
                    required=True,
                    prompt_text="What is the user ID you want to look up?"
                )
            ],
            activities=[
                ActionDefinition(
                    name="collect_user_id",
                    description="Collect the user ID from the user",
                    type="slot_collection",
                    slot_name="user_id"
                ),
                ActionDefinition(
                    name="lookup_user",
                    description="Look up the user in the system",
                    type="http_action",
                    connector_name="Sample REST API",
                    endpoint="/users/{user_id}",
                    method="GET",
                    parameters={"user_id": "{user_id}"}
                ),
                ActionDefinition(
                    name="respond_with_user_info",
                    description="Respond with the user information",
                    type="content_action",
                    content_template="Found user: {name} ({email}). They are located in {address.city}."
                )
            ],
            required_connectors=["Sample REST API"],
            permissions={"all_users": True}
        )
        self.processes[sample_process.id] = sample_process
    
    # Process CRUD operations
    def create_process(self, process: ConversationalProcessDefinition) -> str:
        """Create a new conversational process."""
        process.created_at = datetime.now()
        process.updated_at = datetime.now()
        self.processes[process.id] = process
        return process.id
    
    def get_process(self, process_id: str) -> Optional[ConversationalProcessDefinition]:
        """Get a conversational process by ID."""
        return self.processes.get(process_id)
    
    def update_process(self, process_id: str, process: ConversationalProcessDefinition) -> bool:
        """Update a conversational process."""
        if process_id in self.processes:
            process.updated_at = datetime.now()
            self.processes[process_id] = process
            return True
        return False
    
    def delete_process(self, process_id: str) -> bool:
        """Delete a conversational process."""
        if process_id in self.processes:
            del self.processes[process_id]
            return True
        return False
    
    def list_processes(self) -> List[ConversationalProcessDefinition]:
        """List all conversational processes."""
        return list(self.processes.values())
    
    # Connector CRUD operations
    def create_connector(self, connector: ConnectorDefinition) -> str:
        """Create a new connector."""
        connector.created_at = datetime.now()
        connector.updated_at = datetime.now()
        self.connectors[connector.id] = connector
        return connector.id
    
    def get_connector(self, connector_id: str) -> Optional[ConnectorDefinition]:
        """Get a connector by ID."""
        return self.connectors.get(connector_id)
    
    def update_connector(self, connector_id: str, connector: ConnectorDefinition) -> bool:
        """Update a connector."""
        if connector_id in self.connectors:
            connector.updated_at = datetime.now()
            self.connectors[connector_id] = connector
            return True
        return False
    
    def delete_connector(self, connector_id: str) -> bool:
        """Delete a connector."""
        if connector_id in self.connectors:
            del self.connectors[connector_id]
            return True
        return False
    
    def list_connectors(self) -> List[ConnectorDefinition]:
        """List all connectors."""
        return list(self.connectors.values())
    
    # Test results
    def add_test_result(self, result: TestResult):
        """Add a test result."""
        if result.process_id not in self.test_results:
            self.test_results[result.process_id] = []
        self.test_results[result.process_id].append(result)
    
    def get_test_results(self, process_id: str) -> List[TestResult]:
        """Get test results for a process."""
        return self.test_results.get(process_id, [])
    
    # Deployments
    def create_deployment(self, deployment: DeploymentConfig) -> bool:
        """Create a deployment configuration."""
        deployment.deployed_at = datetime.now()
        self.deployments[deployment.process_id] = deployment
        return True
    
    def get_deployment(self, process_id: str) -> Optional[DeploymentConfig]:
        """Get deployment configuration for a process."""
        return self.deployments.get(process_id)


# Global repository instance
agent_studio_repo = AgentStudioRepository()

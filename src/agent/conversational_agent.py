"""
Moveworks-style Conversational Agent using LangGraph reasoning.
Implements proper plugin architecture with conversational processes.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
from datetime import datetime

from ..reasoning.langgraph_reasoning_agent import MoveworksReasoningAgent
from ..connectors.base_connector import ConnectorManager
from ..connectors.moveworks_connectors import (
    MoveworksPurpleConnector,
    ServiceNowConnector,
    JiraConnector,
    SalesforceConnector
)


@dataclass
class Plugin:
    """Represents a Moveworks-style plugin with conversational process."""
    name: str
    description: str
    triggers: List[str]
    process_name: str
    required_connectors: List[str]
    launch_permissions: Dict[str, Any]


@dataclass
class ConversationContext:
    """Context for a conversation session."""
    session_id: str
    user_id: str
    user_attributes: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    active_plugins: List[str]
    current_process: Optional[str] = None


class MoveworksConversationalAgent:
    """
    Main conversational agent implementing Moveworks patterns.
    
    Features:
    - Plugin-based architecture
    - LangGraph reasoning engine
    - Connector management
    - Conversational processes
    - Memory and context management
    """
    
    def __init__(self):
        self.reasoning_agent = MoveworksReasoningAgent()
        self.connector_manager = ConnectorManager()
        self.plugins: Dict[str, Plugin] = {}
        self.active_sessions: Dict[str, ConversationContext] = {}
        
        # Initialize default connectors and plugins
        self._initialize_connectors()
        self._initialize_plugins()
    
    def _initialize_connectors(self):
        """Initialize default connectors."""
        # Moveworks Purple API (for demos)
        purple_connector = MoveworksPurpleConnector.create("moveworks_purple")
        self.connector_manager.register_connector(purple_connector)
        
        # Mock enterprise connectors (would be configured with real credentials)
        # ServiceNow
        servicenow_connector = ServiceNowConnector.create(
            name="servicenow_demo",
            instance_name="demo",
            client_id="demo_client",
            client_secret="demo_secret",
            username="demo_user",
            password="demo_pass"
        )
        self.connector_manager.register_connector(servicenow_connector)
        
        # Jira
        jira_connector = JiraConnector.create(
            name="jira_demo",
            base_url="https://demo.atlassian.net",
            username="demo@company.com",
            api_token="demo_token"
        )
        self.connector_manager.register_connector(jira_connector)
        
        # Salesforce
        salesforce_connector = SalesforceConnector.create(
            name="salesforce_demo",
            client_id="demo_client",
            client_secret="demo_secret",
            username="demo@company.com",
            password="demo_pass",
            security_token="demo_token",
            sandbox=True
        )
        self.connector_manager.register_connector(salesforce_connector)
    
    def _initialize_plugins(self):
        """Initialize default plugins."""
        
        # PTO Balance Plugin
        self.plugins["pto_balance"] = Plugin(
            name="PTO Balance Checker",
            description="Check paid time off balance for users",
            triggers=[
                "check pto balance",
                "how many vacation days",
                "time off balance",
                "pto remaining",
                "vacation days left"
            ],
            process_name="pto_balance",
            required_connectors=["moveworks_purple"],
            launch_permissions={"all_users": True}
        )
        
        # Password Reset Plugin
        self.plugins["password_reset"] = Plugin(
            name="Password Reset Assistant",
            description="Help users reset their passwords",
            triggers=[
                "password reset",
                "forgot password",
                "can't login",
                "reset my password",
                "password help"
            ],
            process_name="password_reset",
            required_connectors=["servicenow_demo"],
            launch_permissions={"all_users": True}
        )
        
        # IT Support Plugin
        self.plugins["it_support"] = Plugin(
            name="IT Support Ticket",
            description="Create IT support tickets",
            triggers=[
                "create ticket",
                "it support",
                "technical issue",
                "computer problem",
                "need help"
            ],
            process_name="it_support",
            required_connectors=["servicenow_demo", "jira_demo"],
            launch_permissions={"all_users": True}
        )
        
        # Account Lookup Plugin
        self.plugins["account_lookup"] = Plugin(
            name="Account Lookup",
            description="Look up customer accounts in Salesforce",
            triggers=[
                "lookup account",
                "find customer",
                "account details",
                "customer information"
            ],
            process_name="account_lookup",
            required_connectors=["salesforce_demo"],
            launch_permissions={"sales_team": True, "support_team": True}
        )
    
    async def process_message(
        self,
        content: str,
        user_id: str,
        session_id: str,
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message through the conversational agent.
        
        Args:
            content: User message content
            user_id: Unique user identifier
            session_id: Session identifier
            user_attributes: User attributes and permissions
            
        Returns:
            Dict containing response and metadata
        """
        if user_attributes is None:
            user_attributes = {
                "email": f"{user_id}@example.com",
                "department": "IT",
                "role": "employee"
            }
        
        # Get or create conversation context
        context = self._get_conversation_context(session_id, user_id, user_attributes)
        
        # Add message to conversation history
        context.conversation_history.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Route to appropriate plugin
        selected_plugin = self._route_to_plugin(content, user_attributes)
        
        # Process through reasoning agent
        reasoning_result = await self.reasoning_agent.process_message(
            content, user_id, session_id, user_attributes
        )
        
        # Execute plugin-specific actions if needed
        plugin_result = None
        if selected_plugin:
            plugin_result = await self._execute_plugin_actions(
                selected_plugin, reasoning_result, context
            )
        
        # Generate final response
        response = self._generate_response(reasoning_result, plugin_result, selected_plugin)
        
        # Add response to conversation history
        context.conversation_history.append({
            "role": "assistant",
            "content": response["content"],
            "timestamp": datetime.now().isoformat(),
            "plugin_used": selected_plugin.name if selected_plugin else None
        })
        
        return response
    
    def _get_conversation_context(
        self, 
        session_id: str, 
        user_id: str, 
        user_attributes: Dict[str, Any]
    ) -> ConversationContext:
        """Get or create conversation context for a session."""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ConversationContext(
                session_id=session_id,
                user_id=user_id,
                user_attributes=user_attributes,
                conversation_history=[],
                active_plugins=[]
            )
        
        return self.active_sessions[session_id]
    
    def _route_to_plugin(self, content: str, user_attributes: Dict[str, Any]) -> Optional[Plugin]:
        """Route user message to appropriate plugin."""
        content_lower = content.lower()
        
        # Find matching plugins based on triggers
        matching_plugins = []
        for plugin in self.plugins.values():
            for trigger in plugin.triggers:
                if trigger.lower() in content_lower:
                    # Check permissions
                    if self._check_plugin_permissions(plugin, user_attributes):
                        matching_plugins.append((plugin, len(trigger)))
        
        # Return plugin with longest matching trigger (most specific)
        if matching_plugins:
            return max(matching_plugins, key=lambda x: x[1])[0]
        
        return None
    
    def _check_plugin_permissions(self, plugin: Plugin, user_attributes: Dict[str, Any]) -> bool:
        """Check if user has permission to use plugin."""
        permissions = plugin.launch_permissions
        
        if permissions.get("all_users", False):
            return True
        
        user_role = user_attributes.get("role", "")
        user_department = user_attributes.get("department", "")
        
        # Check role-based permissions
        if permissions.get(f"{user_role}_team", False):
            return True
        
        if permissions.get(f"{user_department}_team", False):
            return True
        
        return False
    
    async def _execute_plugin_actions(
        self,
        plugin: Plugin,
        reasoning_result: Dict[str, Any],
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Execute plugin-specific actions using connectors."""
        
        # Get required connectors
        connector_results = {}
        
        for connector_name in plugin.required_connectors:
            connector = self.connector_manager.get_connector(connector_name)
            if connector:
                # Execute actions based on plugin type
                if plugin.process_name == "pto_balance":
                    result = await connector.execute_action("get_pto_balance", {
                        "user_email": context.user_attributes.get("email"),
                        "pto_type": "vacation"
                    })
                    connector_results[connector_name] = result
                
                elif plugin.process_name == "password_reset":
                    result = await connector.execute_action("create_incident", {
                        "description": "Password reset request",
                        "user_id": context.user_id,
                        "assignment_group": "IT Support"
                    })
                    connector_results[connector_name] = result
                
                elif plugin.process_name == "account_lookup":
                    # Extract account name from reasoning result
                    slots = reasoning_result.get("slots_collected", {})
                    account_name = slots.get("account_name", "Demo Account")
                    
                    result = await connector.execute_action("lookup_account", {
                        "account_name": account_name
                    })
                    connector_results[connector_name] = result
        
        return {
            "plugin": plugin.name,
            "connector_results": connector_results,
            "execution_time": datetime.now().isoformat()
        }
    
    def _generate_response(
        self,
        reasoning_result: Dict[str, Any],
        plugin_result: Optional[Dict[str, Any]],
        selected_plugin: Optional[Plugin]
    ) -> Dict[str, Any]:
        """Generate final response combining reasoning and plugin results."""
        
        base_response = reasoning_result.get("response", "I've processed your request.")
        
        # Enhance response with plugin-specific information
        if plugin_result and selected_plugin:
            connector_results = plugin_result.get("connector_results", {})
            
            if selected_plugin.process_name == "pto_balance":
                for result in connector_results.values():
                    if result.success and result.data:
                        pto_data = result.data.get("pto_details", {})
                        vacation_days = pto_data.get("vacation_days", 15)
                        sick_days = pto_data.get("sick_days", 8)
                        base_response = f"Your current PTO balance: {vacation_days} vacation days and {sick_days} sick days remaining."
            
            elif selected_plugin.process_name == "password_reset":
                for result in connector_results.values():
                    if result.success and result.data:
                        ticket_id = result.data.get("ticket_id", "Unknown")
                        base_response = f"I've created a password reset ticket for you. Ticket ID: {ticket_id}. IT support will contact you shortly."
            
            elif selected_plugin.process_name == "account_lookup":
                for result in connector_results.values():
                    if result.success and result.data:
                        account = result.data
                        base_response = f"Found account: {account.get('Name', 'Unknown')} in {account.get('Industry', 'Unknown')} industry."
        
        return {
            "content": base_response,
            "reasoning_trace": reasoning_result.get("reasoning_trace", []),
            "plugin_used": selected_plugin.name if selected_plugin else None,
            "process_name": reasoning_result.get("process_used"),
            "slots_collected": reasoning_result.get("slots_collected", {}),
            "plugin_results": plugin_result,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_available_plugins(self, user_attributes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of plugins available to the user."""
        available_plugins = []
        
        for plugin in self.plugins.values():
            if self._check_plugin_permissions(plugin, user_attributes):
                available_plugins.append({
                    "name": plugin.name,
                    "description": plugin.description,
                    "triggers": plugin.triggers[:3],  # Show first 3 triggers
                    "process_name": plugin.process_name
                })
        
        return available_plugins
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status including connector health."""
        connector_status = await self.connector_manager.test_all_connections()
        
        return {
            "active_sessions": len(self.active_sessions),
            "available_plugins": len(self.plugins),
            "connector_status": {
                name: result.success for name, result in connector_status.items()
            },
            "reasoning_agent_status": "active",
            "timestamp": datetime.now().isoformat()
        }
    
    async def close(self):
        """Close the agent and cleanup resources."""
        await self.connector_manager.close_all()

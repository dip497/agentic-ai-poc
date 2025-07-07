"""
Moveworks-specific connectors following the documented patterns.
Includes Purple API, ServiceNow, Jira, and other enterprise system connectors.
"""

from typing import Dict, Any, Optional
from .base_connector import HTTPConnector, SystemConnector, ConnectorConfig, AuthType, ActionResult


class MoveworksPurpleConnector(HTTPConnector):
    """Connector for Moveworks Purple API (demo/testing purposes)."""
    
    @classmethod
    def create(cls, name: str = "moveworks_purple") -> "MoveworksPurpleConnector":
        """Create a Moveworks Purple API connector."""
        config = ConnectorConfig(
            name=name,
            description="Moveworks Purple APIs Connector",
            base_url="https://us-central1-creator-studio-workflows.cloudfunctions.net",
            auth_type=AuthType.NO_AUTH,
            auth_config={}
        )
        
        connector = cls(config)
        
        # Register common actions
        connector.register_action("get_pto_balance", {
            "endpoint": "/getPTOBalanceByType",
            "method": "GET",
            "params": {
                "email": "{{user_email}}",
                "pto_type": "{{pto_type}}"
            }
        })
        
        connector.register_action("submit_purchase_request", {
            "endpoint": "/submitPurchaseRequest",
            "method": "POST",
            "data": {
                "item_name": "{{item_name}}",
                "quantity": "{{quantity}}",
                "justification": "{{business_justification}}"
            }
        })
        
        return connector


class ServiceNowConnector(SystemConnector):
    """ServiceNow system connector following Moveworks patterns."""
    
    @classmethod
    def create(
        cls, 
        name: str,
        instance_name: str,
        client_id: str,
        client_secret: str,
        username: str,
        password: str
    ) -> "ServiceNowConnector":
        """Create a ServiceNow connector with OAuth2 Password Grant."""
        
        base_url = f"https://{instance_name}.service-now.com"
        
        config = ConnectorConfig(
            name=name,
            description=f"ServiceNow connector for {instance_name}",
            base_url=base_url,
            auth_type=AuthType.OAUTH2_PASSWORD_GRANT,
            auth_config={
                "client_id": client_id,
                "client_secret": client_secret,
                "username": username,
                "password": password,
                "token_url": f"{base_url}/oauth_token.do",
                "system_type": "servicenow"
            }
        )
        
        return cls(config)
    
    async def execute_action(self, action_name: str, input_args: Dict[str, Any]) -> ActionResult:
        """Execute ServiceNow-specific actions."""
        if action_name == "create_incident":
            return await self._create_incident(input_args)
        elif action_name == "lookup_user_assets":
            return await self._lookup_user_assets(input_args)
        elif action_name == "get_ticket_status":
            return await self._get_ticket_status(input_args)
        else:
            return await super().execute_action(action_name, input_args)
    
    async def _create_incident(self, input_args: Dict[str, Any]) -> ActionResult:
        """Create an incident in ServiceNow."""
        # Mock implementation
        return ActionResult(
            success=True,
            data={
                "ticket_id": "INC0012345",
                "short_description": input_args.get("description", "User request"),
                "state": "New",
                "assigned_to": input_args.get("assignment_group", "IT Support"),
                "created_by": input_args.get("user_id", "system")
            }
        )
    
    async def _lookup_user_assets(self, input_args: Dict[str, Any]) -> ActionResult:
        """Look up assets assigned to a user."""
        user_id = input_args.get("user_id", "unknown")
        
        # Mock asset data
        assets = [
            {
                "asset_id": "LAPTOP001",
                "model": "Dell Latitude 7420",
                "serial_number": "DL7420-001",
                "status": "In Use",
                "assigned_to": user_id
            },
            {
                "asset_id": "MONITOR001", 
                "model": "Dell UltraSharp U2720Q",
                "serial_number": "DU2720-001",
                "status": "In Use",
                "assigned_to": user_id
            }
        ]
        
        return ActionResult(
            success=True,
            data={
                "user_id": user_id,
                "assets": assets,
                "total_count": len(assets)
            }
        )
    
    async def _get_ticket_status(self, input_args: Dict[str, Any]) -> ActionResult:
        """Get the status of a ticket."""
        ticket_id = input_args.get("ticket_id", "")
        
        return ActionResult(
            success=True,
            data={
                "ticket_id": ticket_id,
                "status": "In Progress",
                "assigned_to": "John Smith",
                "last_updated": "2025-01-07T17:30:00Z",
                "description": "Password reset request"
            }
        )


class JiraConnector(SystemConnector):
    """Jira Service Desk connector following Moveworks patterns."""
    
    @classmethod
    def create(
        cls,
        name: str,
        base_url: str,
        username: str,
        api_token: str
    ) -> "JiraConnector":
        """Create a Jira connector with Basic Auth."""
        
        config = ConnectorConfig(
            name=name,
            description=f"Jira Service Desk connector",
            base_url=base_url,
            auth_type=AuthType.BASIC_AUTH,
            auth_config={
                "username": username,
                "password": api_token,  # API token acts as password
                "system_type": "jira"
            }
        )
        
        return cls(config)
    
    async def execute_action(self, action_name: str, input_args: Dict[str, Any]) -> ActionResult:
        """Execute Jira-specific actions."""
        if action_name == "create_issue":
            return await self._create_issue(input_args)
        elif action_name == "update_issue":
            return await self._update_issue(input_args)
        elif action_name == "get_issue":
            return await self._get_issue(input_args)
        else:
            return await super().execute_action(action_name, input_args)
    
    async def _create_issue(self, input_args: Dict[str, Any]) -> ActionResult:
        """Create an issue in Jira."""
        return ActionResult(
            success=True,
            data={
                "issue_key": "HELP-123",
                "summary": input_args.get("summary", "User request"),
                "status": "Open",
                "assignee": input_args.get("assignee"),
                "reporter": input_args.get("reporter", "system"),
                "created": "2025-01-07T17:30:00Z"
            }
        )
    
    async def _update_issue(self, input_args: Dict[str, Any]) -> ActionResult:
        """Update an issue in Jira."""
        issue_key = input_args.get("issue_key", "")
        
        return ActionResult(
            success=True,
            data={
                "issue_key": issue_key,
                "updated_fields": input_args.get("fields", {}),
                "updated": "2025-01-07T17:30:00Z"
            }
        )
    
    async def _get_issue(self, input_args: Dict[str, Any]) -> ActionResult:
        """Get an issue from Jira."""
        issue_key = input_args.get("issue_key", "")
        
        return ActionResult(
            success=True,
            data={
                "issue_key": issue_key,
                "summary": "Password reset request",
                "status": "In Progress",
                "assignee": "jane.doe@company.com",
                "reporter": "john.smith@company.com",
                "created": "2025-01-07T16:00:00Z",
                "updated": "2025-01-07T17:00:00Z"
            }
        )


class SalesforceConnector(SystemConnector):
    """Salesforce connector following Moveworks patterns."""
    
    @classmethod
    def create(
        cls,
        name: str,
        client_id: str,
        client_secret: str,
        username: str,
        password: str,
        security_token: str,
        sandbox: bool = False
    ) -> "SalesforceConnector":
        """Create a Salesforce connector."""
        
        domain = "test" if sandbox else "login"
        base_url = f"https://{domain}.salesforce.com"
        
        config = ConnectorConfig(
            name=name,
            description="Salesforce connector",
            base_url=base_url,
            auth_type=AuthType.OAUTH2_PASSWORD_GRANT,
            auth_config={
                "client_id": client_id,
                "client_secret": client_secret,
                "username": username,
                "password": f"{password}{security_token}",
                "token_url": f"{base_url}/services/oauth2/token",
                "system_type": "salesforce"
            }
        )
        
        return cls(config)
    
    async def execute_action(self, action_name: str, input_args: Dict[str, Any]) -> ActionResult:
        """Execute Salesforce-specific actions."""
        if action_name == "lookup_account":
            return await self._lookup_account(input_args)
        elif action_name == "create_case":
            return await self._create_case(input_args)
        elif action_name == "update_case":
            return await self._update_case(input_args)
        else:
            return await super().execute_action(action_name, input_args)
    
    async def _lookup_account(self, input_args: Dict[str, Any]) -> ActionResult:
        """Look up a Salesforce account."""
        account_name = input_args.get("account_name", "")
        
        return ActionResult(
            success=True,
            data={
                "Id": "0011234567890ABC",
                "Name": account_name,
                "BillingAddress": {
                    "street": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "postalCode": "94105"
                },
                "Industry": "Technology",
                "Description": "Leading technology company"
            }
        )
    
    async def _create_case(self, input_args: Dict[str, Any]) -> ActionResult:
        """Create a case in Salesforce."""
        return ActionResult(
            success=True,
            data={
                "Id": "5001234567890ABC",
                "CaseNumber": "00001234",
                "Subject": input_args.get("subject", "User request"),
                "Status": "New",
                "Priority": input_args.get("priority", "Medium"),
                "Origin": "Chat"
            }
        )
    
    async def _update_case(self, input_args: Dict[str, Any]) -> ActionResult:
        """Update a case in Salesforce."""
        case_id = input_args.get("case_id", "")
        
        return ActionResult(
            success=True,
            data={
                "Id": case_id,
                "updated_fields": input_args.get("fields", {}),
                "LastModifiedDate": "2025-01-07T17:30:00Z"
            }
        )

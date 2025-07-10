"""
Moveworks Data Type System

Implements u_ prefixed data types following Moveworks patterns:
- u_JiraIssue: Jira ticket/issue objects
- u_ServiceNowTicket: ServiceNow ticket objects  
- u_SalesforceAccount: Salesforce account objects
- u_Employee: Employee/user objects
- u_AsanaTask: Asana task objects

Each data type has proper schema and resolver integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .moveworks_slot_system import MoveworksDataType, ResolverStrategy, ResolverMethodType, StaticResolverOption


# Jira Issue Data Type
JIRA_ISSUE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Jira issue ID"},
        "key": {"type": "string", "description": "Jira issue key (e.g., BUG-123)"},
        "summary": {"type": "string", "description": "Issue summary/title"},
        "description": {"type": "string", "description": "Issue description"},
        "status": {"type": "string", "description": "Issue status"},
        "priority": {"type": "string", "description": "Issue priority"},
        "assignee": {"type": "string", "description": "Assigned user"},
        "reporter": {"type": "string", "description": "Issue reporter"},
        "created": {"type": "string", "format": "date-time"},
        "updated": {"type": "string", "format": "date-time"},
        "project": {"type": "string", "description": "Project key"},
        "issue_type": {"type": "string", "description": "Type of issue"}
    },
    "required": ["id", "key", "summary"]
}

# ServiceNow Ticket Data Type  
SERVICENOW_TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "sys_id": {"type": "string", "description": "ServiceNow system ID"},
        "number": {"type": "string", "description": "Ticket number (e.g., INC0000123)"},
        "short_description": {"type": "string", "description": "Ticket summary"},
        "description": {"type": "string", "description": "Detailed description"},
        "state": {"type": "string", "description": "Ticket state"},
        "priority": {"type": "string", "description": "Ticket priority"},
        "urgency": {"type": "string", "description": "Ticket urgency"},
        "assigned_to": {"type": "string", "description": "Assigned user"},
        "caller_id": {"type": "string", "description": "Ticket requester"},
        "opened_at": {"type": "string", "format": "date-time"},
        "updated_at": {"type": "string", "format": "date-time"},
        "category": {"type": "string", "description": "Ticket category"},
        "subcategory": {"type": "string", "description": "Ticket subcategory"}
    },
    "required": ["sys_id", "number", "short_description"]
}

# Salesforce Account Data Type
SALESFORCE_ACCOUNT_SCHEMA = {
    "type": "object", 
    "properties": {
        "Id": {"type": "string", "description": "Salesforce account ID"},
        "Name": {"type": "string", "description": "Account name"},
        "Type": {"type": "string", "description": "Account type"},
        "Industry": {"type": "string", "description": "Industry"},
        "AnnualRevenue": {"type": "number", "description": "Annual revenue"},
        "NumberOfEmployees": {"type": "integer", "description": "Employee count"},
        "BillingAddress": {"type": "object", "description": "Billing address"},
        "Phone": {"type": "string", "description": "Phone number"},
        "Website": {"type": "string", "description": "Website URL"},
        "Owner": {"type": "string", "description": "Account owner"},
        "CreatedDate": {"type": "string", "format": "date-time"},
        "LastModifiedDate": {"type": "string", "format": "date-time"}
    },
    "required": ["Id", "Name"]
}

# Employee Data Type
EMPLOYEE_SCHEMA = {
    "type": "object",
    "properties": {
        "employee_id": {"type": "string", "description": "Employee ID"},
        "email": {"type": "string", "format": "email", "description": "Email address"},
        "first_name": {"type": "string", "description": "First name"},
        "last_name": {"type": "string", "description": "Last name"},
        "display_name": {"type": "string", "description": "Display name"},
        "department": {"type": "string", "description": "Department"},
        "title": {"type": "string", "description": "Job title"},
        "manager": {"type": "string", "description": "Manager email"},
        "location": {"type": "string", "description": "Office location"},
        "phone": {"type": "string", "description": "Phone number"},
        "start_date": {"type": "string", "format": "date"},
        "status": {"type": "string", "description": "Employment status"}
    },
    "required": ["employee_id", "email", "first_name", "last_name"]
}

# Asana Task Data Type
ASANA_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "gid": {"type": "string", "description": "Asana task GID"},
        "name": {"type": "string", "description": "Task name"},
        "notes": {"type": "string", "description": "Task description"},
        "completed": {"type": "boolean", "description": "Completion status"},
        "assignee": {"type": "object", "description": "Assigned user"},
        "projects": {"type": "array", "description": "Associated projects"},
        "due_on": {"type": "string", "format": "date", "description": "Due date"},
        "created_at": {"type": "string", "format": "date-time"},
        "modified_at": {"type": "string", "format": "date-time"},
        "tags": {"type": "array", "description": "Task tags"},
        "priority": {"type": "string", "description": "Task priority"}
    },
    "required": ["gid", "name"]
}


class MoveworksDataTypeRegistry:
    """Registry for Moveworks data types with u_ prefix convention."""
    
    def __init__(self):
        self.data_types: Dict[str, MoveworksDataType] = {}
        self._register_builtin_types()
    
    def _register_builtin_types(self):
        """Register built-in Moveworks data types."""
        
        # u_JiraIssue
        self.register_data_type(MoveworksDataType(
            name="u_JiraIssue",
            schema=JIRA_ISSUE_SCHEMA,
            description="Jira issue or ticket object",
            properties={
                "system": "jira",
                "api_endpoint": "/rest/api/2/issue",
                "search_fields": ["key", "summary", "description"]
            }
        ))
        
        # u_ServiceNowTicket  
        self.register_data_type(MoveworksDataType(
            name="u_ServiceNowTicket",
            schema=SERVICENOW_TICKET_SCHEMA,
            description="ServiceNow ticket object",
            properties={
                "system": "servicenow",
                "api_endpoint": "/api/now/table/incident",
                "search_fields": ["number", "short_description", "description"]
            }
        ))
        
        # u_SalesforceAccount
        self.register_data_type(MoveworksDataType(
            name="u_SalesforceAccount", 
            schema=SALESFORCE_ACCOUNT_SCHEMA,
            description="Salesforce account object",
            properties={
                "system": "salesforce",
                "api_endpoint": "/services/data/v52.0/sobjects/Account",
                "search_fields": ["Name", "Type", "Industry"]
            }
        ))
        
        # u_Employee
        self.register_data_type(MoveworksDataType(
            name="u_Employee",
            schema=EMPLOYEE_SCHEMA,
            description="Employee/user object",
            properties={
                "system": "ldap",
                "search_fields": ["email", "first_name", "last_name", "display_name"]
            }
        ))
        
        # u_AsanaTask
        self.register_data_type(MoveworksDataType(
            name="u_AsanaTask",
            schema=ASANA_TASK_SCHEMA,
            description="Asana task object",
            properties={
                "system": "asana",
                "api_endpoint": "/api/1.0/tasks",
                "search_fields": ["name", "notes"]
            }
        ))
    
    def register_data_type(self, data_type: MoveworksDataType):
        """Register a new data type."""
        self.data_types[data_type.name] = data_type
    
    def get_data_type(self, name: str) -> Optional[MoveworksDataType]:
        """Get data type by name."""
        return self.data_types.get(name)
    
    def list_data_types(self) -> List[str]:
        """List all registered data type names."""
        return list(self.data_types.keys())
    
    def is_custom_type(self, type_name: str) -> bool:
        """Check if type is a custom u_ prefixed type."""
        return type_name.startswith("u_")
    
    def validate_data_type_schema(self, type_name: str, data: Dict[str, Any]) -> bool:
        """Validate data against data type schema."""
        data_type = self.get_data_type(type_name)
        if not data_type:
            return False
        
        # Basic schema validation (simplified)
        schema = data_type.schema
        if "required" in schema:
            for field in schema["required"]:
                if field not in data:
                    return False
        
        return True
    
    def create_sample_data(self, type_name: str) -> Dict[str, Any]:
        """Create sample data for a data type."""
        samples = {
            "u_JiraIssue": {
                "id": "10001",
                "key": "BUG-123",
                "summary": "Login page not loading",
                "description": "Users cannot access the login page",
                "status": "Open",
                "priority": "High",
                "assignee": "john.doe@company.com",
                "reporter": "jane.smith@company.com",
                "project": "WEB",
                "issue_type": "Bug"
            },
            "u_ServiceNowTicket": {
                "sys_id": "abc123def456",
                "number": "INC0000123",
                "short_description": "Password reset request",
                "description": "User needs password reset for email account",
                "state": "New",
                "priority": "3 - Moderate",
                "urgency": "3 - Medium",
                "assigned_to": "it.support@company.com",
                "caller_id": "user@company.com",
                "category": "Access Management"
            },
            "u_SalesforceAccount": {
                "Id": "001XX000003DHP0",
                "Name": "Acme Corporation",
                "Type": "Customer",
                "Industry": "Technology",
                "AnnualRevenue": 5000000,
                "NumberOfEmployees": 250,
                "Phone": "+1-555-123-4567",
                "Website": "https://acme.com"
            },
            "u_Employee": {
                "employee_id": "EMP001",
                "email": "john.doe@company.com",
                "first_name": "John",
                "last_name": "Doe",
                "display_name": "John Doe",
                "department": "Engineering",
                "title": "Software Engineer",
                "manager": "jane.manager@company.com",
                "location": "San Francisco",
                "status": "Active"
            },
            "u_AsanaTask": {
                "gid": "1234567890",
                "name": "Implement user authentication",
                "notes": "Add OAuth2 authentication to the web application",
                "completed": False,
                "assignee": {"gid": "987654321", "name": "John Doe"},
                "due_on": "2024-12-31",
                "priority": "High"
            }
        }
        
        return samples.get(type_name, {})


# Global registry instance
data_type_registry = MoveworksDataTypeRegistry()

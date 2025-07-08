#!/usr/bin/env python3
"""
Mock External API Server

This server simulates external systems like Jira, ServiceNow, Salesforce, etc.
for testing our Dynamic Resolver Execution Engine.

Endpoints:
- /jira/issues/{issue_id} - Get specific Jira issue
- /jira/users/{user_id}/issues - Get user's assigned issues
- /servicenow/incidents - Get ServiceNow incidents
- /salesforce/accounts - Get Salesforce accounts
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional
import uvicorn
from datetime import datetime, timedelta
import random

app = FastAPI(title="Mock External APIs", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data
MOCK_JIRA_ISSUES = {
    "BUG-732": {
        "id": "BUG-732",
        "key": "BUG-732",
        "title": "Login authentication failure",
        "summary": "Users cannot log in with valid credentials",
        "description": "Multiple users reporting login failures despite correct credentials. Error occurs on both web and mobile platforms.",
        "status": "Open",
        "priority": "High",
        "assignee": "test_user_123",
        "reporter": "user_456",
        "created": "2025-01-06T10:30:00Z",
        "updated": "2025-01-07T14:20:00Z",
        "labels": ["authentication", "critical", "login"],
        "components": ["Authentication Service", "Web Frontend"]
    },
    "BUG-733": {
        "id": "BUG-733", 
        "key": "BUG-733",
        "title": "Password reset not working",
        "summary": "Password reset emails are not being sent",
        "description": "Users clicking 'Forgot Password' are not receiving reset emails. SMTP service appears to be functioning.",
        "status": "In Progress",
        "priority": "Medium",
        "assignee": "test_user_123",
        "reporter": "user_789",
        "created": "2025-01-05T16:45:00Z",
        "updated": "2025-01-07T09:15:00Z",
        "labels": ["email", "password", "reset"],
        "components": ["Email Service", "User Management"]
    },
    "TASK-101": {
        "id": "TASK-101",
        "key": "TASK-101", 
        "title": "Implement dark mode",
        "summary": "Add dark mode theme to the application",
        "description": "Users have requested a dark mode option for better usability in low-light environments.",
        "status": "To Do",
        "priority": "Low",
        "assignee": "dev_user_456",
        "reporter": "product_manager",
        "created": "2025-01-04T11:00:00Z",
        "updated": "2025-01-04T11:00:00Z",
        "labels": ["enhancement", "ui", "theme"],
        "components": ["Frontend", "UI/UX"]
    }
}

MOCK_SERVICENOW_INCIDENTS = [
    {
        "id": "INC0001234",
        "number": "INC0001234",
        "short_description": "Email server down",
        "description": "Corporate email server is not responding. Users cannot send or receive emails.",
        "state": "New",
        "priority": "1 - Critical",
        "assigned_to": "admin_user",
        "caller_id": "user_123",
        "opened_at": "2025-01-07T08:30:00Z",
        "category": "Infrastructure",
        "subcategory": "Email"
    },
    {
        "id": "INC0001235",
        "number": "INC0001235",
        "short_description": "VPN connection issues",
        "description": "Multiple users reporting VPN connection failures when working remotely.",
        "state": "In Progress", 
        "priority": "2 - High",
        "assigned_to": "network_admin",
        "caller_id": "user_456",
        "opened_at": "2025-01-07T10:15:00Z",
        "category": "Network",
        "subcategory": "VPN"
    }
]

MOCK_SALESFORCE_ACCOUNTS = [
    {
        "id": "001XX000003DHPi",
        "name": "Acme Corporation",
        "type": "Customer - Direct",
        "industry": "Technology",
        "annual_revenue": 5000000,
        "number_of_employees": 250,
        "phone": "+1-555-123-4567",
        "website": "https://acme-corp.com",
        "billing_city": "San Francisco",
        "billing_state": "CA"
    },
    {
        "id": "001XX000003DHPj",
        "name": "Global Industries Inc",
        "type": "Prospect",
        "industry": "Manufacturing",
        "annual_revenue": 15000000,
        "number_of_employees": 500,
        "phone": "+1-555-987-6543",
        "website": "https://globalindustries.com",
        "billing_city": "Chicago",
        "billing_state": "IL"
    }
]

# Jira API endpoints
@app.get("/jira/issues/{issue_id}")
async def get_jira_issue(issue_id: str):
    """Get a specific Jira issue by ID."""
    if issue_id in MOCK_JIRA_ISSUES:
        return {
            "success": True,
            "issue": MOCK_JIRA_ISSUES[issue_id]
        }
    else:
        raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")

@app.get("/jira/users/{user_id}/issues")
async def get_user_jira_issues(user_id: str, status: Optional[str] = None):
    """Get all issues assigned to a specific user."""
    user_issues = [
        issue for issue in MOCK_JIRA_ISSUES.values() 
        if issue["assignee"] == user_id
    ]
    
    if status:
        user_issues = [issue for issue in user_issues if issue["status"].lower() == status.lower()]
    
    return {
        "success": True,
        "issues": user_issues,
        "total": len(user_issues)
    }

@app.get("/jira/search")
async def search_jira_issues(q: str, assignee: Optional[str] = None):
    """Search Jira issues by query."""
    results = []
    
    for issue in MOCK_JIRA_ISSUES.values():
        # Simple text search in title and description
        if (q.lower() in issue["title"].lower() or 
            q.lower() in issue["description"].lower() or
            q.lower() in " ".join(issue["labels"]).lower()):
            
            if assignee is None or issue["assignee"] == assignee:
                results.append(issue)
    
    return {
        "success": True,
        "issues": results,
        "total": len(results)
    }

# ServiceNow API endpoints
@app.get("/servicenow/incidents")
async def get_servicenow_incidents(assigned_to: Optional[str] = None, state: Optional[str] = None):
    """Get ServiceNow incidents."""
    incidents = MOCK_SERVICENOW_INCIDENTS.copy()
    
    if assigned_to:
        incidents = [inc for inc in incidents if inc["assigned_to"] == assigned_to]
    
    if state:
        incidents = [inc for inc in incidents if inc["state"].lower() == state.lower()]
    
    return {
        "success": True,
        "result": incidents,
        "total": len(incidents)
    }

@app.get("/servicenow/incidents/{incident_id}")
async def get_servicenow_incident(incident_id: str):
    """Get a specific ServiceNow incident."""
    for incident in MOCK_SERVICENOW_INCIDENTS:
        if incident["id"] == incident_id or incident["number"] == incident_id:
            return {
                "success": True,
                "result": incident
            }
    
    raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

# Salesforce API endpoints
@app.get("/salesforce/accounts")
async def get_salesforce_accounts(name: Optional[str] = None, type: Optional[str] = None):
    """Get Salesforce accounts."""
    accounts = MOCK_SALESFORCE_ACCOUNTS.copy()
    
    if name:
        accounts = [acc for acc in accounts if name.lower() in acc["name"].lower()]
    
    if type:
        accounts = [acc for acc in accounts if acc["type"].lower() == type.lower()]
    
    return {
        "success": True,
        "records": accounts,
        "totalSize": len(accounts)
    }

@app.get("/salesforce/accounts/{account_id}")
async def get_salesforce_account(account_id: str):
    """Get a specific Salesforce account."""
    for account in MOCK_SALESFORCE_ACCOUNTS:
        if account["id"] == account_id:
            return {
                "success": True,
                "record": account
            }
    
    raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "jira": "available",
            "servicenow": "available", 
            "salesforce": "available"
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Mock External APIs",
        "version": "1.0.0",
        "description": "Mock server for testing Moveworks resolver strategies",
        "endpoints": {
            "jira": [
                "/jira/issues/{issue_id}",
                "/jira/users/{user_id}/issues",
                "/jira/search"
            ],
            "servicenow": [
                "/servicenow/incidents",
                "/servicenow/incidents/{incident_id}"
            ],
            "salesforce": [
                "/salesforce/accounts",
                "/salesforce/accounts/{account_id}"
            ]
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Mock External APIs Server...")
    print("📋 Available endpoints:")
    print("   - Jira: http://localhost:8001/jira/")
    print("   - ServiceNow: http://localhost:8001/servicenow/")
    print("   - Salesforce: http://localhost:8001/salesforce/")
    print("   - Health: http://localhost:8001/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)

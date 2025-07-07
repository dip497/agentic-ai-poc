"""
Agent Studio API endpoints for building conversational processes.
Provides REST API for managing plugins, connectors, and testing.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .models import (
    ConversationalProcessDefinition,
    ConnectorDefinition,
    SlotDefinition,
    ActionDefinition,
    TestResult,
    DeploymentConfig
)
from .database import agent_studio_db
from .langgraph_integration import agent_studio_integration


# Pydantic models for API requests
class ProcessCreateRequest(BaseModel):
    name: str
    description: str
    triggers: List[str]
    keywords: List[str] = []


class ProcessUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    triggers: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    activities: Optional[List[Dict[str, Any]]] = None
    slots: Optional[List[Dict[str, Any]]] = None
    required_connectors: Optional[List[str]] = None
    permissions: Optional[Dict[str, Any]] = None


class ConnectorCreateRequest(BaseModel):
    name: str
    description: str
    type: str
    base_url: str
    auth_type: str = "none"
    auth_config: Dict[str, Any] = {}
    headers: Dict[str, str] = {}


class TestProcessRequest(BaseModel):
    process_id: str
    test_input: str
    user_attributes: Dict[str, Any] = {}


class DeployProcessRequest(BaseModel):
    process_id: str
    environment: str
    enabled: bool = True
    user_groups: List[str] = []


def create_agent_studio_router() -> APIRouter:
    """Create the Agent Studio API router."""
    router = APIRouter(prefix="/api/agent-studio", tags=["Agent Studio"])
    
    # ========== CONVERSATIONAL PROCESSES ==========
    
    @router.get("/processes")
    async def list_processes():
        """List all conversational processes."""
        try:
            processes = await agent_studio_db.list_processes()
            return JSONResponse(content={
                "processes": [
                    {
                        "id": str(p["id"]),
                        "name": p["name"],
                        "description": p["description"],
                        "version": p["version"],
                        "status": p["status"],
                        "triggers": p["triggers"],
                        "keywords": p["keywords"],
                        "activities_count": len(p["activities"]),
                        "slots_count": len(p["slots"]),
                        "required_connectors": p["required_connectors"],
                        "created_at": p["created_at"].isoformat(),
                        "updated_at": p["updated_at"].isoformat(),
                        "created_by": p["created_by"]
                    } for p in processes
                ],
                "total": len(processes)
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/processes")
    async def create_process(request: ProcessCreateRequest):
        """Create a new conversational process."""
        try:
            process_data = {
                "name": request.name,
                "description": request.description,
                "triggers": request.triggers,
                "keywords": request.keywords
            }

            process_id = await agent_studio_db.create_process(process_data)

            return JSONResponse(content={
                "id": process_id,
                "message": "Process created successfully"
            }, status_code=201)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/processes/{process_id}")
    async def get_process(process_id: str):
        """Get a specific conversational process."""
        try:
            process = agent_studio_repo.get_process(process_id)
            if not process:
                raise HTTPException(status_code=404, detail="Process not found")
            
            return JSONResponse(content={
                "id": process.id,
                "name": process.name,
                "description": process.description,
                "version": process.version,
                "status": process.status,
                "triggers": process.triggers,
                "keywords": process.keywords,
                "activities": [
                    {
                        "name": a.name,
                        "description": a.description,
                        "type": a.type,
                        "connector_name": a.connector_name,
                        "endpoint": a.endpoint,
                        "method": a.method,
                        "parameters": a.parameters,
                        "content_template": a.content_template,
                        "slot_name": a.slot_name,
                        "condition": a.condition
                    } for a in process.activities
                ],
                "slots": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "type": s.type,
                        "required": s.required,
                        "default_value": s.default_value,
                        "validation_pattern": s.validation_pattern,
                        "choices": s.choices,
                        "prompt_text": s.prompt_text
                    } for s in process.slots
                ],
                "required_connectors": process.required_connectors,
                "permissions": process.permissions,
                "created_at": process.created_at.isoformat(),
                "updated_at": process.updated_at.isoformat(),
                "created_by": process.created_by
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.put("/processes/{process_id}")
    async def update_process(process_id: str, request: ProcessUpdateRequest):
        """Update a conversational process."""
        try:
            process = await agent_studio_db.get_process(process_id)
            if not process:
                raise HTTPException(status_code=404, detail="Process not found")

            # Handle slots - convert from ProcessEditor format to database format
            slots_data = []
            if request.slots is not None:
                for slot in request.slots:
                    slots_data.append({
                        "name": slot.get("name", ""),
                        "description": slot.get("description", ""),
                        "type": slot.get("type", "text"),
                        "required": slot.get("required", False),
                        "default_value": slot.get("default_value"),
                        "validation_pattern": slot.get("validation"),
                        "choices": slot.get("options", []),
                        "prompt_text": f"Please provide your {slot.get('name', 'input')}"
                    })

            # Handle activities - convert from ProcessEditor format to database format
            activities_data = []
            if request.activities is not None:
                for activity in request.activities:
                    activities_data.append({
                        "name": activity.get("name", ""),
                        "description": activity.get("description", ""),
                        "type": activity.get("type", "response"),
                        "config": activity.get("config", {}),
                        "connector_id": activity.get("connector_id")
                    })

            # Build update data
            update_data = {
                "name": request.name if request.name is not None else process["name"],
                "description": request.description if request.description is not None else process["description"],
                "triggers": request.triggers if request.triggers is not None else process["triggers"],
                "keywords": request.keywords if request.keywords is not None else process["keywords"],
                "activities": activities_data,
                "slots": slots_data,
                "required_connectors": request.required_connectors if request.required_connectors is not None else process["required_connectors"],
                "permissions": request.permissions if request.permissions is not None else process["permissions"]
            }

            success = await agent_studio_db.update_process(process_id, update_data)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to update process")

            return JSONResponse(content={
                "message": "Process updated successfully"
            })
        except HTTPException:
            raise
        except Exception as e:
            # Add detailed error logging
            import traceback
            print(f"Error updating process {process_id}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Request data: {request}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.delete("/processes/{process_id}")
    async def delete_process(process_id: str):
        """Delete a conversational process."""
        try:
            success = agent_studio_repo.delete_process(process_id)
            if not success:
                raise HTTPException(status_code=404, detail="Process not found")
            
            return JSONResponse(content={
                "message": "Process deleted successfully"
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========== CONNECTORS ==========
    
    @router.get("/connectors")
    async def list_connectors():
        """List all connectors."""
        try:
            connectors = await agent_studio_db.list_connectors()
            return JSONResponse(content={
                "connectors": [
                    {
                        "id": str(c["id"]),
                        "name": c["name"],
                        "description": c["description"],
                        "type": c["type"],
                        "base_url": c["base_url"],
                        "auth_type": c["auth_type"],
                        "status": c["status"],
                        "actions_count": len(c["available_actions"]),
                        "created_at": c["created_at"].isoformat(),
                        "updated_at": c["updated_at"].isoformat()
                    } for c in connectors
                ],
                "total": len(connectors)
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/connectors")
    async def create_connector(request: ConnectorCreateRequest):
        """Create a new connector."""
        try:
            connector_data = {
                "name": request.name,
                "description": request.description,
                "type": request.type,
                "base_url": request.base_url,
                "auth_type": request.auth_type,
                "auth_config": request.auth_config,
                "headers": request.headers
            }

            connector_id = await agent_studio_db.create_connector(connector_data)

            return JSONResponse(content={
                "id": connector_id,
                "message": "Connector created successfully"
            }, status_code=201)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/connectors/{connector_id}")
    async def get_connector(connector_id: str):
        """Get a specific connector."""
        try:
            connector = await agent_studio_db.get_connector(connector_id)
            if not connector:
                raise HTTPException(status_code=404, detail="Connector not found")

            return JSONResponse(content={
                "id": str(connector["id"]),
                "name": connector["name"],
                "description": connector["description"],
                "type": connector["type"],
                "base_url": connector["base_url"],
                "auth_type": connector["auth_type"],
                "auth_config": connector["auth_config"],
                "headers": connector["headers"],
                "available_actions": connector["available_actions"],
                "status": connector["status"],
                "created_at": connector["created_at"].isoformat(),
                "updated_at": connector["updated_at"].isoformat()
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/connectors/{connector_id}")
    async def delete_connector(connector_id: str):
        """Delete a connector."""
        try:
            success = await agent_studio_db.delete_connector(connector_id)
            if not success:
                raise HTTPException(status_code=404, detail="Connector not found")

            return JSONResponse(content={
                "message": "Connector deleted successfully"
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========== TESTING ==========
    
    @router.post("/test")
    async def test_process(request: TestProcessRequest):
        """Test a conversational process with real LangGraph AI execution."""
        try:
            process = await agent_studio_db.get_process(request.process_id)
            if not process:
                raise HTTPException(status_code=404, detail="Process not found")

            start_time = time.time()

            # Use real LangGraph execution through Agent Studio integration
            user_attributes = getattr(request, 'user_attributes', {}) or {}
            test_result = await agent_studio_integration.test_process(
                process_id=request.process_id,
                test_input=request.test_input,
                user_id=user_attributes.get("user_id", "test_user"),
                session_id=f"test_{request.process_id}_{int(start_time)}"
            )

            execution_time = test_result.get("execution_time", 0)
            success = test_result.get("success", False)
            response = test_result.get("response", "")
            reasoning_trace = test_result.get("reasoning_trace", [])

            # Convert reasoning trace to steps format for compatibility
            steps_executed = []
            for trace_step in reasoning_trace:
                step = {
                    "activity": trace_step.get("step", "unknown"),
                    "type": trace_step.get("action", "unknown"),
                    "status": trace_step.get("status", "completed"),
                    "timestamp": trace_step.get("timestamp", datetime.now().isoformat()),
                    "result": trace_step.get("details", {})
                }
                steps_executed.append(step)

            # Create test result in database
            test_result_data = {
                "process_id": request.process_id,
                "test_input": request.test_input,
                "success": success,
                "response": response,
                "execution_time": execution_time,
                "steps_executed": steps_executed,
                "errors": [test_result.get("error")] if test_result.get("error") else []
            }

            await agent_studio_db.add_test_result(test_result_data)

            return JSONResponse(content={
                "success": success,
                "response": response,
                "execution_time": execution_time,
                "steps_executed": steps_executed,
                "reasoning_trace": reasoning_trace,
                "process_used": test_result.get("process_used"),
                "slots_collected": test_result.get("slots_collected", {}),
                "activities_executed": test_result.get("activities_executed", []),
                "agent_studio_integration": True,
                "timestamp": datetime.now().isoformat()
            })
        except HTTPException:
            raise
        except Exception as e:
            # Create failed test result
            test_result_data = {
                "process_id": request.process_id,
                "test_input": request.test_input,
                "success": False,
                "response": "",
                "execution_time": 0,
                "steps_executed": [],
                "errors": [str(e)]
            }
            await agent_studio_db.add_test_result(test_result_data)

            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/processes/{process_id}/test-results")
    async def get_test_results(process_id: str):
        """Get test results for a process."""
        try:
            results = await agent_studio_db.get_test_results(process_id)
            return JSONResponse(content={
                "test_results": [
                    {
                        "test_input": r["test_input"],
                        "success": r["success"],
                        "response": r["response"],
                        "execution_time": r["execution_time"],
                        "steps_executed": r["steps_executed"],
                        "errors": r["errors"],
                        "timestamp": r["timestamp"].isoformat()
                    } for r in results
                ],
                "total": len(results)
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========== DEPLOYMENT ==========
    
    @router.post("/deploy")
    async def deploy_process(request: DeployProcessRequest):
        """Deploy a conversational process."""
        try:
            process = await agent_studio_db.get_process(request.process_id)
            if not process:
                raise HTTPException(status_code=404, detail="Process not found")

            deployment_data = {
                "process_id": request.process_id,
                "environment": request.environment,
                "enabled": request.enabled,
                "user_groups": request.user_groups,
                "deployed_by": "system"
            }

            success = await agent_studio_db.create_deployment(deployment_data)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to deploy process")

            # Update process status
            process_update = {
                "name": process["name"],
                "description": process["description"],
                "triggers": process["triggers"],
                "keywords": process["keywords"],
                "activities": process["activities"],
                "slots": process["slots"],
                "required_connectors": process["required_connectors"],
                "permissions": process["permissions"]
            }
            await agent_studio_db.update_process(request.process_id, process_update)

            return JSONResponse(content={
                "message": f"Process deployed to {request.environment} successfully",
                "deployment_id": request.process_id,
                "deployed_at": datetime.now().isoformat()
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/processes/{process_id}/deployment")
    async def get_deployment_status(process_id: str):
        """Get deployment status for a process."""
        try:
            deployment = await agent_studio_db.get_deployment(process_id)
            if not deployment:
                return JSONResponse(content={
                    "deployed": False,
                    "message": "Process not deployed"
                })

            return JSONResponse(content={
                "deployed": True,
                "environment": deployment["environment"],
                "enabled": deployment["enabled"],
                "user_groups": deployment["user_groups"],
                "deployed_at": deployment["deployed_at"].isoformat() if deployment["deployed_at"] else None,
                "deployed_by": deployment["deployed_by"]
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return router

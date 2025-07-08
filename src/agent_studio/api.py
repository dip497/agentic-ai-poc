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
from models.moveworks import (
    DataType,
    ResolverStrategyCreateRequest,
    ResolverStrategyUpdateRequest,
    ResolverStrategyResponse,
    ResolverStrategyListResponse,
    MethodSelectionRequest,
    MethodSelectionResponse,
    ResolverMethodCreateRequest,
    ResolverMethodUpdateRequest
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


class CustomDataTypeCreateRequest(BaseModel):
    name: str
    description: str
    schema: Dict[str, Any]
    default_resolver_strategy: Optional[str] = None


class CustomDataTypeUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None
    default_resolver_strategy: Optional[str] = None
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
            process = await agent_studio_db.get_process(process_id)
            if not process:
                raise HTTPException(status_code=404, detail="Process not found")

            import json

            # Parse JSON fields that might be stored as strings
            def safe_json_parse(value, default=None):
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        return default if default is not None else []
                return value if value is not None else (default if default is not None else [])

            return JSONResponse(content={
                "id": str(process["id"]),
                "name": process["name"],
                "description": process["description"],
                "version": process["version"],
                "status": process["status"],
                "triggers": safe_json_parse(process["triggers"], []),
                "keywords": safe_json_parse(process["keywords"], []),
                "activities": safe_json_parse(process.get("activities"), []),
                "slots": safe_json_parse(process.get("slots"), []),
                "required_connectors": safe_json_parse(process.get("required_connectors"), []),
                "permissions": safe_json_parse(process.get("permissions"), {"user_groups": [], "roles": []}),
                "created_at": process["created_at"].isoformat(),
                "updated_at": process["updated_at"].isoformat(),
                "created_by": process["created_by"]
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
            success = await agent_studio_db.delete_process(process_id)
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

    # ========== CUSTOM DATA TYPES ==========

    @router.get("/data-types")
    async def list_custom_data_types():
        """List all custom data types."""
        try:
            data_types = await agent_studio_db.list_custom_data_types()
            return JSONResponse(content={"custom_data_types": data_types})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/data-types")
    async def create_custom_data_type(request: CustomDataTypeCreateRequest):
        """Create a new custom data type."""
        try:
            # Validate name convention
            if not request.name.startswith("u_") or len(request.name) <= 2:
                raise HTTPException(
                    status_code=400,
                    detail="Custom data type name must follow u_<DataTypeName> convention"
                )

            data_type_id = await agent_studio_db.create_custom_data_type(
                name=request.name,
                description=request.description,
                schema=request.schema,
                default_resolver_strategy=request.default_resolver_strategy
            )

            return JSONResponse(content={
                "message": "Custom data type created successfully",
                "data_type_id": data_type_id,
                "name": request.name
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/data-types/{data_type_name}")
    async def get_custom_data_type(data_type_name: str):
        """Get a specific custom data type."""
        try:
            data_type = await agent_studio_db.get_custom_data_type(data_type_name)
            if not data_type:
                raise HTTPException(status_code=404, detail="Custom data type not found")
            return JSONResponse(content={"data_type": data_type})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/data-types/{data_type_name}")
    async def update_custom_data_type(data_type_name: str, request: CustomDataTypeUpdateRequest):
        """Update a custom data type."""
        try:
            # Validate name convention if name is being updated
            if request.name and (not request.name.startswith("u_") or len(request.name) <= 2):
                raise HTTPException(
                    status_code=400,
                    detail="Custom data type name must follow u_<DataTypeName> convention"
                )

            success = await agent_studio_db.update_custom_data_type(
                current_name=data_type_name,
                name=request.name,
                description=request.description,
                schema=request.schema,
                default_resolver_strategy=request.default_resolver_strategy
            )

            if not success:
                raise HTTPException(status_code=404, detail="Custom data type not found")

            return JSONResponse(content={
                "message": "Custom data type updated successfully",
                "name": request.name or data_type_name
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/data-types/{data_type_name}")
    async def delete_custom_data_type(data_type_name: str):
        """Delete a custom data type."""
        try:
            success = await agent_studio_db.delete_custom_data_type(data_type_name)
            if not success:
                raise HTTPException(status_code=404, detail="Custom data type not found")

            return JSONResponse(content={
                "message": "Custom data type deleted successfully",
                "name": data_type_name
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/data-types/import-schema")
    async def import_schema_from_json(json_example: Dict[str, Any]):
        """Generate a data type schema from a JSON example."""
        try:
            # Simple schema generation from JSON example
            def generate_schema(obj):
                if isinstance(obj, dict):
                    properties = {}
                    for key, value in obj.items():
                        properties[key] = generate_schema(value)
                    return {
                        "type": "object",
                        "properties": properties
                    }
                elif isinstance(obj, list):
                    if obj:
                        return {
                            "type": "array",
                            "items": generate_schema(obj[0])
                        }
                    else:
                        return {"type": "array", "items": {}}
                elif isinstance(obj, str):
                    return {"type": "string"}
                elif isinstance(obj, int):
                    return {"type": "integer"}
                elif isinstance(obj, float):
                    return {"type": "number"}
                elif isinstance(obj, bool):
                    return {"type": "boolean"}
                else:
                    return {"type": "string"}

            schema = generate_schema(json_example)
            return JSONResponse(content={"schema": schema})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ========== RESOLVER STRATEGIES ==========

    @router.get("/resolver-strategies")
    async def list_resolver_strategies():
        """List all resolver strategies."""
        try:
            strategies = await agent_studio_db.list_resolver_strategies()
            return JSONResponse(content={"strategies": strategies, "total": len(strategies)})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/resolver-strategies")
    async def create_resolver_strategy(request: ResolverStrategyCreateRequest):
        """Create a new resolver strategy."""
        try:
            # Validate method type constraints
            static_methods = [m for m in request.methods if m.method_type == "Static"]
            dynamic_methods = [m for m in request.methods if m.method_type == "Dynamic"]

            # Rule: 1 Static Method OR Multiple Dynamic Methods (never both)
            if len(static_methods) > 0 and len(dynamic_methods) > 0:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot mix Static and Dynamic methods in one strategy"
                )
            if len(static_methods) > 1:
                raise HTTPException(
                    status_code=400,
                    detail="Only one Static method allowed per strategy"
                )
            if len(static_methods) == 0 and len(dynamic_methods) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Strategy must have at least one method"
                )

            strategy_id = await agent_studio_db.create_resolver_strategy(
                name=request.name,
                data_type=request.data_type,
                description=request.description,
                methods=[method.dict() for method in request.methods]
            )

            return JSONResponse(content={
                "message": "Resolver strategy created successfully",
                "strategy_id": strategy_id,
                "name": request.name
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/resolver-strategies/{strategy_name}")
    async def get_resolver_strategy(strategy_name: str):
        """Get a specific resolver strategy."""
        try:
            strategy = await agent_studio_db.get_resolver_strategy(strategy_name)
            if not strategy:
                raise HTTPException(status_code=404, detail="Resolver strategy not found")
            return JSONResponse(content={"strategy": strategy})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/resolver-strategies/{strategy_name}")
    async def update_resolver_strategy(strategy_name: str, request: ResolverStrategyUpdateRequest):
        """Update a resolver strategy."""
        try:
            # Validate method type constraints if methods are being updated
            if request.methods:
                static_methods = [m for m in request.methods if m.method_type == "Static"]
                dynamic_methods = [m for m in request.methods if m.method_type == "Dynamic"]

                if len(static_methods) > 0 and len(dynamic_methods) > 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Cannot mix Static and Dynamic methods in one strategy"
                    )
                if len(static_methods) > 1:
                    raise HTTPException(
                        status_code=400,
                        detail="Only one Static method allowed per strategy"
                    )

            success = await agent_studio_db.update_resolver_strategy(
                current_name=strategy_name,
                name=request.name,
                data_type=request.data_type,
                description=request.description,
                methods=[method.dict() for method in request.methods] if request.methods else None
            )

            if not success:
                raise HTTPException(status_code=404, detail="Resolver strategy not found")

            return JSONResponse(content={
                "message": "Resolver strategy updated successfully",
                "name": request.name or strategy_name
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/resolver-strategies/{strategy_name}")
    async def delete_resolver_strategy(strategy_name: str):
        """Delete a resolver strategy."""
        try:
            success = await agent_studio_db.delete_resolver_strategy(strategy_name)
            if not success:
                raise HTTPException(status_code=404, detail="Resolver strategy not found")

            return JSONResponse(content={
                "message": "Resolver strategy deleted successfully",
                "name": strategy_name
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/resolver-strategies/{strategy_name}/select-method")
    async def select_method_for_input(strategy_name: str, request: MethodSelectionRequest):
        """AI-powered method selection for user input."""
        try:
            # This would integrate with the AI reasoning system
            # For now, return a simple mock response
            return JSONResponse(content={
                "selected_method": "get_user_assigned_issues",
                "confidence": 0.85,
                "reasoning": "User mentioned 'my tasks' which indicates they want their assigned issues",
                "alternative_methods": ["get_issues_by_criteria"]
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ========== SLOT CONFIGURATION (NEW ARCHITECTURE) ==========

    @router.get("/processes/{process_id}/slots")
    async def get_process_slots(process_id: str):
        """Get slots for a process using new Moveworks architecture."""
        try:
            process = await agent_studio_db.get_process(process_id)
            if not process:
                raise HTTPException(status_code=404, detail="Process not found")

            # Convert legacy slots to new format
            new_slots = []
            for slot in process.get("slots", []):
                new_slot = {
                    "name": slot.get("name", ""),
                    "data_type": slot.get("type", "string"),
                    "slot_description": slot.get("description", ""),
                    "slot_inference_policy": "INFER_IF_AVAILABLE",
                    "resolver_strategy_name": None,  # Will be set when user configures
                    "legacy_resolver_method": None,
                    "legacy_resolver_type": None,
                    "legacy_static_options": None
                }
                new_slots.append(new_slot)

            return JSONResponse(content={"slots": new_slots})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/processes/{process_id}/slots")
    async def add_process_slot(process_id: str, slot_data: Dict[str, Any]):
        """Add a slot to a process using new Moveworks architecture."""
        try:
            process = await agent_studio_db.get_process(process_id)
            if not process:
                raise HTTPException(status_code=404, detail="Process not found")

            # Validate data type
            data_type = slot_data.get("data_type", "string")
            if data_type not in [dt.value for dt in DataType]:
                # Check if it's a custom data type
                custom_types = await agent_studio_db.list_custom_data_types()
                custom_type_names = [ct["name"] for ct in custom_types]
                if data_type not in custom_type_names:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid data type: {data_type}"
                    )

            # Create new slot in Moveworks format
            new_slot = {
                "name": slot_data.get("name", ""),
                "data_type": data_type,
                "slot_description": slot_data.get("slot_description", ""),
                "slot_inference_policy": slot_data.get("slot_inference_policy", "INFER_IF_AVAILABLE"),
                "resolver_strategy_name": slot_data.get("resolver_strategy_name"),
                "custom_data_type_name": slot_data.get("custom_data_type_name"),
                "slot_validation_policy": slot_data.get("slot_validation_policy"),
                "slot_validation_description": slot_data.get("slot_validation_description")
            }

            # Add to process slots
            current_slots = process.get("slots", [])
            current_slots.append(new_slot)

            # Update process
            success = await agent_studio_db.update_process(process_id, {"slots": current_slots})
            if not success:
                raise HTTPException(status_code=500, detail="Failed to update process")

            return JSONResponse(content={
                "message": "Slot added successfully",
                "slot": new_slot
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/processes/{process_id}/slots/{slot_name}")
    async def update_process_slot(process_id: str, slot_name: str, slot_data: Dict[str, Any]):
        """Update a slot in a process."""
        try:
            process = await agent_studio_db.get_process(process_id)
            if not process:
                raise HTTPException(status_code=404, detail="Process not found")

            # Find and update slot
            slots = process.get("slots", [])
            slot_found = False
            for i, slot in enumerate(slots):
                if slot.get("name") == slot_name:
                    # Update slot with new data
                    slots[i].update(slot_data)
                    slot_found = True
                    break

            if not slot_found:
                raise HTTPException(status_code=404, detail="Slot not found")

            # Update process
            success = await agent_studio_db.update_process(process_id, {"slots": slots})
            if not success:
                raise HTTPException(status_code=500, detail="Failed to update process")

            return JSONResponse(content={
                "message": "Slot updated successfully",
                "slot": slots[i]
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/processes/{process_id}/slots/{slot_name}")
    async def delete_process_slot(process_id: str, slot_name: str):
        """Delete a slot from a process."""
        try:
            process = await agent_studio_db.get_process(process_id)
            if not process:
                raise HTTPException(status_code=404, detail="Process not found")

            # Remove slot
            slots = process.get("slots", [])
            original_count = len(slots)
            slots = [slot for slot in slots if slot.get("name") != slot_name]

            if len(slots) == original_count:
                raise HTTPException(status_code=404, detail="Slot not found")

            # Update process
            success = await agent_studio_db.update_process(process_id, {"slots": slots})
            if not success:
                raise HTTPException(status_code=500, detail="Failed to update process")

            return JSONResponse(content={
                "message": "Slot deleted successfully",
                "slot_name": slot_name
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router

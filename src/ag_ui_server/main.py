"""
AG-UI Server for Moveworks-style conversational AI system.

This module provides the AG-UI protocol server that wraps the LangGraph reasoning agent
and emits real-time events for the frontend.
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from reasoning.moveworks_reasoning_engine import MoveworksThreeLoopEngine, create_moveworks_reasoning_engine
from agent_studio.api import create_agent_studio_router
from agent_studio.database import agent_studio_db
from agent_studio.langgraph_integration import agent_studio_integration


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Moveworks AI Platform...")
    try:
        await server.initialize()
        # Initialize Agent Studio LangGraph integration
        try:
            await agent_studio_integration.initialize()
            logger.info("✅ Agent Studio LangGraph integration initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Agent Studio LangGraph integration: {e}")

        logger.info("✅ Server initialization complete")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        raise

    yield

    # Shutdown
    logger.info("🔄 Shutting down server...")
    if hasattr(server, 'shutdown'):
        await server.shutdown()
    logger.info("✅ Server shutdown complete")


# Pydantic models for API
class ChatMessage(BaseModel):
    content: str
    user_id: str
    session_id: Optional[str] = None
    user_attributes: Optional[Dict[str, Any]] = None


class AGUIEvent(BaseModel):
    """AG-UI protocol event."""
    type: str
    data: Dict[str, Any]
    timestamp: str = None
    event_id: str = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.event_id:
            self.event_id = str(uuid.uuid4())


class ConnectionManager:
    """Manages WebSocket connections for real-time communication."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, str] = {}  # session_id -> connection_id
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connected: {connection_id}")
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        self.active_connections.pop(connection_id, None)
        # Remove session mapping
        session_to_remove = None
        for session_id, conn_id in self.session_connections.items():
            if conn_id == connection_id:
                session_to_remove = session_id
                break
        if session_to_remove:
            self.session_connections.pop(session_to_remove, None)
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    def associate_session(self, session_id: str, connection_id: str):
        """Associate a session with a connection."""
        self.session_connections[session_id] = connection_id
    
    async def send_event(self, connection_id: str, event: AGUIEvent):
        """Send an event to a specific connection."""
        websocket = self.active_connections.get(connection_id)
        if websocket:
            try:
                await websocket.send_text(event.model_dump_json())
            except Exception as e:
                logger.error(f"Error sending event to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def send_to_session(self, session_id: str, event: AGUIEvent):
        """Send an event to a session."""
        connection_id = self.session_connections.get(session_id)
        if connection_id:
            await self.send_event(connection_id, event)


class AGUIServer:
    """AG-UI protocol server for Moveworks reasoning agent."""
    
    def __init__(self):
        self.app = FastAPI(title="Moveworks AG-UI Server", version="1.0.0")
        self.connection_manager = ConnectionManager()
        
        # Initialize components
        self.reasoning_engine: Optional[MoveworksThreeLoopEngine] = None
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_agent_studio()

    async def _ensure_initialized(self):
        """Ensure the reasoning engine is initialized."""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return

            try:
                logger.info("🔄 Initializing reasoning engine...")
                print("🔄 Initializing reasoning engine...")

                # Initialize Agent Studio database
                await agent_studio_db.initialize()

                # Initialize the reasoning engine
                self.reasoning_engine = await create_moveworks_reasoning_engine()

                self._initialized = True
                logger.info("✅ Reasoning engine initialized successfully")
                print("✅ Reasoning engine initialized successfully")

            except Exception as e:
                logger.error(f"❌ Failed to initialize reasoning engine: {e}")
                print(f"❌ Failed to initialize reasoning engine: {e}")
                raise
    
    def _setup_middleware(self):
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_agent_studio(self):
        """Setup Agent Studio API routes."""
        agent_studio_router = create_agent_studio_router()
        self.app.include_router(agent_studio_router)

    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.websocket("/ws/{connection_id}")
        async def websocket_endpoint(websocket: WebSocket, connection_id: str):
            await self.connection_manager.connect(websocket, connection_id)
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    if message_data.get("type") == "chat_message":
                        await self._handle_chat_message(connection_id, message_data)
                    elif message_data.get("type") == "session_init":
                        session_id = message_data.get("session_id")
                        if session_id:
                            self.connection_manager.associate_session(session_id, connection_id)
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(connection_id)
        
        @self.app.post("/api/chat")
        async def chat_endpoint(message: ChatMessage):
            """HTTP endpoint for chat messages."""
            try:
                response = await self._process_chat_message(
                    message.content,
                    message.user_id,
                    message.session_id or str(uuid.uuid4()),
                    message.user_attributes
                )
                return JSONResponse(content=response)
            except Exception as e:
                logger.error(f"Error in chat endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/plugins")
        async def list_plugins():
            """List all available plugins."""
            try:
                if hasattr(self, 'conversational_agent') and self.conversational_agent:
                    # Get user attributes from request (in real app, from auth)
                    user_attributes = {"role": "employee", "department": "IT"}
                    plugins = await self.conversational_agent.get_available_plugins(user_attributes)
                    return JSONResponse(content={"plugins": plugins})
                else:
                    # Fallback mock plugins
                    mock_plugins = [
                        {
                            "name": "Password Reset Assistant",
                            "description": "Help users reset their passwords",
                            "triggers": ["password reset", "forgot password", "can't login"],
                            "process_name": "password_reset"
                        },
                        {
                            "name": "PTO Balance Checker",
                            "description": "Check paid time off balance for users",
                            "triggers": ["check pto balance", "how many vacation days", "time off balance"],
                            "process_name": "pto_balance"
                        }
                    ]
                    return JSONResponse(content={"plugins": mock_plugins})
            except Exception as e:
                logger.error(f"Error listing plugins: {e}")
                raise HTTPException(status_code=500, detail="Failed to list plugins")

        @self.app.get("/api/system/status")
        async def get_system_status():
            """Get system status and analytics."""
            try:
                if hasattr(self, 'conversational_agent') and self.conversational_agent:
                    system_status = await self.conversational_agent.get_system_status()

                    # Add additional UI-friendly data
                    system_status.update({
                        "last_updated": datetime.now().strftime("%I:%M:%S %p"),
                        "server_status": "active",
                        "websocket_connections": len(self.connection_manager.active_connections)
                    })

                    return JSONResponse(content=system_status)
                else:
                    # Fallback status
                    return JSONResponse(content={
                        "active_sessions": len(self.connection_manager.active_connections),
                        "available_plugins": 0,
                        "connector_status": {},
                        "reasoning_agent_status": "initializing",
                        "server_status": "active",
                        "websocket_connections": len(self.connection_manager.active_connections),
                        "last_updated": datetime.now().strftime("%I:%M:%S %p")
                    })
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get system status")
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint."""
            return JSONResponse(content={
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "conversational_agent": hasattr(self, 'conversational_agent') and self.conversational_agent is not None,
                    "websocket_server": True,
                    "reasoning_agent": hasattr(self, 'conversational_agent') and self.conversational_agent is not None
                }
            })

        @self.app.post("/api/chat/resume")
        async def chat_resume_endpoint(request: dict):
            """Resume interrupted conversation with user input."""
            try:
                session_id = request.get("session_id")
                user_response = request.get("user_response")

                if not session_id or user_response is None:
                    return {
                        "success": False,
                        "error": "session_id and user_response are required"
                    }

                response = await self._resume_chat_conversation(session_id, user_response)
                return response

            except Exception as e:
                logger.error(f"Chat resume endpoint error: {e}")
                return {
                    "response": f"Error resuming conversation: {str(e)}",
                    "success": False,
                    "error": str(e)
                }

        @self.app.get("/api/connectors/test")
        async def test_connectors():
            """Test all connectors and return their status."""
            try:
                if hasattr(self, 'conversational_agent') and self.conversational_agent:
                    # Test connector connections
                    connector_results = await self.conversational_agent.connector_manager.test_all_connections()

                    return JSONResponse(content={
                        "connector_tests": {
                            name: {
                                "success": result.success,
                                "status_code": result.status_code,
                                "error": result.error,
                                "data": result.data
                            } for name, result in connector_results.items()
                        },
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    return JSONResponse(content={
                        "error": "Conversational agent not initialized",
                        "connector_tests": {},
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error testing connectors: {e}")
                return JSONResponse(content={
                    "error": str(e),
                    "connector_tests": {},
                    "timestamp": datetime.now().isoformat()
                })
    
    async def _handle_chat_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle incoming chat message from WebSocket."""
        try:
            content = message_data.get("content", "")
            user_id = message_data.get("user_id", "anonymous")
            session_id = message_data.get("session_id", str(uuid.uuid4()))
            user_attributes = message_data.get("user_attributes", {})
            
            # Associate session with connection
            self.connection_manager.associate_session(session_id, connection_id)
            
            # Send run started event
            await self.connection_manager.send_event(
                connection_id,
                AGUIEvent(
                    type="RUN_STARTED",
                    data={
                        "session_id": session_id,
                        "user_message": content,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            )
            
            # Process message and stream response
            async for event in self._process_message_stream(content, user_id, session_id, user_attributes):
                await self.connection_manager.send_event(connection_id, event)
            
        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
            await self.connection_manager.send_event(
                connection_id,
                AGUIEvent(
                    type="ERROR",
                    data={"error": str(e), "timestamp": datetime.now().isoformat()}
                )
            )
    
    async def _process_chat_message(
        self,
        content: str,
        user_id: str,
        session_id: str,
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a chat message using Moveworks reasoning engine."""
        try:
            # Ensure reasoning engine is initialized
            await self._ensure_initialized()

            if not self.reasoning_engine:
                return {
                    "response": "Reasoning engine not initialized. Please try again later.",
                    "success": False,
                    "error": "No reasoning engine available"
                }

            # Prepare user context
            user_context = {
                "user_id": user_id,
                "session_id": session_id,
                "department": user_attributes.get("department", "unknown") if user_attributes else "unknown",
                "role": user_attributes.get("role", "user") if user_attributes else "user",
                "email": user_attributes.get("email", f"{user_id}@company.com") if user_attributes else f"{user_id}@company.com"
            }

            # Process through Moveworks three-loop reasoning with interrupt handling
            config = {"configurable": {"thread_id": session_id}}

            try:
                reasoning_result = await self.reasoning_engine.process_request(
                    user_query=content,
                    user_context=user_context,
                    conversation_id=session_id
                )
            except Exception as e:
                # Check if this is a LangGraph interrupt
                if hasattr(e, '__interrupt__') or "interrupt" in str(e).lower():
                    # This is an interrupt, not an error - handle it properly
                    return await self._handle_langgraph_interrupt(session_id, content, user_context)
                else:
                    raise e

            # Format response
            return {
                "response": reasoning_result.response,
                "success": reasoning_result.success,
                "selected_plugins": reasoning_result.selected_plugins,
                "user_actions": reasoning_result.user_actions,
                "execution_summary": reasoning_result.execution_summary,
                "conversation_id": reasoning_result.conversation_id,
                "timestamp": reasoning_result.timestamp.isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "success": False,
                "error": str(e)
            }

    async def _handle_langgraph_interrupt(
        self,
        session_id: str,
        content: str,
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle LangGraph interrupt by getting state and returning interrupt info."""
        try:
            if not self.reasoning_engine or not self.reasoning_engine.graph:
                return {
                    "response": "Reasoning engine not available for interrupt handling.",
                    "success": False,
                    "error": "No reasoning engine"
                }

            # Get the current state from LangGraph
            config = {"configurable": {"thread_id": session_id}}
            state = self.reasoning_engine.graph.get_state(config)

            if state and hasattr(state, 'interrupts') and state.interrupts:
                interrupt_data = state.interrupts[0].value

                return {
                    "response": "I need your input to continue.",
                    "success": True,
                    "interrupt": True,
                    "interrupt_data": interrupt_data,
                    "session_id": session_id,
                    "awaiting_user_input": True
                }
            else:
                return {
                    "response": "Process paused, but no interrupt data available.",
                    "success": True,
                    "interrupt": True,
                    "session_id": session_id,
                    "awaiting_user_input": True
                }

        except Exception as e:
            logger.error(f"Error handling LangGraph interrupt: {e}")
            return {
                "response": f"Error handling interrupt: {str(e)}",
                "success": False,
                "error": str(e)
            }

    async def _resume_chat_conversation(
        self,
        session_id: str,
        user_response: Any
    ) -> Dict[str, Any]:
        """Resume interrupted LangGraph conversation with user input."""
        try:
            if not self.reasoning_engine or not self.reasoning_engine.graph:
                return {
                    "response": "Reasoning engine not available for resuming.",
                    "success": False,
                    "error": "No reasoning engine"
                }

            # Resume the LangGraph execution with user input
            from langgraph.types import Command
            config = {"configurable": {"thread_id": session_id}}

            reasoning_result = await self.reasoning_engine.graph.ainvoke(
                Command(resume=user_response),
                config=config
            )

            # Convert to standard response format
            return {
                "response": reasoning_result.get("final_response", "Process completed."),
                "success": True,
                "selected_plugins": reasoning_result.get("selected_plugins", []),
                "user_actions": reasoning_result.get("user_actions", []),
                "execution_summary": reasoning_result.get("execution_summary", {}),
                "conversation_id": session_id,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error resuming conversation: {e}")
            return {
                "response": f"Error resuming conversation: {str(e)}",
                "success": False,
                "error": str(e)
            }

    async def _process_message_stream(
        self,
        content: str,
        user_id: str,
        session_id: str,
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[AGUIEvent, None]:
        """Process message and yield streaming events."""
        try:
            # Send thinking event
            yield AGUIEvent(
                type="THINKING",
                data={
                    "message": "Processing your request...",
                    "session_id": session_id
                }
            )

            # Route to agent (simplified for now)
            selected_agent_id = "general_assistant"
            routing_info = {
                "selected_agent": {
                    "name": "General Assistant",
                    "description": "Handles general inquiries"
                },
                "reason": "Default routing"
            }

            # Send agent selection event
            yield AGUIEvent(
                type="AGENT_SELECTED",
                data={
                    "agent_id": selected_agent_id,
                    "confidence": 0.85,
                    "routing_info": routing_info,
                    "session_id": session_id
                }
            )

            # Ensure reasoning engine is initialized
            await self._ensure_initialized()

            # Send reasoning started event
            yield AGUIEvent(
                type="REASONING_STARTED",
                data={
                    "agent_id": selected_agent_id,
                    "session_id": session_id,
                    "message": "Starting reasoning process..."
                }
            )

            # Process message with reasoning engine
            if self.reasoning_engine:
                # Prepare user context
                user_context = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "department": user_attributes.get("department", "unknown") if user_attributes else "unknown",
                    "role": user_attributes.get("role", "user") if user_attributes else "user",
                }

                result = await self.reasoning_engine.process_request(
                    user_query=content,
                    user_context=user_context,
                    conversation_id=session_id
                )

                # Send reasoning steps (simplified for now)
                yield AGUIEvent(
                    type="REASONING_STEP",
                    data={
                        "step": "processing",
                        "action": "Analyzing request with Moveworks reasoning engine...",
                        "status": "completed",
                        "timestamp": datetime.now().isoformat()
                    }
                )

                response_text = result.response

                # Send plugin information if used
                if result.selected_plugins:
                    for plugin in result.selected_plugins:
                        yield AGUIEvent(
                            type="PLUGIN_EXECUTED",
                            data={
                                "plugin_name": plugin,
                                "timestamp": datetime.now().isoformat()
                            }
                        )

                # Check if human-in-the-loop actions are needed
                if result.user_actions:
                    for action in result.user_actions:
                        yield AGUIEvent(
                            type="CONFIRMATION_REQUIRED",
                            data={
                                "action": action.get("action", "Unknown action"),
                                "importance": action.get("importance", "medium"),
                                "timestamp": datetime.now().isoformat()
                            }
                        )
            else:
                # Fallback if no conversational agent
                response_text = "I'll process your request and ensure you get the help you need for this matter."

            # Process through reasoning engine and handle interrupts
            try:
                user_context = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "department": user_attributes.get("department", "unknown") if user_attributes else "unknown",
                    "role": user_attributes.get("role", "user") if user_attributes else "user"
                }

                reasoning_result = await self.reasoning_engine.process_request(
                    user_query=content,
                    user_context=user_context,
                    conversation_id=session_id
                )

                response_text = reasoning_result.response

                # Send plugin information if used
                if reasoning_result.selected_plugins:
                    yield AGUIEvent(
                        type="PLUGIN_EXECUTED",
                        data={
                            "plugins": reasoning_result.selected_plugins,
                            "execution_summary": reasoning_result.execution_summary,
                            "timestamp": datetime.now().isoformat()
                        }
                    )

            except Exception as e:
                # Check if this is a LangGraph interrupt
                if hasattr(e, '__interrupt__') or "interrupt" in str(e).lower():
                    # Handle interrupt by emitting AG-UI event
                    interrupt_info = await self._handle_langgraph_interrupt(session_id, content, user_context)

                    if interrupt_info.get("interrupt_data"):
                        interrupt_data = interrupt_info["interrupt_data"]

                        if interrupt_data.get("type") == "user_confirmation":
                            yield AGUIEvent(
                                type="CONFIRMATION_REQUIRED",
                                data={
                                    "confirmation": {
                                        "action": interrupt_data.get("question", "Proceed?"),
                                        "importance": "high",
                                        "options": interrupt_data.get("options", ["yes", "no"]),
                                        "status": interrupt_data.get("status", ""),
                                        "timestamp": datetime.now().isoformat()
                                    },
                                    "session_id": session_id
                                }
                            )
                            return
                        else:
                            # Generic interrupt
                            yield AGUIEvent(
                                type="USER_INPUT_REQUIRED",
                                data={
                                    "message": interrupt_data.get("question", "I need your input to continue."),
                                    "options": interrupt_data.get("options", []),
                                    "session_id": session_id,
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
                            return
                    else:
                        # Fallback for unknown interrupt
                        yield AGUIEvent(
                            type="CONFIRMATION_REQUIRED",
                            data={
                                "confirmation": {
                                    "action": "Continue processing?",
                                    "importance": "medium",
                                    "timestamp": datetime.now().isoformat()
                                },
                                "session_id": session_id
                            }
                        )
                        return
                else:
                    # Real error, not an interrupt
                    response_text = f"I encountered an error: {str(e)}"

            # Check for slot clarification scenarios
            if any(word in content.lower() for word in ["book", "schedule", "meeting"]) and "when" not in content.lower():
                yield AGUIEvent(
                    type="SLOT_CLARIFICATION_REQUIRED",
                    data={
                        "slot_name": "date_time",
                        "message": "When would you like to schedule this?",
                        "options": ["Today", "Tomorrow", "Next week"],
                        "session_id": session_id
                    }
                )
                return

            # Generate response
            import random
            mock_responses = [
                "I understand you need help with that. Let me process your request and provide you with the appropriate assistance.",
                "I'm analyzing your request and determining the best way to help you with this task.",
                "Based on your request, I can help you complete this task efficiently and effectively.",
                "Let me gather the necessary information to provide you with a comprehensive solution.",
                "I'll process your request and ensure you get the help you need for this matter."
            ]
            response_text = random.choice(mock_responses)

            # Implement proper AG-UI streaming pattern: Start -> Content -> End
            message_id = f"msg_{int(datetime.now().timestamp() * 1000)}"

            # 1. Send TEXT_MESSAGE_START
            yield AGUIEvent(
                type="TEXT_MESSAGE_START",
                data={
                    "messageId": message_id,
                    "role": "assistant"
                }
            )

            # 2. Send TEXT_MESSAGE_CONTENT chunks (word by word streaming)
            words = response_text.split()

            for word in words:
                # Add delay before sending each word for visible streaming
                await asyncio.sleep(0.2)  # Balanced speed for production streaming

                yield AGUIEvent(
                    type="TEXT_MESSAGE_CONTENT",
                    data={
                        "messageId": message_id,
                        "delta": word + " "
                    }
                )

            # 3. Send TEXT_MESSAGE_END
            yield AGUIEvent(
                type="TEXT_MESSAGE_END",
                data={
                    "messageId": message_id
                }
            )

            # Send completion event
            yield AGUIEvent(
                type="RUN_COMPLETED",
                data={
                    "response": {
                        "response": response_text,
                        "success": True,
                        "reasoning_state": "complete"
                    },
                    "session_id": session_id,
                    "agent_id": selected_agent_id,
                    "routing_info": routing_info
                }
            )

        except Exception as e:
            logger.error(f"Error in message stream: {e}")
            yield AGUIEvent(
                type="ERROR",
                data={
                    "error": str(e),
                    "session_id": session_id
                }
            )

    async def initialize(self, db_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the AG-UI server with database configuration."""
        try:
            # Initialize Agent Studio database
            await agent_studio_db.initialize()

            # Initialize the simple reasoning engine
            logger.info("Initializing AG-UI server with reasoning engine...")

            self.reasoning_engine = await create_moveworks_reasoning_engine()

            logger.info("AG-UI server initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AG-UI server: {e}")
            raise


# Create global server instance
server = AGUIServer()

# Add startup event to initialize the server
@server.app.on_event("startup")
async def startup_event():
    """Initialize the server on startup."""
    print("🔄 Startup event triggered - initializing server...")
    logger.info("Startup event triggered - initializing server...")
    try:
        await server.initialize()
        print("✅ Server initialization completed successfully")
        logger.info("Server initialization completed successfully")
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        logger.error(f"Server initialization failed: {e}")
        raise

# Create the main FastAPI app with lifespan
app = FastAPI(
    title="Moveworks AI Platform",
    version="1.0.0",
    description="Moveworks-style Agentic Reasoning Engine",
    lifespan=lifespan
)

# Main execution block
if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting AG-UI Server with Moveworks Three-Loop Reasoning...")

    # Create server instance
    server = AGUIServer()

    # Run the server
    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )

    async def shutdown(self) -> None:
        """Shutdown the server and cleanup resources."""
        try:
            if self.agent_router:
                await self.agent_router.close()
            logger.info("AG-UI server shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global server instance
server = AGUIServer()


# FastAPI app instance for uvicorn
# Add CORS middleware to main app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the server routes
app.mount("/", server.app)


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run the server
    uvicorn.run(
        "src.ag_ui_server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

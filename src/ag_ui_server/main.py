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

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning.moveworks_smart_reasoning_agent import MoveworksSmartReasoningAgent
from reasoning.dynamic_slot_resolver import DynamicSlotResolver
from langgraph.checkpoint.memory import MemorySaver
from agent_studio.database import agent_studio_db
from agent_studio.api import create_agent_studio_router
from agent_studio.langgraph_integration import AgentStudioLangGraphIntegration
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
    
    def __init__(self, lifespan=None):
        self.app = FastAPI(
            title="Moveworks AG-UI Server",
            version="1.0.0",
            lifespan=lifespan
        )
        self.connection_manager = ConnectionManager()
        
        # Initialize components
        self.reasoning_engine: Optional[MoveworksSmartReasoningAgent] = None
        self.agent_studio_integration: Optional[AgentStudioLangGraphIntegration] = None
        self._initialization_lock = asyncio.Lock()
        self._initialized = False

        # Create shared memory instance for consistent state across components
        self.shared_memory = MemorySaver()
        self.dynamic_slot_resolver = None
        
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
                logger.info("Starting reasoning engine initialization", extra={
                    "component": "reasoning_engine",
                    "action": "initialize",
                    "stage": "begin"
                })

                # Initialize Agent Studio database
                logger.info("Initializing Agent Studio database", extra={
                    "component": "database",
                    "action": "initialize",
                    "database": "agent_studio"
                })
                await agent_studio_db.initialize()
                logger.info("Agent Studio database initialized", extra={
                    "component": "database",
                    "action": "initialize",
                    "database": "agent_studio",
                    "status": "success"
                })

                # Initialize the reasoning engine with shared memory
                logger.info("Creating reasoning engine instance", extra={
                    "component": "reasoning_engine",
                    "action": "create_instance",
                    "engine_type": "MoveworksSmartReasoningAgent"
                })
                self.reasoning_engine = MoveworksSmartReasoningAgent(shared_memory=self.shared_memory)

                logger.info("Initializing reasoning engine", extra={
                    "component": "reasoning_engine",
                    "action": "initialize",
                    "stage": "engine_init"
                })
                await self.reasoning_engine.initialize()

                # Initialize dynamic slot resolver with shared memory
                logger.info("Creating dynamic slot resolver instance", extra={
                    "component": "slot_resolver",
                    "action": "create_instance",
                    "resolver_type": "DynamicSlotResolver"
                })
                self.dynamic_slot_resolver = DynamicSlotResolver(shared_memory=self.shared_memory)

                logger.info("Initializing dynamic slot resolver", extra={
                    "component": "slot_resolver",
                    "action": "initialize"
                })
                await self.dynamic_slot_resolver.initialize()

                # Initialize Agent Studio integration during server startup
                logger.info("Initializing Agent Studio integration", extra={
                    "component": "agent_studio_integration",
                    "action": "initialize",
                    "stage": "begin"
                })
                from src.agent_studio.langgraph_integration import agent_studio_integration
                await agent_studio_integration.initialize()
                logger.info("Agent Studio integration initialized", extra={
                    "component": "agent_studio_integration",
                    "action": "initialize",
                    "stage": "complete",
                    "status": "success"
                })

                self._initialized = True
                logger.info("Reasoning engine initialization completed successfully", extra={
                    "component": "reasoning_engine",
                    "action": "initialize",
                    "stage": "complete",
                    "status": "success"
                })

            except Exception as e:
                logger.error("Failed to initialize reasoning engine", extra={
                    "component": "reasoning_engine",
                    "action": "initialize",
                    "stage": "failed",
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                })
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
                # Return mock plugins for now (can be enhanced with reasoning engine plugin discovery)
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
                # Return system status (using reasoning engine status)
                # System status
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
                    "reasoning_engine": hasattr(self, 'reasoning_engine') and self.reasoning_engine is not None,
                    "websocket_server": True,
                    "smart_reasoning_agent": hasattr(self, 'reasoning_engine') and self.reasoning_engine is not None
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
                # Return connector test status (can be enhanced with actual connector testing)
                # For now, return mock connector status
                    return JSONResponse(content={
                        "message": "Connector testing not implemented yet",
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

            # Use the global Agent Studio integration
            from src.agent_studio.langgraph_integration import agent_studio_integration

            result = await agent_studio_integration.process_message(
                content=content,
                user_id=user_id,
                session_id=session_id,
                user_attributes=user_attributes
            )

            # Convert Agent Studio result to reasoning result format
            reasoning_result = type('ReasoningResult', (), {
                'response': result.get('response', 'No response available'),
                'success': result.get('success', False),
                'selected_plugins': result.get('selected_plugins', []),
                'user_actions': result.get('user_actions', []),
                'execution_summary': result.get('execution_summary', {}),
                'conversation_id': session_id,
                'timestamp': datetime.now()
            })()

            # Convert result to dict for API response
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

    # Note: Interrupt handling is now managed by the MoveworksSmartReasoningAgent
    # The reasoning agent handles all LangGraph state management internally

    # Note: Conversation resumption is now managed by the MoveworksSmartReasoningAgent
    # The reasoning agent handles all conversation state and resumption internally



    async def _process_message_stream(
        self,
        content: str,
        user_id: str,
        session_id: str,
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[AGUIEvent, None]:
        """Process message using Moveworks Smart Reasoning Agent with streaming events."""
        try:
            # Send thinking event
            yield AGUIEvent(
                type="THINKING",
                data={
                    "message": "Analyzing your request...",
                    "session_id": session_id
                }
            )

            # Use Agent Studio integration for proper process detection and slot resolution
            result = await self._process_chat_message(content, user_id, session_id, user_attributes)

            # Send agent selection event based on reasoning result
            agent_info = "Moveworks Smart Reasoning Agent"
            if result.get("selected_plugins"):
                agent_info = f"Moveworks Agent with {len(result['selected_plugins'])} plugins"

            yield AGUIEvent(
                type="AGENT_SELECTED",
                data={
                    "agent_id": "moveworks_smart_reasoning_agent",
                    "confidence": 0.95,
                    "routing_info": {
                        "selected_agent": {
                            "name": agent_info,
                            "description": "Moveworks three-loop reasoning with plugin selection"
                        },
                        "reason": "Using Moveworks Smart Reasoning Agent",
                        "plugins_used": result.get("selected_plugins", [])
                    },
                    "session_id": session_id
                }
            )

            # Send reasoning started event
            yield AGUIEvent(
                type="REASONING_STARTED",
                data={
                    "agent_id": "moveworks_smart_reasoning_agent",
                    "session_id": session_id,
                    "message": "🧠 Processing with Moveworks reasoning engine...",
                    "explanation": "Using three-loop reasoning: planning, execution, and user feedback."
                }
            )

            # Get response from reasoning result
            response_text = result.get("response", "I apologize, but I couldn't process your request.")
            success = result.get("success", False)

            if not success:
                response_text = result.get("response", "I encountered an error processing your request.")

            # Send plugin information if used
            selected_plugins = result.get("selected_plugins", [])
            if selected_plugins:
                for plugin in selected_plugins:
                    yield AGUIEvent(
                        type="PLUGIN_EXECUTED",
                        data={
                            "plugin_name": plugin,
                            "timestamp": datetime.now().isoformat()
                        }
                    )

            # Check if human-in-the-loop actions are needed
            user_actions = result.get("user_actions", [])
            if user_actions:
                for action in user_actions:
                    yield AGUIEvent(
                        type="CONFIRMATION_REQUIRED",
                        data={
                            "action": action.get("action", "Unknown action"),
                            "importance": action.get("importance", "medium"),
                            "timestamp": datetime.now().isoformat()
                        }
                    )

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
                        "success": success,
                        "reasoning_state": "complete"
                    },
                    "session_id": session_id,
                    "agent_id": "moveworks_smart_reasoning_agent",
                    "selected_plugins": selected_plugins,
                    "user_actions": user_actions
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

            self.reasoning_engine = MoveworksSmartReasoningAgent()
            await self.reasoning_engine.initialize()

            logger.info("AG-UI server initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AG-UI server: {e}")
            raise

    # Note: Process detection is now handled by MoveworksSmartReasoningAgent
    # The reasoning agent uses the proper Moveworks architecture with plugin selection,
    # manifest generation, and three-loop reasoning instead of manual process detection.


# Create global server instance first
server = None

# Lifespan context manager for proper startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting application initialization", extra={
        "component": "server",
        "action": "startup",
        "stage": "begin"
    })
    try:
        # Initialize the server
        global server
        await server.initialize()
        logger.info("Application initialization completed successfully", extra={
            "component": "server",
            "action": "startup",
            "stage": "complete"
        })
        logger.info("Server initialization completed successfully")
        yield
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        logger.error(f"Server initialization failed: {e}")
        raise
    finally:
        print("🔄 Application shutdown")
        logger.info("Application shutdown")

# Create server instance with lifespan
server = AGUIServer(lifespan=lifespan)

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
        port=8081,
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
    allow_origins=["*"],
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
        port=8080,
        reload=True,
        log_level="info"
    )

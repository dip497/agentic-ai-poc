"""
Moveworks Smart Reasoning Agent.
LLM that decides whether to respond directly or use reasoning/plugin tools.
Includes the three-loop reasoning engine as a tool for complex tasks.
"""

import logging
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated

from .moveworks_reasoning_engine import MoveworksThreeLoopEngine
from reasoning.plugin_selection_engine import MoveworksPluginSelector
from reasoning.multi_plugin_response import MultiPluginResponseEngine
from reasoning.moveworks_slot_memory_manager import moveworks_slot_memory_manager
from config.loader import MoveworksConfigLoader
from config.reasoning_config import get_reasoning_config
from llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class MoveworksSmartReasoningState(TypedDict):
    """State for Moveworks Smart Reasoning Agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    user_context: Dict[str, Any]
    conversation_id: str
    
    # Plugin state
    selected_plugins: List[str]
    plugin_results: List[Dict[str, Any]]
    
    # Reasoning state
    reasoning_active: bool
    reasoning_result: Optional[Dict[str, Any]]
    
    # Memory state
    memory_snapshot: Optional[Dict[str, Any]]
    user_actions: List[Dict[str, Any]]


@dataclass
class MoveworksSmartReasoningResult:
    """Result from Moveworks Smart Reasoning Agent."""
    success: bool
    response: str
    selected_plugins: List[str] = field(default_factory=list)
    user_actions: List[Dict[str, Any]] = field(default_factory=list)
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    conversation_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class MoveworksSmartReasoningAgent:
    """
    Moveworks Smart Reasoning Agent that decides whether to:
    1. Respond directly (for simple queries like "hi")
    2. Use three-loop reasoning (for complex tasks requiring planning)
    3. Use plugin tools (for specific actions)
    4. Use search tools (for information gathering)
    """
    
    def __init__(self, shared_memory=None):
        """Initialize the Moveworks Smart Reasoning Agent."""
        self.config = get_reasoning_config()
        self.llm_factory = LLMFactory()
        self.shared_memory = shared_memory

        # Initialize three-loop reasoning engine
        self.three_loop_engine: Optional[MoveworksThreeLoopEngine] = None

        # Initialize LLM with tools
        self.llm = None
        self.graph = None

        logger.info("Moveworks Smart Reasoning Agent initialized")
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Moveworks Smart Reasoning Agent components...")
        
        # Initialize three-loop reasoning engine
        from config.loader import MoveworksConfigLoader
        config_loader = MoveworksConfigLoader()
        self.three_loop_engine = MoveworksThreeLoopEngine(config_loader)
        await self.three_loop_engine.initialize()
        
        # Initialize LLM with tools
        self.llm = self._initialize_llm_with_tools()
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("Moveworks Smart Reasoning Agent fully initialized")
    
    def _initialize_llm_with_tools(self):
        """Initialize LLM with reasoning and plugin tools."""
        base_llm = self.llm_factory.get_reasoning_llm()
        
        # Define tools that the LLM can choose to use
        tools = [
            self._create_three_loop_reasoning_tool(),
            self._create_plugin_tool(),
            self._create_search_tool()
        ]
        
        # Bind tools to LLM
        return base_llm.bind_tools(tools)
    
    def _create_three_loop_reasoning_tool(self):
        """Create three-loop reasoning tool for complex tasks."""
        @tool
        def moveworks_three_loop_reasoning(
            task_description: str,
            complexity_level: Literal["medium", "complex"] = "medium"
        ) -> str:
            """
            Use this tool for complex tasks that require detailed planning and execution.
            Examples: password reset, software installation, troubleshooting, multi-step processes.
            This triggers the full Moveworks three-loop reasoning process.
            """
            logger.info(f"Three-loop reasoning tool called for: {task_description}")
            
            # This will be handled by the tool execution node
            return f"Initiated three-loop reasoning for: {task_description} (complexity: {complexity_level})"
        
        return moveworks_three_loop_reasoning
    
    def _create_plugin_tool(self):
        """Create plugin tool for specific actions."""
        @tool
        def moveworks_plugin_execution(
            plugin_name: str,
            action: str,
            parameters: Dict[str, Any] = None
        ) -> str:
            """
            Use this tool to execute specific Moveworks plugins for actions like:
            - IT operations (password reset, software install)
            - HR operations (time off, benefits)
            - Facilities (room booking, equipment)
            """
            logger.info(f"Plugin tool called: {plugin_name} - {action}")
            
            if parameters is None:
                parameters = {}
            
            return f"Executed {plugin_name} plugin with action: {action}"
        
        return moveworks_plugin_execution
    
    def _create_search_tool(self):
        """Create search tool for information gathering."""
        @tool
        def moveworks_search(
            query: str,
            search_type: Literal["web", "knowledge", "documents"] = "knowledge"
        ) -> str:
            """
            Use this tool to search for information when you need to:
            - Look up company policies
            - Find documentation
            - Get current information
            """
            logger.info(f"Search tool called: {query} (type: {search_type})")
            
            return f"Search results for: {query} (type: {search_type})"
        
        return moveworks_search
    
    def _build_graph(self):
        """Build the Moveworks Smart Reasoning Agent graph."""
        # Create the graph
        builder = StateGraph(MoveworksSmartReasoningState)
        
        # Add nodes
        builder.add_node("reasoning_llm", self._reasoning_llm_node)
        builder.add_node("tool_execution", self._tool_execution_node)
        
        # Add edges
        builder.add_edge(START, "reasoning_llm")
        builder.add_conditional_edges(
            "reasoning_llm",
            self._should_use_tools,
            {
                "tools": "tool_execution",
                "end": END
            }
        )
        builder.add_edge("tool_execution", "reasoning_llm")
        
        # Compile with shared memory
        checkpointer = self.shared_memory or MemorySaver()
        return builder.compile(checkpointer=checkpointer)
    
    def _reasoning_llm_node(self, state: MoveworksSmartReasoningState) -> Dict[str, Any]:
        """Main reasoning LLM node that decides what to do."""
        logger.info("Reasoning LLM processing query")

        # Simple approach like LangChain ReAct - let LLM decide naturally
        system_message = self._create_system_message(state)
        messages = [system_message] + state["messages"]

        # Use LLM with tools - it will decide whether to use them or not
        response = self.llm.invoke(messages)

        return {"messages": [response]}
    


    def _create_system_message(self, state: MoveworksSmartReasoningState) -> BaseMessage:
        """Create system message for complex queries with tools."""
        # Simple, natural system prompt like LangChain ReAct agents
        system_prompt = "You are a helpful AI assistant."

        return AIMessage(content=system_prompt)
    
    async def _tool_execution_node(self, state: MoveworksSmartReasoningState) -> Dict[str, Any]:
        """Execute tools, including three-loop reasoning when needed."""
        last_message = state["messages"][-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return {"messages": []}
        
        tool_messages = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "moveworks_three_loop_reasoning":
                # Execute three-loop reasoning
                result = await self._execute_three_loop_reasoning(state, tool_args)
                tool_messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
            else:
                # Execute other tools normally
                tool_messages.append(ToolMessage(
                    content=f"Executed {tool_name} with args: {tool_args}",
                    tool_call_id=tool_call["id"]
                ))
        
        return {"messages": tool_messages}
    
    async def _execute_three_loop_reasoning(self, state: MoveworksSmartReasoningState, tool_args: Dict[str, Any]) -> str:
        """Execute the three-loop reasoning engine."""
        if not self.three_loop_engine:
            return "Three-loop reasoning engine not available"
        
        try:
            result = await self.three_loop_engine.process_request(
                user_query=state["user_query"],
                user_context=state["user_context"],
                conversation_id=state["conversation_id"]
            )
            
            # Store reasoning result in state
            state["reasoning_active"] = True
            state["reasoning_result"] = {
                "success": result.success,
                "response": result.response,
                "plugins": result.selected_plugins
            }
            
            return result.response
            
        except Exception as e:
            logger.error(f"Error in three-loop reasoning: {e}")
            return f"Error in reasoning process: {str(e)}"
    
    def _should_use_tools(self, state: MoveworksSmartReasoningState) -> str:
        """Determine if tools should be used based on the last message."""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info("Tools will be executed")
            return "tools"
        else:
            logger.info("Direct response, no tools needed")
            return "end"
    
    async def process_request(
        self,
        user_query: str,
        user_context: Dict[str, Any],
        conversation_id: str
    ) -> MoveworksSmartReasoningResult:
        """Process user request using Moveworks Smart Reasoning Agent."""
        try:
            # Ensure initialization
            if not self.graph:
                await self.initialize()
            
            logger.info(f"Processing request: {user_query[:100]}...")
            
            # Initialize state
            state = MoveworksSmartReasoningState(
                messages=[HumanMessage(content=user_query)],
                user_query=user_query,
                user_context=user_context,
                conversation_id=conversation_id,
                selected_plugins=[],
                plugin_results=[],
                reasoning_active=False,
                reasoning_result=None,
                memory_snapshot=None,
                user_actions=[]
            )
            
            # Run the graph
            config = {"configurable": {"thread_id": conversation_id}}
            final_state = await self.graph.ainvoke(state, config)
            
            # Extract final response
            final_response = final_state["messages"][-1].content
            
            return MoveworksSmartReasoningResult(
                success=True,
                response=final_response,
                selected_plugins=final_state.get("selected_plugins", []),
                user_actions=final_state.get("user_actions", []),
                execution_summary={
                    "reasoning_used": final_state.get("reasoning_active", False),
                    "tools_used": len([m for m in final_state["messages"] if isinstance(m, ToolMessage)]) > 0,
                    "conversation_id": conversation_id
                },
                conversation_id=conversation_id
            )
            
        except Exception as e:
            logger.error(f"Error in Moveworks Smart Reasoning Agent: {e}")
            return MoveworksSmartReasoningResult(
                success=False,
                response=f"I encountered an error processing your request: {str(e)}",
                execution_summary={"error": str(e)},
                conversation_id=conversation_id
            )

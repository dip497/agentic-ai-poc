"""
Main application for the Moveworks-style conversational AI system.

This application demonstrates the exact Moveworks architecture using LangChain
for LLM integration and following Moveworks' conversational process patterns.
"""

import asyncio
import logging
import os
from typing import Dict, Any

from src.agent.moveworks_agent import MoveworksConversationalAgent
from src.config.loader import load_moveworks_config
from src.llm.llm_factory import LLMFactory


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MoveworksSystem:
    """Main system class that orchestrates the Moveworks-style AI."""
    
    def __init__(self, config_path: str = "config/moveworks_config.yml"):
        """Initialize the system."""
        self.config_path = config_path
        self.agent = None
        self.config = None
    
    async def initialize(self):
        """Initialize the system with configuration."""
        logger.info("🚀 Initializing Moveworks-style Conversational AI System...")
        
        # Check for required API keys based on LLM configuration
        self._check_llm_requirements()
        
        try:
            # Load configuration
            self.config = load_moveworks_config(self.config_path)
            logger.info(f"✅ Loaded configuration with {len(self.config['plugins'])} plugins")
            
            # Initialize agent with LLM config
            llm_config = self.config.get("llm_config", {}).get("default", {
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "temperature": 0.1,
                "max_tokens": 2000
            })

            self.agent = MoveworksConversationalAgent(llm_config)
            
            # Register plugins
            for plugin in self.config["plugins"]:
                self.agent.register_plugin(plugin)
            
            logger.info("✅ System initialization complete!")
            
            # Show available processes
            self._show_available_processes()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            raise

    def _check_llm_requirements(self):
        """Check LLM provider requirements and API keys."""
        # For now, just check if OpenAI key exists (will be enhanced after config loading)
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
            logger.warning("⚠️  No LLM API keys found in environment variables")
            logger.info("💡 Set one of the following:")
            logger.info("   - OpenAI: export OPENAI_API_KEY='your-key'")
            logger.info("   - Gemini: export GOOGLE_API_KEY='your-key'")
            logger.info("   - OpenRouter: export OPENROUTER_API_KEY='your-key'")

    def _show_available_processes(self):
        """Show available conversational processes."""
        logger.info("\n📋 Available Conversational Processes:")
        logger.info("=" * 50)
        
        for plugin in self.config["plugins"]:
            logger.info(f"\n🔌 Plugin: {plugin.name}")
            logger.info(f"   Description: {plugin.description}")
            
            for process in plugin.conversational_processes:
                logger.info(f"\n   📝 Process: {process.title}")
                logger.info(f"      Description: {process.description}")
                logger.info(f"      Trigger Examples:")
                for utterance in process.trigger_utterances[:3]:  # Show first 3
                    logger.info(f"        • \"{utterance}\"")
                if len(process.trigger_utterances) > 3:
                    logger.info(f"        • ... and {len(process.trigger_utterances) - 3} more")
        
        logger.info("\n" + "=" * 50)
    
    async def chat(
        self,
        message: str,
        user_id: str = "demo_user",
        session_id: str = "demo_session",
        user_attributes: Dict[str, Any] = None
    ) -> str:
        """Process a chat message."""
        if not self.agent:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        if user_attributes is None:
            user_attributes = {
                "email_addr": "demo.user@company.com",
                "department": "Engineering",
                "role": "Developer",
                "employment_status": "active",
                "tenure_months": 12
            }
        
        result = await self.agent.process_message(message, user_id, session_id, user_attributes)
        return result["response"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        if not self.agent:
            return {"error": "System not initialized"}
        
        return self.agent.get_conversation_stats()


async def interactive_demo():
    """Run interactive demo of the system."""
    system = MoveworksSystem()
    await system.initialize()
    
    print("\n🤖 Moveworks-style AI Assistant")
    print("=" * 50)
    print("Type 'quit' to exit, 'stats' to see system statistics")
    print("Type 'help' to see available processes")
    print("Type 'clear' to start a new conversation")
    print("=" * 50)
    
    session_id = "interactive_session"
    user_attributes = {
        "email_addr": "demo.user@company.com",
        "department": "Engineering",
        "role": "Developer",
        "employment_status": "active",
        "tenure_months": 12
    }
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'stats':
                stats = system.get_stats()
                print(f"\n📊 System Stats:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            elif user_input.lower() == 'help':
                system._show_available_processes()
                continue
            elif user_input.lower() == 'clear':
                system.agent.clear_conversation(session_id)
                print("🔄 Conversation cleared. Starting fresh!")
                continue
            elif not user_input:
                continue
            
            # Process message
            response = await system.chat(user_input, "demo_user", session_id, user_attributes)
            print(f"\n🤖 Assistant: {response}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


async def run_example_scenarios():
    """Run example scenarios to demonstrate system capabilities."""
    system = MoveworksSystem()
    await system.initialize()
    
    print("\n🎯 Running Example Scenarios")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "PTO Balance Request",
            "messages": [
                "What's my vacation balance?",
                "vacation"
            ]
        },
        {
            "name": "Feature Request Update",
            "messages": [
                "Update feature request FR-001 to completed",
                "completed"
            ]
        },
        {
            "name": "Procurement Purchase Request",
            "messages": [
                "I need to submit a purchase request for a laptop",
                "1",  # quantity
                "I need this for development work",  # justification
                "yes"  # organization acknowledgment
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n📋 Scenario {i+1}: {scenario['name']}")
        print("-" * 40)
        
        session_id = f"scenario_{i+1}"
        user_attributes = {
            "email_addr": "demo.user@company.com",
            "department": "Engineering",
            "role": "Developer",
            "employment_status": "active",
            "tenure_months": 12
        }
        
        for j, message in enumerate(scenario['messages']):
            print(f"\n💬 User: {message}")
            response = await system.chat(message, "demo_user", session_id, user_attributes)
            print(f"🤖 Assistant: {response}")
        
        print("\n" + "=" * 50)
    
    # Show final stats
    stats = system.get_stats()
    print(f"\n📊 Final System Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")


async def test_process_matching():
    """Test process matching capabilities."""
    system = MoveworksSystem()
    await system.initialize()
    
    print("\n🧪 Testing Process Matching")
    print("=" * 50)
    
    test_utterances = [
        "What's my vacation balance?",
        "How much PTO do I have left?",
        "Update FR-123 to completed",
        "Change feature request status",
        "I need to buy office supplies",
        "Submit a purchase request",
        "How do I reset my password?",  # Should not match
        "What's the weather like?"      # Should not match
    ]
    
    for utterance in test_utterances:
        print(f"\n💬 Testing: \"{utterance}\"")
        
        try:
            response = await system.chat(utterance, "test_user", f"test_{hash(utterance)}")
            print(f"🤖 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 30)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "demo":
            # Run example scenarios
            asyncio.run(run_example_scenarios())
        elif command == "test":
            # Test process matching
            asyncio.run(test_process_matching())
        elif command == "interactive":
            # Interactive chat
            asyncio.run(interactive_demo())
        else:
            print("Usage: python main.py [demo|test|interactive]")
            print("  demo        - Run example scenarios")
            print("  test        - Test process matching")
            print("  interactive - Interactive chat mode")
            sys.exit(1)
    else:
        # Default to interactive mode
        asyncio.run(interactive_demo())

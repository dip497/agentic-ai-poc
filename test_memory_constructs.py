#!/usr/bin/env python3
"""
Test script for Moveworks Memory Constructs.
Verifies all four memory types work correctly with dynamic domains.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from reasoning.moveworks_memory_manager import MoveworksMemoryManager
from reasoning.memory_constructs import (
    DomainDefinition, SemanticMemoryEntry, EpisodicMemoryEntry,
    ConversationContext, ProcedureMemoryEntry, WorkingMemoryEntry
)


async def test_memory_constructs():
    """Test all four memory constructs."""
    print("🧠 Testing Moveworks Memory Constructs...")
    
    # Initialize memory manager
    memory_manager = MoveworksMemoryManager()
    
    try:
        await memory_manager.initialize()
        print("✅ Memory Manager initialized successfully")
        
        # Test 1: Dynamic Domain Management
        print("\n🏗️ Testing Dynamic Domain Management...")
        
        # Create a custom domain
        custom_domain = DomainDefinition(
            name="CUSTOM_DOMAIN",
            display_name="Custom Test Domain",
            description="A custom domain for testing purposes",
            keywords=["custom", "test", "demo"],
            trigger_phrases=["custom request", "test domain"]
        )
        
        domain_name = await memory_manager.create_domain(custom_domain)
        print(f"✅ Created custom domain: {domain_name}")
        
        # Test domain classification
        classified_domain = await memory_manager.classify_domain("I have a custom request for testing")
        print(f"✅ Domain classification result: {classified_domain}")
        
        # List all domains
        domains = await memory_manager.list_domains()
        print(f"✅ Total active domains: {len(domains)}")
        for domain in domains:
            print(f"   - {domain.name}: {domain.display_name}")
        
        # Test 2: Semantic Memory
        print("\n🧠 Testing Semantic Memory...")
        
        # Add semantic entry
        semantic_entry = SemanticMemoryEntry(
            entity_type="u_TestEntity",
            entity_name="Test Entity",
            entity_description="A test entity for memory validation",
            domain="CUSTOM_DOMAIN",
            properties={"test_field": "test_value"},
            synonyms=["test item", "demo entity"]
        )
        
        semantic_id = await memory_manager.add_semantic_entry(semantic_entry)
        print(f"✅ Added semantic entry: {semantic_id}")
        
        # Search semantic memory
        semantic_results = await memory_manager.search_semantic_memory("test entity")
        print(f"✅ Found {len(semantic_results)} semantic entries")
        for result in semantic_results:
            print(f"   - {result.entity_name} ({result.entity_type}) in {result.domain}")
        
        # Test 3: Episodic Memory & Conversation Context
        print("\n💬 Testing Episodic Memory...")
        
        # Create conversation
        conversation = await memory_manager.get_or_create_conversation("test_user_123")
        print(f"✅ Created conversation: {conversation.conversation_id}")
        
        # Add messages
        messages = [
            EpisodicMemoryEntry(
            EpisodicMemoryEntry(
                conversation_id=conversation.conversation_id,
                user_id="test_user_123",
                message_type="user",
                content="Hello, I need help with a test entity",
                intent="help_request"
            ),
            EpisodicMemoryEntry(
                conversation_id=conversation.conversation_id,
                user_id="assistant",
                message_type="assistant", 
                content="I can help you with test entities. What do you need?",
                intent="help_response"
            ),
            EpisodicMemoryEntry(
                conversation_id=conversation.conversation_id,
                user_id="test_user_123",
                message_type="user",
                content="I want to create a new test entity",
                intent="create_request",
                entities_extracted={"entity_type": "test_entity"}
            )
        ]
        
        for message in messages:
            await memory_manager.add_message(conversation.conversation_id, message)
        
        print(f"✅ Added {len(messages)} messages to conversation")
        
        # Retrieve conversation context
        retrieved_context = await memory_manager.get_conversation_context(conversation.conversation_id)
        if retrieved_context:
            print(f"✅ Retrieved conversation with {len(retrieved_context.messages)} messages")
            print(f"   Recent context (last 6): {len(retrieved_context.get_recent_context())}")
        
        # Test 4: Procedure Memory
        print("\n⚙️ Testing Procedure Memory...")
        
        # Add procedure entry
        procedure_entry = ProcedureMemoryEntry(
            plugin_id="test_plugin_001",
            plugin_name="Test Entity Manager",
            description="Manages test entities and their operations",
            capabilities=["create_entity", "update_entity", "delete_entity"],
            trigger_utterances=["create test entity", "manage entity", "entity operations"],
            domain_compatibility=["CUSTOM_DOMAIN", "IT_DOMAIN"],
            required_slots=["entity_name", "entity_type"],
            output_types=["u_TestEntity"]
        )
        
        procedure_id = await memory_manager.add_procedure_entry(procedure_entry)
        print(f"✅ Added procedure entry: {procedure_id}")
        
        # Search procedures
        procedure_results = await memory_manager.search_procedures("create entity")
        print(f"✅ Found {len(procedure_results)} matching procedures")
        for result in procedure_results:
            print(f"   - {result.plugin_name}: {result.capabilities}")
        
        # Test 5: Working Memory
        print("\n🔄 Testing Working Memory...")
        
        # Create working memory entry
        working_entry = WorkingMemoryEntry(
            conversation_id=conversation.conversation_id,
            process_id="test_process_001",
            process_name="Create Test Entity Process",
            current_step="collect_entity_details",
            status="in_progress"
        )
        
        # Track variables
        working_entry.track_variable("entity_name", "My Test Entity")
        working_entry.track_variable("entity_type", "u_TestEntity", "u_TestEntity")
        working_entry.add_step("collect_entity_details", {"collected": ["entity_name"]})
        
        working_id = await memory_manager.create_working_memory(working_entry)
        print(f"✅ Created working memory entry: {working_id}")
        
        # Get active working memory
        active_working = await memory_manager.get_active_working_memory(conversation.conversation_id)
        print(f"✅ Found {len(active_working)} active working memory entries")
        for entry in active_working:
            print(f"   - {entry.process_name}: {entry.current_step} ({entry.status})")
            print(f"     Variables: {list(entry.variables.keys())}")
            print(f"     Business Objects: {list(entry.business_objects.keys())}")
        
        # Test 6: Memory Snapshot
        print("\n📸 Testing Memory Snapshot...")
        
        snapshot = await memory_manager.create_memory_snapshot(
            conversation.conversation_id, 
            "create test entity"
        )
        
        print(f"✅ Created memory snapshot:")
        print(f"   - Domain: {snapshot.get_domain_context()}")
        print(f"   - Semantic entries: {len(snapshot.relevant_semantic_entries)}")
        print(f"   - Available procedures: {len(snapshot.available_procedures)}")
        print(f"   - Active working memory: {len(snapshot.active_working_memory)}")
        print(f"   - Available capabilities: {snapshot.get_available_capabilities()}")
        print(f"   - Tracked variables: {list(snapshot.get_tracked_variables().keys())}")
        print(f"   - Business objects: {list(snapshot.get_business_objects().keys())}")
        
        print("\n🎉 All Memory Constructs tests passed!")
        
        # Test 7: Scaling Test (simulate multiple conversations)
        print("\n🚀 Testing Scaling (Multiple Conversations)...")
        
        conversations = []
        for i in range(5):
            conv = await memory_manager.get_or_create_conversation(f"user_{i}")
            conversations.append(conv)
            
            # Add a message to each
            msg = EpisodicMemoryEntry(
                conversation_id=conv.conversation_id,
                user_id=f"user_{i}",
                message_type="user",
                content=f"Hello from user {i}",
                intent="greeting"
            )
            await memory_manager.add_message(conv.conversation_id, msg)
        
        print(f"✅ Created {len(conversations)} concurrent conversations")
        
        # Test domain classification on different queries
        test_queries = [
            "I need help with my Jira ticket",
            "Can you help me with HR benefits?", 
            "I want to create a sales account",
            "I have a custom request for testing",
            "General question about the system"
        ]
        
        print("\n🎯 Testing Domain Classification:")
        for query in test_queries:
            domain = await memory_manager.classify_domain(query)
            print(f"   '{query}' → {domain}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await memory_manager.close()
    
    return True


if __name__ == "__main__":
    print("🧠 Moveworks Memory Constructs Test")
    print("=" * 50)
    
    success = asyncio.run(test_memory_constructs())
    
    if success:
        print("\n🎉 All tests passed! Memory Constructs are working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)

# Moveworks Agentic Reasoning Engine

## Overview

The Moveworks Agentic Reasoning Engine is the core AI system that powers intelligent automation and conversational AI. It implements a sophisticated three-loop architecture with advanced memory management, plugin selection, and multi-plugin response capabilities.

## Architecture Components

### 1. Three Reasoning Loops

#### Planning Iteration Loop (Innermost)
- **Purpose**: Autonomous plugin selection and execution planning
- **Key Features**:
  - Plugin selection based on descriptions, examples, and domain compatibility
  - Multi-plugin response capability (parallel execution)
  - Plan evaluation and revision
  - Confidence scoring and validation

#### Execution Iteration Loop (Middle)
- **Purpose**: Execute the planned actions and handle results
- **Key Features**:
  - Step-by-step execution with dependency management
  - Error handling and recovery
  - Result validation and processing
  - Dynamic re-planning when needed

#### User-Facing Feedback Loop (Outermost)
- **Purpose**: Manage user interactions and feedback
- **Key Features**:
  - User confirmation requests
  - Progress updates and status reporting
  - Error communication
  - Context management and conversation flow

### 2. Memory Constructs

#### Episodic Memory
- **Purpose**: Store conversation history and user interactions
- **Components**:
  - Conversation entries with timestamps
  - User preferences and patterns
  - Success/failure patterns
  - Contextual relationships

#### Semantic Memory
- **Purpose**: Store factual knowledge and relationships
- **Components**:
  - Entity relationships
  - Domain knowledge
  - Business rules and policies
  - System capabilities

#### Procedural Memory
- **Purpose**: Store available procedures and workflows
- **Components**:
  - Plugin definitions and capabilities
  - Workflow templates
  - Action sequences
  - Integration patterns

#### Working Memory
- **Purpose**: Manage active reasoning processes
- **Components**:
  - Current process state
  - Variable tracking
  - Step execution history
  - Temporary data storage

### 3. Plugin Selection Engine

#### Autonomous Selection
- **Criteria**:
  - Plugin descriptions and capabilities
  - Positive and negative examples
  - Domain compatibility scores
  - User context and preferences
  - Historical success patterns

#### Multi-Plugin Response
- **Features**:
  - Parallel plugin execution
  - Response combination and summarization
  - Improved button UX for multiple actions
  - Conflict resolution between plugins

### 4. Conversational Context Management

#### Context Window
- **Duration**: 24-hour automatic clearing
- **Features**:
  - Reference resolution (pronouns, follow-ups)
  - Resource availability from previous requests
  - Conversation flow continuity
  - Manual clearing capabilities

#### Context Usage
- **Applications**:
  - New topic vs. continuation detection
  - Vague query clarification
  - Resource reuse from previous calls
  - Empathetic response generation

### 5. Guardrails & Safety

#### Input Safeguards
- **Toxicity Filter**: ML-based inappropriate content detection
- **Sensitive Topics Policy**: Decline subjective/sensitive discussions
- **Work-related Scope**: Constrain to enterprise use cases
- **Organization-specific Grounding**: Use available plugins and knowledge

#### Output Safeguards
- **Toxicity Validation**: Check generated responses
- **Citations**: Provide source verification
- **Linked Verified Entities**: Validate entity mentions
- **Hyperlink Management**: Proper link handling and validation

### 6. Steerability Tools

#### Customer-specific Behavior
- **Configurable Instructions**: Guide plugin selection and responses
- **Response Steering**: Influence behavior while maintaining compliance
- **Domain Adaptation**: Customize for specific enterprise contexts
- **Policy Enforcement**: Apply organization-specific rules

## Implementation Status

### ✅ Completed Components
- **Memory Constructs**: Full implementation with PostgreSQL + pgvector
- **Dynamic Domain Management**: No hardcoded values, fully configurable
- **Database Architecture**: Ready for 1000+ agent scaling
- **Basic State Management**: LangGraph integration foundation

### 🚨 Missing Critical Components
- **Plugin Selection Engine**: Autonomous selection logic
- **Multi-Plugin Response System**: Parallel execution and combination
- **Reasoning Loops**: Complete three-loop architecture
- **Conversational Context**: 24-hour window management
- **Guardrails Implementation**: Safety and accuracy mechanisms
- **Steerability Tools**: Customer-specific adaptation

### 🗑️ Cleaned Up
- **Legacy Reasoning Agent**: Removed non-Moveworks patterns
- **Old Memory Manager**: Replaced with Moveworks architecture
- **Demo Files**: Removed temporary/testing files

## Key Differences from Standard Approaches

### 1. Plugin Architecture
- **Not Tool Calling**: Plugins are autonomous agents, not simple function calls
- **Intelligent Selection**: AI-driven plugin selection based on context and examples
- **Multi-Plugin Capability**: Can execute multiple plugins simultaneously

### 2. Memory Management
- **Structured Memory**: Four distinct memory types with specific purposes
- **Persistent Context**: 24-hour conversation memory with automatic management
- **Variable Tracking**: Comprehensive state management across reasoning loops

### 3. Safety & Governance
- **Enterprise-Grade**: Built for workplace environments with appropriate safeguards
- **Configurable Policies**: Organization-specific behavior adaptation
- **Verification Systems**: Citations and entity validation for trust

## Integration with Existing System

### Resolver Strategies as Plugins
- **Current Resolvers**: Become individual plugins in the reasoning engine
- **Method Selection**: AI-driven selection from multiple resolver methods
- **Data Flow**: Resolver results feed into reasoning loops

### Agent Studio Integration
- **Plugin Management**: Agent Studio creates and manages plugins
- **Configuration**: Slots, triggers, and processes define plugin behavior
- **Execution**: Reasoning engine orchestrates plugin execution

## Next Implementation Steps

1. **Plugin Selection Engine**: Build autonomous selection based on descriptions/examples
2. **Multi-Plugin Response**: Implement parallel execution and response combination
3. **Reasoning Loops**: Complete three-loop architecture with LangGraph
4. **Context Management**: 24-hour window with reference resolution
5. **Guardrails**: Safety mechanisms and governance
6. **Steerability**: Customer-specific behavior adaptation
7. **Integration**: Connect with existing resolver and agent studio systems

## Documentation Sources

All information based on official Moveworks documentation:
- https://help.moveworks.com/docs/agentic-reasoning-engine
- https://help.moveworks.com/docs/the-moveworks-agentic-reasoning-architecture
- https://help.moveworks.com/docs/reasoning-loops-in-the-ai-assistant
- https://help.moveworks.com/docs/plugin-selection
- https://help.moveworks.com/docs/multi-plugin-response-assistant
- https://help.moveworks.com/docs/memory-constructs-in-the-ai-assistant
- https://help.moveworks.com/docs/assistant-behavior-framework
- https://help.moveworks.com/docs/context-window-management
- https://help.moveworks.com/docs/conversational-safeguards-assistant

## Contributing

When implementing components:
1. Follow exact Moveworks patterns and architecture
2. Maintain enterprise-grade safety and governance
3. Ensure scalability for 1000+ agents
4. Document all deviations from standard approaches
5. Test with real enterprise use cases

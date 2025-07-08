# Reasoning Loops - Moveworks Agentic Reasoning Engine

## Overview

Based on analysis of the official Moveworks documentation at https://help.moveworks.com/docs/reasoning-loops-in-the-ai-assistant, this document examines our current Reasoning Loops implementation against the actual Moveworks architecture.

## Actual Moveworks Reasoning Loops Architecture

According to the official documentation, Moveworks uses **three interconnected reasoning loops** to explore solutions and serve user requests:

### 1. Planning Iteration Loop (Innermost Loop)
**Purpose**: Independently explore the solution space of all possible tools and plugins to select the most useful one

**Components**:
- **Planner**: Uses LLM call to create plan by going through list of all available tools for the user
- **Plan Evaluator**: Uses another LLM to assess whether plan addresses user's need
- **Feedback Integration**: If gap identified, evaluator provides feedback to planner for new plan creation

**Process Flow**:
1. User query triggers planning module
2. Planner creates plan using available tools list
3. Plan evaluator assesses plan completeness
4. If gaps found, feedback provided to planner
5. Planning cycles continue until final plan created
6. Final plan provided to execution iteration loop

### 2. Execution Iteration Loop (Middle Loop)
**Purpose**: Execute the plan step-by-step and assess what to do next at each stage through plan → execute → adapt cycle

**Components**:
- **Plan Execution**: Takes plan from planning iteration and executes by calling tools
- **Outcome Observation**: Reasoning engine observes execution outcomes
- **Adaptive Decision Making**: Agentic reasoning LLM determines next steps

**Tool Types**:
- Search tools
- Workflow execution tools
- API calling tools
- Code execution tools
- LLM summarization tools
- Many more specialized components

**Outcome Handling**:
- **Successful Completion**: Output returned by search or workflow plugin
- **Error Handling**: Invalid operation, missing information, or user confirmation requests
- **Context Integration**: Uses semantic or episodic memory for decision making
- **User Updates**: Reasoning LLM provides user updates on steps being taken

### 3. User-Facing Feedback Loop (Outermost Loop)
**Purpose**: Interaction between agentic reasoning LLM and user for plan communication and feedback

**Two Main Scenarios**:
1. **User Input Required**: Agentic reasoning LLM determines user input needed
   - Prompts for confirmation
   - Requests feedback
   - Asks for required information

2. **New Request Initiation**: User initiates new or follow-on request
   - Sets planning iteration in motion
   - Triggers execution iteration cycle
   - Continues the reasoning loop cycle

## Issues in Our Current Implementation

### 1. Incorrect Loop Architecture
**Problem**: Our implementation doesn't match Moveworks three-loop pattern
- We have custom loop types that don't align with actual Moveworks architecture
- Missing proper planner/plan evaluator separation
- No proper tool exploration and selection mechanism
- Missing adaptive execution with outcome observation

### 2. Hardcoded Values Found
**Problem**: Multiple hardcoded values violate configuration-driven approach
- `max_planning_iterations = 3` - should be configurable
- `max_execution_steps = 5` - should be configurable
- `response_timeout = 30.0` - should be configurable
- No dynamic tool discovery mechanism

### 3. Missing Moveworks-Specific Features
**Problem**: Key Moveworks reasoning features not implemented
- No proper planner/plan evaluator LLM separation
- Missing tool exploration and selection logic
- No outcome observation and adaptive decision making
- Missing proper user feedback integration patterns
- No plan → execute → adapt cycle implementation

### 4. Tool Integration Issues
**Problem**: Tool calling doesn't match Moveworks patterns
- Missing specialized tool types (search, workflow, API, code, summarization)
- No proper tool outcome handling
- Missing error handling for invalid operations
- No user confirmation request handling

## Required Fixes

### 1. Implement Three-Loop Architecture
- Separate planning iteration loop with planner/evaluator
- Implement execution iteration loop with outcome observation
- Add user-facing feedback loop with proper interaction patterns

### 2. Remove All Hardcoded Values
- Make iteration limits configurable
- Configure timeout values
- Add dynamic tool discovery and selection

### 3. Add Proper Tool Integration
- Implement specialized tool types
- Add outcome observation and handling
- Support error handling and user confirmation requests
- Integrate with memory constructs for context

### 4. Fix Loop Interconnection
- Ensure proper data flow between loops
- Implement adaptive decision making
- Add proper feedback integration mechanisms

## Implementation Status
❌ **Not Implemented**: Three-loop architecture with proper separation
❌ **Not Implemented**: Planner/plan evaluator LLM separation
❌ **Hardcoded Values**: Multiple loop configuration values hardcoded
❌ **Missing Features**: Tool exploration, outcome observation, adaptive execution

## Next Steps
1. Study Moveworks three-loop architecture implementation
2. Remove all hardcoded values and implement configuration-driven approach
3. Implement proper planner/plan evaluator separation
4. Add tool exploration and selection mechanisms
5. Implement outcome observation and adaptive decision making
6. Add proper user feedback integration patterns

---

**Related Components:**
- [Memory Constructs](./memory-constructs.md) - Provides context for decision making
- [Plugin Selection](./plugin-selection.md) - Operates within planning and execution loops
- [Overview](./moveworks-agentic-reasoning-overview.md) - Complete architecture analysis

**Documentation Source:** https://help.moveworks.com/docs/reasoning-loops-in-the-ai-assistant

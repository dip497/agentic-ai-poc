# Plugin Selection - Moveworks Agentic Reasoning Engine

## Overview

Based on analysis of the official Moveworks documentation at https://help.moveworks.com/docs/plugin-selection, this document examines our current Plugin Selection implementation against the actual Moveworks architecture.

## Actual Moveworks Plugin Selection Architecture

According to the official documentation, Plugin Selection is the **AI Assistant's ability to autonomously determine which plugins to call to execute the plan for the user's request**.

### Integration with Reasoning Loops
Plugin Selection operates within **two inner reasoning loops**:

1. **Planning Iteration Loop**: Defines plan involving:
   - **Plugin Selection**: Which plugins to call
   - **Argument Passing**: Arguments to pass to selected plugins

2. **Execution Iteration Loop**: 
   - Takes plan from planning iteration
   - Executes by calling identified plugins
   - Observes outcomes
   - Can call other plugins or return to user

### Selection Criteria
Plugin selection is based on:

1. **Available Plugins**: Varies from customer to customer in the environment
2. **Plugin Descriptions**: Description of each plugin (native or custom)
3. **Examples**: Positive and negative examples provided for each plugin
4. **Request Matching**: Determines whether plugin is good match for user request

### Multi-Plugin Capability
- **Multi-Plugin Response**: AI Assistant can choose **multiple plugins at once**
- **Parallel Execution**: Can execute multiple plugins simultaneously
- **Outcome Integration**: Combines results from multiple plugins

### Plugin Types
**Native and Custom Plugin Treatment**:
- **Equal Treatment**: Reasoning treats native and custom plugins the same way
- **Description Dependency**: Plugin descriptions and examples are critical for both types
- **Native Plugin Limitations**: Cannot modify parameters for native plugins

### Plugin Competition Resolution
When native and custom plugins compete for same request:

1. **Offer Both**: Present both types to user
2. **Modify Examples**: Adjust custom plugin examples to cover different utterances
3. **Use Steerability Tools**: Guide reasoning to select one over the other
4. **Disable Native**: If functionality sufficiently covered by custom plugin

## Issues in Our Current Implementation

### 1. Missing Plugin Selection Architecture
**Problem**: Our implementation doesn't have proper plugin selection mechanism
- No autonomous plugin determination logic
- Missing plugin description and example evaluation
- No proper argument passing to selected plugins
- Missing multi-plugin response capability

### 2. Hardcoded Plugin Logic
**Problem**: Plugin selection appears to be hardcoded rather than dynamic
- No dynamic plugin discovery from environment
- Missing plugin description parsing
- No example-based matching logic
- No customer-specific plugin availability handling

### 3. Missing Multi-Plugin Support
**Problem**: No support for multiple plugin execution
- Cannot select multiple plugins at once
- No parallel plugin execution
- Missing outcome integration from multiple plugins
- No multi-plugin response coordination

### 4. Plugin Competition Issues
**Problem**: No mechanism to handle plugin competition
- Missing native vs custom plugin handling
- No steerability tools integration
- No plugin priority or preference system
- Missing plugin conflict resolution

## Required Fixes

### 1. Implement Plugin Selection Engine
- Add autonomous plugin determination logic
- Implement plugin description and example evaluation
- Support dynamic argument passing to plugins
- Add customer-specific plugin availability handling

### 2. Remove Hardcoded Plugin Logic
- Implement dynamic plugin discovery
- Add plugin description parsing and matching
- Support example-based plugin selection
- Make plugin selection configurable

### 3. Add Multi-Plugin Support
- Implement multiple plugin selection capability
- Add parallel plugin execution support
- Support outcome integration from multiple plugins
- Add multi-plugin response coordination

### 4. Add Plugin Competition Resolution
- Implement native vs custom plugin handling
- Integrate with steerability tools
- Add plugin priority and preference system
- Support plugin conflict resolution

## Implementation Status
❌ **Not Implemented**: Autonomous plugin selection engine
❌ **Not Implemented**: Plugin description and example evaluation
❌ **Hardcoded Logic**: Plugin selection appears hardcoded
❌ **Missing Features**: Multi-plugin response capability

## Next Steps
1. Study Moveworks plugin selection architecture implementation
2. Remove hardcoded plugin logic and implement dynamic selection
3. Add plugin description and example evaluation system
4. Implement multi-plugin response capability
5. Add plugin competition resolution mechanisms
6. Integrate with steerability tools for plugin guidance

---

**Related Components:**
- [Reasoning Loops](./reasoning-loops.md) - Plugin selection operates within these loops
- [Memory Constructs](./memory-constructs.md) - Uses procedure memory for tool selection
- [Overview](./moveworks-agentic-reasoning-overview.md) - Complete architecture analysis

**Documentation Source:** https://help.moveworks.com/docs/plugin-selection

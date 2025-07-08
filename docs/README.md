# Moveworks Agent Studio Documentation

## Core Components

### Agentic Reasoning Engine
- [Moveworks Agentic Reasoning Engine](./moveworks-agentic-reasoning-engine.md) - Complete three-loop architecture with memory constructs, plugin selection, and safety guardrails

### Resolver Strategies
- [Resolver Strategies](./resolver-strategies/README.md) - Complete guide to resolver strategies

### Actions
- [Actions](./actions/README.md) - HTTP Actions, Script Actions, Compound Actions, Built-in Actions

### Slots
- [Slots Overview](./slots/README.md) - Slot configuration and validation

## Key Discoveries

### 🚨 Critical Findings
1. **Resolver strategies = Collections of methods** (not single HTTP Actions)
2. **AI agent selects best method** based on user context
3. **1 Static OR Multiple Dynamic methods** per strategy (never mixed)
4. **Comprehensive data type system** (6 primitives + List variants + custom types):
   - **Primitives**: string, integer, number, boolean, User, object
   - **Lists**: List[string], List[integer], List[number], List[boolean], List[User], List[object]
   - **Custom**: u_CompanyNameDataType with schemas

### What We Got Wrong
- **Architecture**: Treated resolvers as slot config instead of separate entities
- **Method Selection**: Missing AI agent intelligent method picking
- **Data Types**: Only implemented 3 basic types instead of 6 primitives + List variants + custom types
- **Multiple Methods**: Thought 1 strategy = 1 method, actually 1 strategy = multiple methods

## Implementation Status
- ✅ **UI Framework**: Professional slot configuration interface
- ✅ **Basic Slots**: Name, data type, description configuration
- ✅ **Basic HTTP Actions**: Create/edit HTTP actions with UI
- ❌ **Resolver Architecture**: Needs complete rebuild for multiple methods
- ❌ **AI Method Selection**: Missing intelligent method picking logic
- ❌ **Data Type System**: Missing comprehensive type support
- ❌ **Action Types**: Missing Script Actions, Compound Actions, Built-in Actions
- ❌ **Runtime Execution**: Missing method execution engine and data bank

## Next Steps
1. **Redesign resolver strategy architecture** for multiple methods
2. **Implement AI agent method selection** logic
3. **Expand data type system** to include all 6 primitives + List variants + custom types
4. **Complete action system**: Script Actions, Compound Actions, Built-in Actions
5. **Build runtime execution engine** with data bank and action chaining

## Quick Reference

### Slot Configuration
```yaml
Slot:
  Name: "slot_name"
  Data Type: "string" | "number" | "boolean" | "User" | "List[type]" | "custom_type"
  Slot Description: "Description for AI inference"
  Slot Validation Policy: "DSL validation rule"
  Slot Validation Description: "Error message for validation failure"
  Slot Inference Policy: "Infer slot value if available" | "Always explicitly ask for slot"
  Resolver Strategy: See resolver-strategies/README.md
```

### Resolver Strategy Configuration
```yaml
Resolver Strategy:
  Method Name: "resolver_method_name"
  Method Type: "Static" | "API" | "Inline"
  # For Static type:
  Static Options:
    - Display Value: "User-friendly name"
      Raw Value: "system_value"
```

## Data Sources

All documentation is based on official Moveworks documentation from:
- https://help.moveworks.com/docs/slots
- https://help.moveworks.com/docs/resolver-strategies
- https://help.moveworks.com/docs/data-types
- https://help.moveworks.com/docs/quickstart-basic-task-agent
- https://help.moveworks.com/docs/quickstart-2-slots-resolvers-copy

## Implementation Notes

This documentation reflects the **actual** Moveworks behavior as documented in their official help documentation, not assumptions or interpretations. Each feature includes:

1. **Official Definition** - Exact behavior from Moveworks docs
2. **Configuration Format** - Precise configuration syntax
3. **Examples** - Real examples from Moveworks documentation
4. **Implementation Status** - Current status in our system
5. **Missing Features** - What needs to be implemented

## Contributing

When updating this documentation:
1. Always reference official Moveworks documentation
2. Include source URLs for all information
3. Mark implementation status clearly
4. Update cross-references between documents

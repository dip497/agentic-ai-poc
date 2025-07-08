# Moveworks Slots - Complete Documentation

## Overview

Slots are the fundamental data collection mechanism in Moveworks conversational processes. They define what information the AI Assistant needs to collect from users and how that information should be processed.

**Source**: [Moveworks Slots Documentation](https://help.moveworks.com/docs/slots)

## Slot Configuration Structure

```yaml
Slot Configuration:
  Name: string                    # Unique identifier for the slot
  Data Type: string              # See data-types/README.md
  Slot Description: string       # Guides AI inference behavior
  Slot Validation Policy: string # DSL validation rule (optional)
  Slot Validation Description: string # Error message for validation failure
  Slot Inference Policy: enum    # AI inference behavior
  Resolver Strategy: object      # See resolver-strategies/README.md
```

## Core Properties

### 1. Slot Name
- **Purpose**: Unique identifier for the slot within the process
- **Format**: String, typically snake_case
- **Examples**: `pto_type`, `feature_request`, `target_status`

### 2. Data Type
- **Purpose**: Defines the type of data the slot collects
- **Options**: See [Data Types Documentation](../data-types/README.md)
- **Examples**: `string`, `number`, `boolean`, `User`, `List[string]`

### 3. Slot Description
- **Purpose**: Provides context to the AI Assistant for inference and collection
- **Critical Role**: Controls AI behavior and user interaction
- **Examples**:
  ```yaml
  # Basic description
  Slot Description: "The type of PTO balance the user is fetching for."
  
  # With formatting instructions
  Slot Description: "Timestamp should follow ALWAYS the following format: YYYY-MM-DDT23:59:59."
  
  # With default value instructions
  Slot Description: "If the user doesn't specify start_date, set the start_date as the current date & time."
  
  # With disclaimers
  Slot Description: "Present user a disclaimer that start_date should NOT be in the past"
  
  # With inference control
  Slot Description: "Do NOT infer the country_code from user's profile. Always ask the user explicitly for it"
  ```

## Slot Inference Policies

**Source**: [Moveworks Slots - Inference Policies](https://help.moveworks.com/docs/slots)

### Available Options

1. **"Infer slot value if available"** (Default)
   - AI Assistant attempts to infer the value from conversation context
   - Falls back to asking user if inference fails
   - Most common setting

2. **"Always explicitly ask for slot"**
   - AI Assistant ALWAYS asks the user for the value
   - Never attempts inference from context
   - Use for sensitive or critical data

### Implementation Status
- ✅ **UI Support**: Dropdown with both options
- ✅ **Backend Storage**: Saved in slot configuration
- ❌ **Runtime Behavior**: Not implemented in AI processing

## Slot Validation

**Source**: [Moveworks Slots - Validation](https://help.moveworks.com/docs/slots)

### Validation Policy
- **Format**: Moveworks DSL expression
- **Purpose**: Ensures slot values meet specific criteria
- **Examples**:
  ```yaml
  # Numeric validation
  Slot Validation Policy: "value > 0"
  
  # Date validation
  Slot Validation Policy: "$PARSE_TIME(value) > $TIME()"
  
  # String validation
  Slot Validation Policy: "$LENGTH(value) > 5"
  ```

### Validation Description
- **Purpose**: Error message shown when validation fails
- **Examples**:
  ```yaml
  Slot Validation Description: "The quantity for the item in the purchase request must be greater than zero"
  Slot Validation Description: "The due date cannot be in the past"
  ```

### Implementation Status
- ✅ **UI Support**: Text input fields for policy and description
- ✅ **Backend Storage**: Saved in slot configuration
- ❌ **Runtime Validation**: Not implemented in processing

## Slot Cardinality

**Source**: [Moveworks Slots - Cardinality](https://help.moveworks.com/docs/slots)

### Single vs List Values

```yaml
# Single value
Data Type: "string"        # Collects one string
Data Type: "number"        # Collects one number
Data Type: "User"          # Collects one user

# List values
Data Type: "List[string]"  # Collects array of strings
Data Type: "List[number]"  # Collects array of numbers
Data Type: "List[User]"    # Collects array of users
```

### Implementation Status
- ✅ **UI Support**: Dropdown includes List[type] options
- ✅ **Backend Storage**: Saved in data type field
- ❌ **Runtime Collection**: Not implemented in processing

## Real-World Examples

### Example 1: PTO Type Slot
```yaml
Slot Configuration:
  Name: "pto_type"
  Data Type: "string"
  Slot Description: "The type of PTO balance the user is fetching for."
  Slot Validation Policy: ""
  Slot Validation Description: ""
  Slot Inference Policy: "Infer slot value if available"
  Resolver Strategy:
    Method Name: "choose_from_existing_pto_balance_types"
    Method Type: "Static"
    Static Options:
      - Display Value: "Vacation"
        Raw Value: "vacation"
      - Display Value: "Sick"
        Raw Value: "sick"
```

### Example 2: Quantity Slot with Validation
```yaml
Slot Configuration:
  Name: "quantity"
  Data Type: "number"
  Slot Description: "The number of units of the item that the user wants to purchase."
  Slot Validation Policy: "value > 0"
  Slot Validation Description: "The quantity for the item in the purchase request must be greater than zero"
  Slot Inference Policy: "Infer slot value if available"
```

### Example 3: Feature Request Slot with Custom Data Type
```yaml
Slot Configuration:
  Name: "feature_request"
  Data Type: "u_FirstnameLastnameFeatureRequest"
  Slot Description: "The feature request that the user wants to update."
  Slot Validation Policy: ""
  Slot Validation Description: ""
  Slot Inference Policy: "Infer slot value if available"
  Resolver Strategy:
    Method Name: "choose_from_existing_feature_requests"
    Method Type: "API"  # Would resolve to actual feature requests
```

## Related Documentation

- [Resolver Strategies](../resolver-strategies/README.md) - How slots resolve values
- [Data Types](../data-types/README.md) - Available data types for slots
- [Slot Inference Policies](./inference-policies.md) - Detailed inference behavior
- [Slot Validation](./validation.md) - Validation rules and DSL
- [Slot Cardinality](./cardinality.md) - Single vs List handling

## Implementation Gaps

### Critical Missing Features
1. **Runtime Inference**: AI doesn't respect inference policies
2. **Runtime Validation**: Validation rules not enforced
3. **Cardinality Handling**: List types not properly collected
4. **Custom Data Types**: Not supported in slot configuration

### Next Steps
1. Implement slot validation engine using Moveworks DSL
2. Add cardinality support to data collection
3. Implement inference policy behavior in AI processing
4. Add custom data type support

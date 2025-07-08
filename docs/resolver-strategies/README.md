# Moveworks Resolver Strategies

## What Are Resolver Strategies?

Resolver strategies convert user input (e.g., "this week's most important bug") to specific data types (e.g., `<JiraIssue>_BUG-732`).

**Key Principle**: One Strategy → One Data Type

## Architecture

### Resolver Strategy = Collection of Methods

```yaml
ResolverStrategy: "JiraIssueResolver"
  Data Type: "JiraIssue"
  Methods:
    - "get_jira_issue_by_id"           # For "update BUG-732"
    - "get_user_assigned_issues"       # For "update my tasks"
    - "get_issues_by_status_project"   # For "update in-progress Project Orion tasks"
```

### AI Agent Method Selection

The AI agent **automatically picks** the best method based on user input:

| User Says | AI Selects | Why |
|-----------|------------|-----|
| "Update BUG-732" | `get_jira_issue_by_id` | Specific ID provided |
| "Update my tasks" | `get_user_assigned_issues` | User wants their tasks |
| "Update Project Orion in-progress tasks" | `get_issues_by_status_project` | Status + project criteria |

## Method Types

### Static Methods
- **Purpose**: Predefined list of options
- **Limitation**: Only **1 static method** per strategy
- **Example**: Status options like "New", "In Progress", "Done"

### Dynamic Methods
- **Purpose**: Fetch live data from APIs
- **Limitation**: Can have **multiple dynamic methods**
- **Example**: Get user's assigned tickets from Jira API

### 🚧 Critical Rule
**Cannot mix**: 1 Static Method **OR** Multiple Dynamic Methods (never both)

## Configuration Examples

### Static Method Configuration
```yaml
Method Name: "choose_from_pto_types"
Method Type: "Static"
Static Options:
  - Display Value: "Vacation"
    Raw Value: "vacation"
  - Display Value: "Sick Leave"
    Raw Value: "sick"
```

### Dynamic Method Configuration
```yaml
Method Name: "get_user_assigned_issues"
Method Type: "Dynamic"
Action: "fetch_jira_issues_http_action"
Input Arguments:
  user_id: "meta_info.user.id"
  status: "data.status_filter"
Output Mapping: ".issues"
```

## Data Bank Access

Methods can access:
- **Input Arguments**: `data.feature_request_id`
- **User Context**: `meta_info.user.email_addr`
- **User Attributes**: `meta_info.user.department`

## Data Types

Data types specify the structure & kind of values a variable can hold. AI agents are "type-aware" and use data types to determine how to collect them (slots), pass them between plugins, or display them in citations.

### Primitive Data Types
"Simple" values provided by most programming languages:

```yaml
string: text
integer: valid integers
number: floating point numbers
boolean: true / false
```

### Object Data Types
"Complex" values that correspond to data objects from business systems:

```yaml
# Built-in Object Types
User: A Moveworks user (built-in resolver strategy)

# Custom Object Types (must follow u_<DataTypeName> convention)
u_SalesforceAccount: An Account object in Salesforce
u_JiraIssue: An Issue object in Jira
u_FirstnameLastnameFeatureRequest: Custom feature request type
```

### 🚧 Important Rules
- **Do not merge data types across systems**: If you have Accounts in both Salesforce & Netsuite, create separate data types
- **Naming convention**: Custom data types must follow `u_<DataTypeName>` format
- **One system per type**: Each data type corresponds to objects from a single business system

### List/Array Types
Any data type can be configured as a list:

```yaml
List[string]: array of strings
List[integer]: array of whole numbers
List[number]: array of decimal numbers
List[boolean]: array of booleans
List[User]: array of User objects
List[u_JiraIssue]: array of Jira issues
```

### Custom Data Type Creation
```yaml
# Example: Feature request data type
Name: "u_FirstnameLastnameFeatureRequest"
Description: "Feature request object with status tracking"
Schema:
  type: object
  properties:
    id: string
    name: string
    current_status: string
    created_by: string
    moderator: string
    product_area: string

# Auto-generated from JSON example:
{
  "id": "FR-12345",
  "name": "Add Dark Mode",
  "current_status": "New",
  "created_by": "user@company.com",
  "moderator": "pm@company.com",
  "product_area": "UI/UX"
}
```

### Default Resolver Strategies
Data types can have default resolver strategies so you don't repeat configuration for every plugin:

```yaml
Data Type: "u_JiraIssue"
Default Resolver Strategy: "JiraIssueResolver"
  Methods:
    - "get_jira_issue_by_id"
    - "get_user_assigned_issues"
    - "get_issues_by_criteria"
```

## Output Formats

### Single Record
```json
{
  "id": "BUG-732",
  "title": "Fix login issue",
  "status": "Open"
}
```

### List of Records
```json
[
  {"id": "BUG-732", "title": "Fix login issue"},
  {"id": "BUG-733", "title": "Add dark mode"}
]
```

## Configuration Approaches

### 1. Data Type Level (Reusable)
Configure resolver strategy on the data type - any slot using that data type inherits the resolver.

### 2. Slot Level (Specific)
Configure resolver strategy directly on the slot - only that slot uses it.

## Data Bank Access

**Source**: [Moveworks Resolver Data Bank](https://help.moveworks.com/docs/resolver-strategies)

### Available Context
```yaml
Resolver Context Data Access:
  data.<input_arg>:
    Description: "References input arguments passed to the resolver method"
    Usage: "data.feature_request_id"
  meta_info.user:
    Description: "References user attributes about the current user"
    Examples:
      - "meta_info.user.email_addr"
      - "meta_info.user.department"
      - "meta_info.user.user_tags"
```

### Input Arguments Schema
```json
{
  "type": "object",
  "properties": {
    "feature_request_id": {
      "type": "string"
    },
    "feature_request_name": {
      "type": "string"
    }
  }
}
```

## Output Formats

### Single Record Output
```json
{
  "id": "BUG-732",
  "title": "Set HTTP Status code correctly on custom REST API",
  "status": "Open",
  "assignee": "john.doe@company.com"
}
```

### List of Records Output
```json
[
  {
    "id": "BUG-732",
    "title": "Set HTTP Status code correctly on custom REST API"
  },
  {
    "id": "BUG-733",
    "title": "Envelope response under 'data' keyword"
  }
]
```

```

### Correct Implementation (Target)
```typescript
// Complete data type system
type PrimitiveDataType = "string" | "integer" | "number" | "boolean";
type ObjectDataType = "User" | string; // string for custom types like "u_JiraIssue"
type ListDataType = `List[${PrimitiveDataType | ObjectDataType}]`;
type DataType = PrimitiveDataType | ObjectDataType | ListDataType;

// Custom data type with schema
interface CustomDataType {
  name: string;                 // Must follow u_<DataTypeName> convention
  description: string;
  schema: JSONSchema;
  default_resolver_strategy?: string;
}

// Resolver strategy with multiple methods
interface ResolverStrategy {
  name: string;
  data_type: DataType;
  methods: ResolverMethod[];    // Multiple methods!
}

interface ResolverMethod {
  name: string;                 // Must be snake_case
  type: "Static" | "Dynamic";
  action?: string;              // For dynamic methods
  static_options?: Option[];    // For static methods
  input_arguments?: JSONSchema;
}

// Slot only references data type (inherits resolver) or specific resolver
interface Slot {
  name: string;
  data_type: DataType;
  resolver_strategy?: {
    name: string;               // Optional: override default resolver
  };
}
```

## Implementation Plan

### Phase 1: Architecture
1. Create `ResolverStrategy` entity with multiple methods
2. Separate resolver strategies from slot configuration
3. Implement method type validation (1 static OR multiple dynamic)

### Phase 2: Method Selection
1. Build AI agent method selection logic
2. Implement context-based method picking
3. Add method execution engine

### Phase 3: Data Types
1. **Primitive Data Types**: string, integer, number, boolean
2. **Object Data Types**:
   - Built-in: User (with built-in resolver)
   - Custom: u_<DataTypeName> with JSON schema validation
3. **List Support**: List[any_type] for arrays
4. **Custom Data Type Creation**:
   - Name validation (u_<DataTypeName> convention)
   - JSON schema import and validation
   - Default resolver strategy assignment
5. **Data Type Management UI**:
   - Create/edit custom data types
   - Import JSON examples for schema generation
   - Assign default resolver strategies

### Phase 4: Integration
1. Update slot configuration UI
2. Add resolver strategy management UI
3. Implement runtime method execution

## Sources
- [Moveworks Resolver Strategies](https://help.moveworks.com/docs/resolver-strategies)
- [Moveworks Slots](https://help.moveworks.com/docs/slots)
- [Moveworks Data Types](https://help.moveworks.com/docs/data-types)

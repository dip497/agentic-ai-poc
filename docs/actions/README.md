# Moveworks Actions

## What Are Actions?

When your plugin executes, it performs various actions to retrieve & manipulate data.

**Examples:**
- Create a Time Off Request in Workday
- Retrieve an account's current account manager from Salesforce
- Execute a prompt to generate a mermaid diagram
- Send a notification to a user

## Types of Actions

### 1. Built-in Actions
Moveworks supports a large number of built-in actions for common AI agent operations:

```yaml
Built-in Actions:
  - mw.generate_text_action: Generate text using LLM
  - mw.generate_structured_value_action: Generate structured data
  - mw.send_plaintext_chat_notification: Send chat messages
  - mw.create_generic_approval_request: Create approval workflows
  - mw.get_user_by_email: Get user from identity store
  - mw.batch_get_users_by_email: Get multiple users
```

### 2. HTTP Actions
Moveworks can make HTTP calls to business systems like Workday, SAP, Salesforce, & ServiceNow:

```yaml
HTTP Action: "get_jira_issues"
  Method: GET
  Endpoint: "/rest/api/2/search"
  Headers:
    Authorization: "Bearer {{token}}"
    Content-Type: "application/json"
  Query Parameters:
    jql: "assignee={{user_email}}"
  Input Arguments:
    - user_email: string
  Response: List of Jira issues
```

### 3. Script Actions
If you're a professional developer, you can write a script in your favorite language:

```yaml
Script Action: "calculate_approval_level"
  Input Arguments:
    - purchase_amount: number
    - user_department: string
  Code: |
    if purchase_amount < 1000:
        return "manager"
    elif purchase_amount < 10000:
        return "director"
    else:
        return "vp"
  Output: Approval level string
```

### 4. Compound Actions
You can combine any of the above into a compound action, which adds control flow, progress updates, and more:

```yaml
Compound Action: "submit_purchase_request"
  Steps:
    - Action: "validate_purchase_amount"
      Output Key: "validation_result"
    - Action: "get_approval_level"
      Input: validation_result.amount
      Output Key: "approval_level"
    - Action: "create_approval_request"
      Input: approval_level
      Output Key: "approval_request"
  Return:
    request_id: approval_request.id
    status: "pending_approval"
```

## How to Use Actions in a Plugin

You'll need to add your actions to a **Compound Action** first. Then promote the Compound Action to a plugin from the Compound Action library.

**Workflow:**
1. Create individual actions (HTTP, Script, or use Built-in)
2. Combine them into a Compound Action
3. Promote the Compound Action to a plugin

## Action Configuration

### HTTP Action Fields
```yaml
HTTP Action:
  action_name: string (unique identifier)
  method: GET | POST | PUT | DELETE | PATCH
  endpoint: string (starts with '/')
  headers: key-value pairs
  query_parameters: key-value pairs  
  body: string (for POST/PUT requests)
  input_arguments: list of typed arguments
  output_mapping: JSONPath or transformation
```

### Script Action Fields
```yaml
Script Action:
  action_name: string (unique identifier)
  code: string (Python code)
  input_args: dictionary of input variables
  output_key: string (variable to store result)
```

### Compound Action Fields
```yaml
Compound Action:
  action_name: string (unique identifier)
  steps: list of action calls
  input_arguments: list of typed arguments
  return: output mapping
```

## Data Flow in Actions

### Input Arguments
```yaml
Input Arguments:
  - name: "user_email"
    type: "string"
    required: true
    example: "user@company.com"
  - name: "ticket_priority"
    type: "integer"
    required: false
    default: 3
```

### Output Keys and Data Bank
```yaml
# Action execution creates data in the data bank
Action 1:
  output_key: "user_info"
  result: {"id": "123", "name": "John Doe"}

# Subsequent actions can reference previous outputs
Action 2:
  input_args:
    user_id: "data.user_info.id"  # References Action 1 output
    user_name: "data.user_info.name"
```

### Progress Updates
```yaml
Action:
  progress_updates:
    on_pending: "Searching for user {{data.user_email}}..."
    on_complete: "Found user {{data.user_info.name}}"
```

## Advanced Action Features

### Error Handling with try_catch
```yaml
try_catch:
  try:
    steps:
      - action:
          action_name: "risky_api_call"
          output_key: "api_result"
  catch:
    on_status_code: [400, 500]
    steps:
      - action:
          action_name: "send_error_notification"
          input_args:
            error: "error_data.api_result"
```

### Conditional Logic with switch
```yaml
switch:
  cases:
    - condition: "data.user.role == 'admin'"
      steps:
        - action:
            action_name: "grant_admin_access"
    - condition: "data.user.role == 'manager'"
      steps:
        - action:
            action_name: "grant_manager_access"
  default:
    steps:
      - action:
          action_name: "grant_user_access"
```

### Parallel Execution
```yaml
parallel:
  branches:
    - steps:
        - action:
            action_name: "log_event"
    - steps:
        - action:
            action_name: "send_notification"
```

### For Loops
```yaml
for:
  in: "data.users"
  each: "user"
  output_key: "notification_results"
  steps:
    - action:
        action_name: "send_welcome_email"
        input_args:
          user_email: "user.email"
```

## Action Integration with Resolver Strategies

Actions are used in **Dynamic Resolver Methods**:

```yaml
Resolver Strategy: "JiraIssueResolver"
  Methods:
    - Method: "get_user_assigned_issues"
      Type: "Dynamic"
      Action: "fetch_jira_issues_http_action"  # References HTTP Action
      Input Arguments:
        user_id: "meta_info.user.id"
      Output: List of JiraIssue objects
```

## Built-in Actions Reference

### Text Generation
```yaml
mw.generate_text_action:
  Input:
    system_prompt: string (optional)
    user_input: string (required)
    model: string (optional, defaults to 4o-mini)
  Output:
    openai_chat_completions_response.choices[0].message.content
```

### Structured Data Generation
```yaml
mw.generate_structured_value_action:
  Input:
    payload: object (required)
    output_schema: object (required)
    system_prompt: string (optional)
    strict: boolean (optional)
  Output:
    Structured data matching schema
```

### User Management
```yaml
mw.get_user_by_email:
  Input:
    user_email: string
  Output:
    User object from identity store

mw.batch_get_users_by_email:
  Input:
    user_emails: List[string]
  Output:
    List of user records
```

## Key Insights from Official Documentation

### 🎯 **Action Hierarchy**
1. **Individual Actions**: HTTP, Script, Built-in actions
2. **Compound Actions**: Combine multiple actions with control flow
3. **Plugins**: Promoted from Compound Actions

### 🔗 **Integration with Plugins**
- Actions must be wrapped in **Compound Actions** first
- Compound Actions are then **promoted to plugins**
- This creates the complete plugin workflow

### 📊 **Action Examples from Moveworks**
- **Time Off Request**: Create in Workday
- **Account Management**: Retrieve from Salesforce
- **AI Generation**: Generate mermaid diagrams
- **Notifications**: Send user messages

## What We Need to Implement

### Current Status
- ✅ **Basic HTTP Action UI**: Create/edit HTTP actions
- ✅ **Action Activity Integration**: Use actions in processes
- ❌ **Script Actions**: Python code execution environment
- ❌ **Compound Actions**: Multi-step orchestration with control flow
- ❌ **Built-in Actions**: Moveworks-provided actions integration
- ❌ **Advanced Features**: try_catch, switch, parallel, for loops
- ❌ **Data Bank**: Action output chaining system
- ❌ **Progress Updates**: User feedback during execution
- ❌ **Plugin Promotion**: Convert Compound Actions to plugins

### Implementation Plan

#### Phase 1: Core Actions
1. **Complete HTTP Actions**: All methods, headers, query params, body
2. **Script Actions**: Python/APIthon code execution environment
3. **Action Testing**: Test individual actions with sample data

#### Phase 2: Compound Actions
1. **Multi-step orchestration**: Chain actions with steps
2. **Data bank**: Store and reference action outputs (`data.action_output`)
3. **Progress updates**: User feedback system (`on_pending`, `on_complete`)
4. **Control flow**: Sequential execution with output chaining

#### Phase 3: Advanced Features
1. **Error handling**: try_catch blocks with status code filtering
2. **Conditional logic**: switch statements with cases and default
3. **Parallel execution**: Concurrent action execution with branches
4. **Loops**: For loop processing over collections

#### Phase 4: Built-in Actions & Plugin Integration
1. **Built-in Actions**: LLM generation, user management, notifications
2. **Plugin Promotion**: Convert Compound Actions to plugins
3. **Complete Integration**: Actions → Compound Actions → Plugins workflow

## Sources
- [Actions Overview](https://help.moveworks.com/docs/actions)
- [HTTP Actions](https://help.moveworks.com/docs/http-actions)
- [Script Actions](https://help.moveworks.com/docs/script-actions)
- [Built-in Actions](https://help.moveworks.com/docs/built-in-actions)
- [Compound Actions](https://help.moveworks.com/docs/compound-actions)

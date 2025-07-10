# Moveworks-Style Conversational AI System

A complete replication of Moveworks' conversational AI architecture using LangChain, built from their official documentation. This system implements **exact Moveworks patterns** including Conversational Processes, Activities, Decision Policies, and LLM-powered slot inference.

## 🎯 **Moveworks Architecture Replication**

This system replicates Moveworks' exact architecture based on their documentation:

### **Core Components**
- ✅ **Conversational Processes** (not intents) - the main logic containers
- ✅ **Activities** - Action, Content, and Decision types
- ✅ **LLM-Powered Slot Inference** - using slot descriptions to guide AI behavior
- ✅ **Decision Policies** - control when Activities run based on conditions
- ✅ **Built-in LLM Actions** - equivalent to `mw.generate_text_action`, etc.
- ✅ **Semantic Process Matching** - LLM-based trigger utterance matching
- ✅ **DSL Integration** - for data mapping and business rules

### **Key Differences from Rasa/Traditional Approaches**
- **No NLU Pipeline** - uses LLM for understanding throughout
- **No Intent Classification** - uses semantic process matching instead
- **No Entity Extraction** - uses LLM slot inference with policies
- **No Training Examples** - uses natural trigger utterances
- **LLM-First** - every decision uses LLM understanding

## 🚀 **Quick Start**

### **1. Installation**
```bash
git clone <repository-url>
cd moveworks-ai-system
pip install -r requirements.txt
```

### **2. Configure LLM Provider**

The system supports multiple LLM providers. Choose one:

#### **Option A: OpenAI (Recommended)**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### **Option B: Google Gemini (Cost-effective)**
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

#### **Option C: OpenRouter (Multiple Models)**
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

#### **Option D: Anthropic Claude**
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

#### **Option E: Local Ollama (Free)**
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.1
# No API key needed!
```

#### **Quick Setup Tool**
```bash
python setup_llm.py
```

### **3. Run Interactive Demo**
```bash
python main.py interactive
```

### **4. Run Example Scenarios**
```bash
python main.py demo
```

### **5. Test Process Matching**
```bash
python main.py test
```

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│ Process Matcher │───▶│ Matched Process │
└─────────────────┘    │   (LLM-based)   │    │   & Activities  │
                       └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Response Gen.   │◀───│Activity Executor│◀───│ Slot Inference  │
└─────────────────┘    │ (Action/Content/│    │   (LLM-based)   │
                       │    Decision)    │    └─────────────────┘
                       └─────────────────┘
```

## 📋 **Example Use Cases**

### **1. PTO Balance Request**
```
User: "What's my vacation balance?"
→ Process: get_pto_balance_process
→ Slot Inference: pto_type = "vacation" (inferred from "vacation")
→ Action: get_pto_balance_action
→ Response: "Your vacation balance is 15 days available..."
```

### **2. Feature Request Update**
```
User: "Update FR-123 to completed"
→ Process: update_feature_request_process
→ Slot Inference: feature_request = FR-123, target_status = "completed"
→ Action: update_feature_request_action
→ Response: "I've successfully updated feature request FR-123..."
```

### **3. AI-Powered Purchase Classification**
```
User: "I need to buy a laptop"
→ Process: support_procurement_purchases_process
→ AI Classification: "laptop" → "Capex" (using historical data)
→ Decision Policy: Route to CapEx portal
→ Response: "This falls under CapEx. Please visit..."
```

## ⚙️ **Configuration (Moveworks Style)**

### **LLM Provider Configuration**

Configure your preferred LLM provider in `config/moveworks_config.yml`:

```yaml
# OpenAI Configuration
llm:
  default:
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.1
    max_tokens: 2000

# Google Gemini Configuration
llm:
  default:
    provider: "gemini"
    model: "gemini-1.5-flash"
    temperature: 0.1
    max_tokens: 2000

# OpenRouter Configuration (Claude via OpenRouter)
llm:
  default:
    provider: "openrouter"
    model: "anthropic/claude-3.5-sonnet"
    temperature: 0.1
    max_tokens: 2000
    provider_config:
      app_name: "Moveworks AI System"

# Local Ollama Configuration
llm:
  default:
    provider: "ollama"
    model: "llama3.1"
    temperature: 0.1
    max_tokens: 2000
    provider_config:
      base_url: "http://localhost:11434"
```

### **Conversational Process Configuration**

The system uses Moveworks-style YAML configuration:

```yaml
plugins:
  - name: "pto_management_plugin"
    conversational_processes:
      - title: "get_pto_balance_process"
        description: "Fetches the PTO balance for a given user"
        trigger_utterances:
          - "How can I check my current paid time off balance?"
          - "What's my vacation balance?"
        slots:
          - name: "pto_type"
            data_type: "string"
            slot_description: "Type of PTO balance to fetch"
            slot_inference_policy: "Infer slot value if available"
            resolver_strategy:
              method_type: "Static"
              static_options:
                - display_value: "Vacation"
                  raw_value: "vacation"
        activities:
          - activity_type: "action"
            action_name: "get_pto_balance_action"
            required_slots: ["pto_type"]
            input_mapping:
              pto_type: "data.pto_type.value"
              user_email: "meta_info.user.email_addr"
```

## 🧠 **Multi-LLM Integration (LangChain)**

### **Supported LLM Providers**

| Provider | Models | Cost | Setup |
|----------|--------|------|-------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-3.5 | $$$ | `export OPENAI_API_KEY=...` |
| **Google Gemini** | Gemini-1.5-Pro, Gemini-1.5-Flash | $$ | `export GOOGLE_API_KEY=...` |
| **OpenRouter** | Claude, Llama, Mixtral, etc. | $$ | `export OPENROUTER_API_KEY=...` |
| **Anthropic** | Claude-3.5-Sonnet, Claude-3-Haiku | $$$ | `export ANTHROPIC_API_KEY=...` |
| **Ollama** | Llama3.1, Mistral, CodeLlama | FREE | Local installation |

### **LLM Configuration**
```yaml
# config/moveworks_config.yml
llm:
  default:
    provider: "gemini"              # or openai, openrouter, anthropic, ollama
    model: "gemini-1.5-flash"       # fast and cost-effective
    temperature: 0.1
    max_tokens: 2000

  process_matching:
    provider: "openrouter"
    model: "anthropic/claude-3.5-sonnet"  # excellent reasoning
    temperature: 0.1
    max_tokens: 1000
```

### **Process Matching**
```python
# Uses any LLM to semantically match user utterances to processes
matcher = MoveworksProcessMatcher(llm_config={
    "provider": "gemini",
    "model": "gemini-1.5-flash"
})
result = await matcher.match_process(
    "What's my vacation balance?",
    available_processes
)
```

### **Slot Inference**
```python
# Uses any LLM with slot descriptions to infer values
inference = MoveworksSlotInference(llm_config={
    "provider": "openrouter",
    "model": "anthropic/claude-3.5-sonnet"
})
result = await inference.infer_slot_value(
    slot=pto_type_slot,
    user_utterance="What's my vacation balance?",
    context=conversation_context
)
```

### **Built-in LLM Actions**
```python
# Works with any configured LLM provider
result = await builtin_actions.generate_text_action(
    GenerateTextActionInput(
        system_prompt="Classify as Opex or Capex",
        user_input="laptop purchase",
        model="gemini-1.5-flash"  # or any supported model
    )
)
```

## 📊 **Moveworks Features Implemented**

### **✅ Conversational Processes**
- Title and Description
- Trigger Utterances (natural language)
- Slots with inference policies
- Activities (Action, Content, Decision)
- Decision Policies for flow control

### **✅ Slot System**
- **Inference Policies**: "Infer slot value if available" vs "Always explicitly ask for slot"
- **Slot Descriptions**: Guide LLM behavior and understanding
- **Validation Policies**: DSL expressions for validation
- **Resolver Strategies**: Static, Dynamic, Vector Search, Custom

### **✅ Activity Types**
- **Action Activities**: Execute actions with input/output mapping
- **Content Activities**: Display content with template variables
- **Decision Activities**: Route based on DSL conditions

### **✅ Built-in Actions**
- `generate_text_action` - LLM text generation
- `generate_structured_value_action` - Structured LLM output
- `get_user_by_email` - User lookup
- `send_chat_notification` - Notifications

### **✅ DSL Integration**
- Data mapping: `data.slot_name.value`
- Meta references: `meta_info.user.email_addr`
- Function calls: `$CONCAT`, `$LOWERCASE`
- Conditions: `data.classification.$LOWERCASE() == "opex"`

## 🔧 **Advanced Features**

### **Semantic Process Matching**
Uses LLM to match user utterances to processes based on semantic similarity rather than pattern matching.

### **Context-Aware Slot Inference**
LLM considers conversation history, user attributes, and slot descriptions to intelligently infer values.

### **Dynamic Activity Routing**
Decision Activities use DSL expressions to dynamically route conversation flow.

### **Confirmation Policies**
Activities can require user consent before execution.

## 🧪 **Testing**

### **Interactive Testing**
```bash
python main.py interactive
```

### **Scenario Testing**
```bash
python main.py demo
```

### **Process Matching Testing**
```bash
python main.py test
```

## 📈 **System Statistics**

The system provides real-time statistics:
- Active conversations
- Registered plugins and processes
- Process matching performance
- Slot inference accuracy

## 🔮 **Moveworks Patterns Demonstrated**

1. **LLM-First Architecture** - Every understanding task uses LLM
2. **Semantic Understanding** - No pattern matching, all semantic
3. **Conversational Processes** - Not intents, but full process flows
4. **Activity Orchestration** - Flexible activity execution
5. **Slot Inference Policies** - AI-guided slot filling
6. **DSL Integration** - Business logic in expressions
7. **Built-in LLM Actions** - Core AI capabilities as actions

## 🎯 **Key Benefits**

- **More Natural**: Understands variations and context better than rule-based systems
- **More Flexible**: Easy to add new processes without retraining
- **More Intelligent**: LLM understanding at every step
- **More Scalable**: Configuration-driven, not code-driven
- **More Maintainable**: Clear separation of concerns

## 🚀 **Production Considerations**

- **LLM Costs**: Monitor token usage for cost optimization
- **Latency**: Consider caching for frequently matched processes
- **Reliability**: Ensure robust error handling and proper failure modes
- **Security**: Validate all LLM outputs before execution
- **Monitoring**: Track conversation success rates and user satisfaction

This system demonstrates how modern conversational AI can be built using LLM-first approaches, following Moveworks' proven patterns for enterprise AI assistants.

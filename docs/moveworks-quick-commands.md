# Moveworks Project - Quick Commands Reference

## 🚀 **COPY-PASTE COMMANDS**

### **📊 Status & Planning**

```
Check the Moveworks project status from the tracker documents and show me:
1. Current progress (X/12 action items completed)
2. Next highest priority item to work on
3. Any blockers or dependencies
4. Estimated effort for next item
```

```
Review the Moveworks implementation roadmap and create a detailed plan for the next 2 weeks of work, including:
1. Specific action items to tackle
2. Implementation order and dependencies  
3. Testing and validation approach
4. Integration points with existing code
```

### **🔧 Implementation Commands**

```
Work on the Moveworks project - implement the Manifest Generator:
1. Check current status from tracker documents
2. Analyze existing plugin selection code
3. Implement ManifestGenerator class following Moveworks patterns
4. Include autonomous plugin selection, intent analysis, and competition resolution
5. Write comprehensive tests
6. Update tracker documents with progress
```

```
Work on the Moveworks project - implement the Slot Resolver System:
1. Check dependencies and current status
2. Create SlotResolverEngine with Static, API, and Inline resolvers
3. Follow Moveworks resolver strategy patterns
4. Integrate with existing data types and plugin system
5. Write tests for all resolver types
6. Update progress in tracker documents
```

```
Work on the Moveworks project - implement the Policy Validator Engine:
1. Review current status and requirements
2. Create PolicyValidator class with compliance checking
3. Add safety guardrails and business policy validation
4. Integrate with reasoning loops for deterministic checking
5. Write comprehensive tests
6. Update tracker with completed work
```

### **🔄 Progress & Updates**

```
Update the Moveworks project tracker documents with completed work:
1. Mark completed action items as done
2. Update progress percentages
3. Note any new issues or blockers discovered
4. Identify next priority items
5. Update effort estimates if needed
```

```
Review the Moveworks implementation progress and provide:
1. Summary of completed vs. remaining work
2. Current phase status (Phase 1/2/3)
3. Timeline assessment - are we on track?
4. Any adjustments needed to the roadmap
5. Recommendations for next steps
```

### **🧪 Testing & Validation**

```
Test the Moveworks implementation components:
1. Run tests for completed components
2. Validate integration with existing system
3. Check Moveworks pattern compliance
4. Test end-to-end reasoning flow
5. Report any issues found
```

```
Validate the Moveworks implementation against official patterns:
1. Compare our implementation with Moveworks documentation
2. Check for any deviations from official patterns
3. Verify all required components are implemented correctly
4. Suggest improvements for better alignment
```

### **🔗 Integration Commands**

```
Check integration points for the Moveworks components:
1. Verify compatibility with existing LLM factory
2. Check memory constructs integration
3. Validate LangGraph state management
4. Test database schema compatibility
5. Ensure UUID handling works correctly
```

```
Integrate the new Moveworks components with the existing reasoning engine:
1. Update MoveworksThreeLoopEngine to use new components
2. Modify reasoning loops to incorporate new functionality
3. Test end-to-end integration
4. Update configuration and initialization
5. Verify backward compatibility
```

### **📚 Documentation Commands**

```
Update Moveworks project documentation:
1. Document newly implemented components
2. Update API documentation
3. Add usage examples and patterns
4. Update architecture diagrams if needed
5. Ensure all tracker documents are current
```

```
Create Moveworks implementation guide:
1. Document the complete architecture
2. Explain how each component works
3. Provide usage examples
4. Include troubleshooting guide
5. Add deployment instructions
```

---

## 🎯 **PHASE-SPECIFIC COMMANDS**

### **Phase 1 (Critical - Weeks 1-4)**
```
Focus on Phase 1 Moveworks implementation:
1. Implement Manifest Generator (autonomous plugin selection)
2. Build Slot Resolver System (Static/API/Inline resolvers)
3. Add Policy Validator Engine (compliance and safety)
4. Test all Phase 1 components thoroughly
5. Update tracker with Phase 1 completion
```

### **Phase 2 (High Priority - Weeks 5-8)**
```
Begin Phase 2 Moveworks implementation:
1. Enhance Action Orchestrator (workflow management)
2. Implement Plugin Selection Loop (multi-plugin coordination)
3. Add Multi-Plugin Response Engine (complex request handling)
4. Test Phase 2 integration with Phase 1 components
5. Update tracker with Phase 2 progress
```

### **Phase 3 (Medium Priority - Weeks 9-12)**
```
Complete Phase 3 Moveworks implementation:
1. Implement Guardrails System (safety mechanisms)
2. Add Steerability Tools (behavior adaptation)
3. Build Grounding System (hallucination prevention)
4. Enhance Planning and Execution Iteration
5. Add Continuous Learning capabilities
6. Complete final testing and documentation
```

---

## 🚨 **Emergency Commands**

### **Quick Status Check**
```
Quick Moveworks status: Show me the current progress percentage and next immediate action item
```

### **Blocker Resolution**
```
Help resolve Moveworks project blockers:
1. Identify current blockers from tracker
2. Suggest solutions or workarounds
3. Update timeline if needed
4. Prioritize critical path items
```

### **Rollback/Recovery**
```
Moveworks implementation issue - help me:
1. Identify what went wrong
2. Suggest rollback strategy if needed
3. Plan recovery approach
4. Update tracker with issue resolution
```

---

## 📋 **TRACKER DOCUMENT REFERENCES**

- **Main Tracker:** `docs/moveworks-reasoning-agent-review-tracker.md`
- **Action Items:** `docs/moveworks-action-items-tracker.md`  
- **Technical Details:** `docs/moveworks-technical-comparison.md`
- **This Prompt:** `docs/moveworks-project-management-prompt.md`

---

**💡 TIP:** Always start with a status check command to understand current progress before beginning new work!

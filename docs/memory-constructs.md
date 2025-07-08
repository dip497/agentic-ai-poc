# Memory Constructs - Moveworks Agentic Reasoning Engine

## Overview

Based on analysis of the official Moveworks documentation at https://help.moveworks.com/docs/memory-constructs-in-the-ai-assistant, this document examines our current Memory Constructs implementation against the actual Moveworks architecture.

## Actual Moveworks Memory Constructs

According to the official documentation, Moveworks uses **four types of memory** to provide complete knowledge and context:

### 1. Semantic Memory
**Purpose**: Knowledge of content, entities and terminology used by the organization
**Function**: Needed to understand the **semantics** of the conversation with the user

### 2. Episodic Memory  
**Purpose**: Awareness of current context in the conversation with the user
**Function**: What questions and answers have been exchanged and what decisions have been made already
**Goal**: Enable true interactive conversation where **every response is unique and tailored** to the context

### 3. Procedure Memory
**Purpose**: Knowledge of tasks that can be performed in the environment for the user
**Function**: What business processes or rules should be followed
**Goal**: Select the **right tools for the request and how to apply them**

### 4. Working Memory
**Purpose**: Awareness of what operations and processes are in progress
**Function**: Track which stage of completion multi-step processes are at
**Goals**: 
- Ensure multi-step synchronous and asynchronous processes are **tracked and driven to completion**
- Ensure responses are anchored in **references** so users can verify facts

### Variable Tracking Framework
Moveworks implements a **variable tracking framework** in working memory that:
1. **Prevents LLM hallucination**: LLMs can't accidentally hallucinate or mix-and-match IDs because the variable tracking system keeps them grounded
2. **Handles large datasets**: Operations are not limited by LLM context window size - can juggle thousands of records and perform calculations

## Issues in Our Current Implementation

### 1. Incorrect Memory Architecture
**Problem**: Our implementation doesn't match Moveworks patterns
- We have custom memory types that don't align with actual Moveworks architecture
- Missing proper variable tracking framework
- No reference grounding system

### 2. Hardcoded Values Found
**Problem**: Multiple hardcoded values violate configuration-driven approach
- `domain: str = "GENERAL_DOMAIN"` - should be dynamic from database
- `max_messages = 20` - hardcoded window size
- No configurable memory management parameters

### 3. Missing Moveworks-Specific Features
**Problem**: Key Moveworks memory features not implemented
- No variable tracking framework for business object integrity
- No reference anchoring system for fact verification
- No proper multi-step process tracking
- Missing conversation context management patterns

### 4. Database Schema Issues
**Problem**: Tables don't match Moveworks memory patterns
- Missing variable tracking tables
- No reference grounding tables
- Incorrect conversation context structure

## Required Fixes

### 1. Implement Variable Tracking Framework
- Add business object integrity tracking
- Implement ID grounding to prevent hallucination
- Support large dataset operations beyond LLM context window

### 2. Remove All Hardcoded Values
- Make domain names dynamic from database
- Configure memory window sizes
- Add configurable memory management parameters

### 3. Add Reference Grounding System
- Implement fact verification anchoring
- Add reference tracking in working memory
- Support user fact verification workflows

### 4. Fix Database Schema
- Add variable tracking tables
- Implement reference grounding tables
- Update conversation context structure

## Implementation Status
❌ **Not Implemented**: Variable tracking framework
❌ **Not Implemented**: Reference grounding system  
❌ **Hardcoded Values**: Multiple configuration values hardcoded
❌ **Schema Issues**: Database tables don't match Moveworks patterns

## Next Steps
1. Study Moveworks variable tracking framework implementation
2. Remove all hardcoded values and implement configuration-driven approach
3. Implement reference grounding system for fact verification
4. Update database schema to match Moveworks memory patterns
5. Add proper multi-step process tracking capabilities

---

**Related Components:**
- [Reasoning Loops](./reasoning-loops.md) - Uses memory constructs for context integration
- [Plugin Selection](./plugin-selection.md) - Leverages procedure memory for tool selection
- [Overview](./moveworks-agentic-reasoning-overview.md) - Complete architecture analysis

**Documentation Source:** https://help.moveworks.com/docs/memory-constructs-in-the-ai-assistant

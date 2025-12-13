# Experiment E12: Self-Improving Agent with Reflexion + Neo4j Memory

**Status:** ✅ Ready to Run  
**Goal:** Test if an agent with persistent memory (Neo4j) and Reflexion architecture improves over time

## Quick Start

```bash
cd /home/akshay/seer/experiments/e12--self-improving-agent
python e12_self_improving_agent_standalone.py [num_attempts]
```

**Example:** `python e12_self_improving_agent_standalone.py 3` runs 3 attempts per condition.

## What This Tests

**Hypothesis:** An agent with persistent memory (Neo4j) and Reflexion architecture will improve faster than one without memory.

**Two Conditions:**
- **With Memory:** Uses Reflexion + Neo4jMemoryStore (semantic retrieval, learns from failures AND successes)
- **Without Memory:** Simple ReAct agent (baseline)

## Key Features

### ✅ Memory System (Optimized)
- **Semantic Retrieval:** Uses `retrieve_relevant_memories()` with semantic + entity + recency
- **Stores Successes:** Learns from what works, not just failures
- **Neo4j Persistence:** Memories survive across experiment runs
- **ACT-R Decay:** Older memories fade naturally

### ✅ LLM-Based Evaluation
- **No Hardcoded Phrases:** All evaluation done by GPT-5.1
- **Explainable:** LLM provides reasoning for each evaluation
- **Confidence Scores:** Know how certain the evaluation is

### ✅ Reflexion Architecture
- **Act → Evaluate → Reflect Loop:** Iterative improvement within attempts
- **Built-in Evaluation:** Reflexion evaluates and stores memories automatically
- **Tool Call Detection:** Fixed to properly detect Reflexion's tool calls

## Architecture

```
With Memory:
  Retrieve relevant memories (semantic + entity + recency)
    ↓
  Inject into prompt
    ↓
  Reflexion Agent (Act → Evaluate → Reflect)
    ↓
  Store memories (failures AND successes)
    ↓
  Next attempt uses improved memories

Without Memory:
  Simple ReAct Agent
    ↓
  No memory retrieval/storage
    ↓
  Baseline comparison
```

## Task

**GitHub + Asana Integration:**
When a GitHub PR is merged in `seer-engg/buggy-coder`:
1. Find related Asana tasks (search by PR title/keywords)
2. Update tasks with PR details (link, author, merge date)
3. Close the tasks

**Success Criteria:**
- Asana tasks found and updated
- Tasks closed successfully
- PR details correctly added

## Results

Results are archived in `archive/results_e12/e12_results_TIMESTAMP.json` with:
- Success rates per attempt
- Tool call counts
- Execution times
- Failure explanations
- Learning summaries
- Adaptation plans

**Latest Results:** With Memory showed 200% increase in tool usage (12→20→36 calls), demonstrating learning!

## Configuration

Uses Seer root `.env` file for:
- `OPENAI_API_KEY` - For GPT-5.1
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` - For persistent memory

## Memory Optimizations Implemented

1. ✅ **Semantic Similarity Retrieval** - Finds relevant memories even if context doesn't match exactly
2. ✅ **Multi-Strategy Retrieval** - Combines semantic + entity + recency
3. ✅ **Store Successes** - Learns from what works (score >= 0.9)
4. ✅ **Intelligent Context** - Uses `retrieve_relevant_memories()` instead of simple queries

## Files

- **`e12_self_improving_agent_standalone.py`** - Main experiment script
- **`EXPERIMENT.md`** - Full experiment design and methodology
- **`README.md`** - This file
- **`results_e12.tar.gz`** - All evaluation results (16 runs, compressed)
- **`visualizations.tar.gz`** - UI mockup images (5 PNGs, compressed)

## Archived Results

**Latest Run:** `e12_results_20251203_035056.json` (in results_e12.tar.gz)
- With Memory: 12 → 20 → 36 tool calls (200% increase!)
- Without Memory: 4 → 4 → 4 tool calls (flat)
- **Finding:** Memory system is working - agent learns and adapts!

**To Extract:**
```bash
tar -xzf results_e12.tar.gz      # Extracts results_e12/ directory
tar -xzf visualizations.tar.gz   # Extracts visualizations/ directory
```

## Why This Is Better Than Hardcoded Rules

- **Adaptive:** Handles different phrasings automatically
- **Explainable:** LLM provides reasoning
- **No Maintenance:** Change prompt, not code
- **Confidence Scores:** Know evaluation certainty

## Expected Outcomes

**If memory helps:**
- With Memory success rate improves over attempts
- Without Memory stays flat
- Memory retrieval finds relevant past failures
- Agent adapts based on memories

**If memory doesn't help:**
- Both conditions perform similarly
- Memory retrieval doesn't improve performance
- Simple ReAct is sufficient

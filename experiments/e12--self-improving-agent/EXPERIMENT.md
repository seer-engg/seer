# Experiment E12: Self-Improving Agent

**Date:** 2025-12-03  
**Author:** akshay@getseer.dev  
**Status:** ✅ Ready to Run

## Core Hypothesis

**H1:** An agent with persistent memory (Neo4j) and Reflexion architecture will improve faster than one without memory.

**H0:** An agent with memory performs no better than one without.

## The Vision

1. **Self-critique accuracy improves over time** - Agent gets better at identifying mistakes
2. **Agent explains last 3 failures** - "I failed because X, so I changed Y"
3. **Visible learning** - See prompt versions, memories, learnings evolve
4. **Tool usage tracking** - See what failed, how agent adapts

## Experimental Design

### Task: GitHub + Asana Integration

**Task:** When a GitHub PR is merged in repository `seer-engg/buggy-coder`:
1. Find related Asana tasks (search by PR title/keywords)
2. Update the Asana tasks with PR details (link, author, merge date)
3. Close the Asana tasks

**Success Criteria:**
- Asana tasks found and updated
- Tasks closed successfully
- PR details correctly added to tasks

**Why This Task:**
- Real-world integration scenario
- Multiple failure modes (auth, API calls, data matching)
- Clear success criteria
- Fast execution (~2-3 minutes per attempt)

### Two Conditions

1. **With Memory (Learning Agent)**
   - Uses Reflexion architecture (Act → Evaluate → Reflect loop)
   - Neo4jMemoryStore for persistent memory
   - Semantic retrieval of relevant past experiences
   - Learns from failures AND successes
   - Memory persists across experiment runs

2. **Without Memory (Baseline)**
   - Simple ReAct agent
   - No memory retrieval/storage
   - Starts fresh each attempt
   - Baseline for comparison

### Runs

- **Attempts:** 3 per condition (configurable via command line)
- **Total Runs:** 6 (3 with memory + 3 without)

## Memory System (Optimized)

### Retrieval Strategy
- **Multi-Strategy:** Semantic similarity + entity matching + recency
- **Intelligent:** Uses `retrieve_relevant_memories()` with LLM-generated search queries
- **Relevant:** Finds memories even if context doesn't match exactly

### Storage Strategy
- **Failures:** Stored when score < 0.8 (eval_threshold)
- **Successes:** Stored when score >= 0.9 (learn from what works)
- **Persistence:** All memories saved to Neo4j, survive across runs
- **ACT-R Decay:** Older memories fade naturally

### Memory Context Injection
- Memories retrieved BEFORE execution
- Injected into prompt as "RELEVANT PAST EXPERIENCES"
- Agent learns from both failures and successes

## Evaluation System

### LLM-Based Evaluation (No Hardcoded Phrases)
- **Evaluator:** GPT-5.1 with high reasoning
- **Input:** Task, agent output, tool calls, success criteria
- **Output:** JSON with success, reasoning, confidence
- **Adaptive:** Handles different phrasings automatically

### Success Detection
- Requires actual tool calls (no planning-only responses)
- Checks for task completion indicators
- Rejects requests for input or code-only responses
- Uses LLM reasoning, not keyword matching

## Metrics

### Primary Metrics
1. **Success Rate by Attempt** - Does memory improve over time?
2. **Tool Call Count** - Are tools being used correctly?
3. **Execution Time** - Does memory add overhead?

### Secondary Metrics
1. **Memory Retrieval Quality** - Are relevant memories found?
2. **Learning Trajectory** - Does agent adapt based on memories?
3. **Failure Explanations** - Quality of self-critique

## Expected Outcomes

**If H1 is true:**
- With Memory success rate improves over attempts
- Without Memory stays flat
- Memory retrieval finds relevant past failures
- Agent adapts based on memories

**If H0 is true:**
- Both conditions perform similarly
- Memory doesn't improve performance
- Simple ReAct is sufficient

## Implementation Details

### Reflexion Architecture
- **Simple Mode:** Act (ReAct) → Evaluate → [loop if score < threshold]
- **Memory Integration:** Retrieves memories before execution, stores after evaluation
- **Tool Detection:** Fixed to properly detect Reflexion's tool calls in `run_trace`

### Neo4j Memory Store
- **Connection:** Uses Seer root `.env` credentials
- **Schema:** ACT-R activation, semantic embeddings, entity relationships
- **Retrieval:** Multi-strategy (semantic + entity + recency)

### Tool Filtering
- Filters to GitHub/Asana tools only
- Limits to 120 tools (OpenAI's 128 limit)
- Filters out tool names > 64 characters

## Results Format

Results saved to `results_e12/e12_results_TIMESTAMP.json`:

```json
{
  "with_memory": [...runs...],
  "without_memory": [...runs...],
  "metrics": {
    "with_memory_success_rates": [0.0, 0.0, 0.0],
    "without_memory_success_rates": [0.0, 0.0, 0.0],
    "improvement_trend": {...}
  }
}
```

Each run includes:
- Success status
- Tool call count
- Execution time
- Failure explanation (if failed)
- Learning summary (if with memory)
- Adaptation plan (if with memory)

## Running the Experiment

```bash
# Default: 1 attempt per condition
python e12_self_improving_agent_standalone.py

# Custom: 3 attempts per condition
python e12_self_improving_agent_standalone.py 3
```

## Key Innovations

1. **No Hardcoded Phrases** - All evaluation done by LLM
2. **Semantic Memory Retrieval** - Finds relevant memories, not just recent
3. **Learn from Successes** - Stores what works, not just failures
4. **Persistent Memory** - Memories survive across runs
5. **Reflexion Architecture** - Built-in evaluation and reflection loop

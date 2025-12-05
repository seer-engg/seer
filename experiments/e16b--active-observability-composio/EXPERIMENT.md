# E16B: Baseline Delegate Orchestrator with Explicit Planning

**Date:** 2025-01-XX  
**Status:** ✅ Ready to Run  
**Task:** GitHub ↔ Asana Integration (Sync PRs to Tasks)

## Core Hypothesis

**H₁:** A baseline delegate orchestrator with explicit planning and dynamic worker spawning will achieve better context efficiency and success rates compared to a single ReAct agent on complex multi-step integration tasks.

**H₀:** Baseline delegate architecture performs no better than single-agent ReAct.

## Architecture Overview

### Baseline Delegate Orchestrator (Treatment)
- **Phase 1: Explicit Planning** - Dedicated planning step generates detailed execution plan
- **Phase 2: Dynamic Execution** - Orchestrator spawns generic workers on-demand
- **Context Isolation** - Workers execute in isolated threads, preventing context bloat
- **Plan Visibility** - Execution plan stored in state, visible and mutable

### Baseline (Control)
- **Single ReAct Agent** - Standard ReAct agent with all tools available
- **Full Context** - All tool outputs and reasoning in single thread
- **ToolHub Integration** - Dynamic tool discovery via semantic search

## Experimental Setup

**Task:**  
"ACTION REQUIRED: Sync a GitHub PR to Asana.

EXECUTE THESE STEPS IN ORDER:
1. Search GitHub for the most recently merged/closed PR in repository 'seer-engg/buggy-coder'.
2. Extract PR details: title, URL, author, merge/close date.
3. Search Asana for tasks matching the PR title or keywords.
4. DECISION:
   - IF task EXISTS: Update the task with PR details (add comment with PR URL, author, date).
   - IF task DOES NOT EXIST: Create a new Asana task with PR title and details.
5. Close the Asana task (whether it was updated or newly created)."

**Success Criteria:**
- PR found and details extracted
- Asana task found/created and updated with PR details
- Task closed successfully
- All steps completed without errors

**Conditions:**
1. **Baseline:** Single ReAct agent with ToolHub (dynamic tool discovery)
2. **Treatment:** Baseline Delegate Orchestrator with explicit planning + dynamic worker spawning

## Implementation Details

### Baseline Delegate Architecture

**Orchestrator:**
- Uses `generate_detailed_plan()` to create execution plan
- Spawns workers dynamically via `spawn_worker(task_instruction)`
- Tracks plan status: `pending` → `completed`/`failed`
- Can adapt plan mid-execution

**Generic Workers:**
- Created on-demand with task-specific instructions
- Use `search_tools()` and `execute_tool()` for dynamic tool discovery
- Return file paths or status summaries

**LangFuse Tracing:**
- Orchestrator trace contains nested worker spans
- Full observability of planning → execution flow
- Worker tool calls visible as child operations

### Baseline Agent

**ReAct Agent:**
- Standard LangGraph ReAct agent
- ToolHub for semantic tool discovery
- All tools available in single context

## Metrics

### Primary Metrics
1. **Success Rate** - Task completion percentage
2. **Context Efficiency** - Peak context size (characters/tokens)
3. **Execution Time** - Total time to complete task
4. **Tool Call Count** - Number of tool invocations

### Secondary Metrics
1. **Plan Quality** - Number of steps, dependencies, clarity
2. **Worker Efficiency** - Average worker execution time
3. **Context Bloat Reduction** - Comparison of context sizes
4. **Adaptation Rate** - Plan modifications during execution

## Expected Outcomes

**If H₁ is true:**
- Baseline delegate achieves similar/better success rate
- Significantly lower context size (workers isolated)
- Better plan quality (explicit planning)
- Visible plan adaptation when needed

**If H₀ is true:**
- Both conditions perform similarly
- Baseline delegate adds overhead without benefit
- Single-agent sufficient for task complexity

## Running the Experiment

```bash
cd /home/akshay/seer/experiments/e16b--active-observability-composio
python3 run_benchmark.py
```

**Output:**
- Results saved to `results/benchmark_TIMESTAMP.json`
- LangFuse traces for both conditions
- Comparative metrics and analysis

## Key Innovations

1. **Explicit Planning** - Dedicated planning phase before execution
2. **Dynamic Worker Spawning** - Workers created on-demand, not pre-configured
3. **Generic Workers** - Single worker type with dynamic tool discovery
4. **Plan Visibility** - Plan stored in state, visible and mutable
5. **Context Isolation** - Workers execute in isolated threads
8. **Full Observability** - LangFuse traces show complete flow

## Architecture Files

- `agents/baseline_delegate.py` - Orchestrator with planning/execution nodes
- `agents/baseline_react.py` - Single ReAct agent (baseline)
- `agents/generic_worker.py` - Generic worker with dynamic tool discovery
- `tools/planning.py` - Planning tools (generate_plan, add_step, mark_complete/failed)
- `tools/spawn_worker.py` - Dynamic worker spawning with callback propagation
- `run_benchmark.py` - Experiment runner comparing both conditions

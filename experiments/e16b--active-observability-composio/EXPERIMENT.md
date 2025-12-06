# E16B: Baseline Delegate Orchestrator with Explicit Planning

**Date:** 2025-01-XX  
**Status:** ✅ Ready to Run  
**Tasks:** 8 integration tasks (GitHub ↔ Asana, Google Docs/Sheets, Slack, Telegram, Twitter, Gmail, Calendar)

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

This experiment tests the baseline delegate orchestrator on **8 integration tasks** from previous experiments (E6, E7) to compare performance across different task types and complexities.

### Task 1: GitHub ↔ Asana Integration (Original E16B Task)
**Instruction:**  
"ACTION REQUIRED: Sync a GitHub PR to Asana.

EXECUTE THESE STEPS IN ORDER:
1. Search GitHub for the most recently merged/closed PR in repository 'seer-engg/buggy-coder'.
2. Extract PR details: title, URL, author, merge/close date.
3. OPTIONALLY search Asana for tasks matching the PR title or keywords (if search fails or finds nothing, proceed to step 4).
4. DECISION:
   - IF task EXISTS (from step 3): Update the task with PR details (add comment with PR URL, author, date).
   - IF task DOES NOT EXIST (search found nothing or search was skipped): Create a new Asana task with PR title and details.
5. Close the Asana task (whether it was updated or newly created)."

**Success Criteria:**
- PR found and details extracted
- Asana task found/created and updated with PR details
- Task closed successfully
- All steps completed without errors

**Services:** GitHub, Asana  
**Complexity:** Complex

### Task 2: Weekly Work Summary
**Instruction:**  
Create a weekly work summary: 
1. Get my Google Calendar events for next week
2. Get my GitHub pull requests from last week (all repos)
3. Get my Slack messages from #engineering channel from last week
4. Combine all information into a structured Google Doc
5. Share the document with my manager (look up their email)
6. Send them a Slack notification with the doc link

**Success Criteria:**
- Google Doc created with all data
- Doc shared with manager
- Slack notification sent

**Services:** Google Calendar, GitHub, Slack, Google Docs  
**Complexity:** Complex

### Task 3: Email Task List
**Instruction:**  
Find all unread emails from last month that contain 'meeting' or 'urgent', extract any action items or deadlines mentioned, create a prioritized task list in Google Sheets with columns: Task, Deadline, Priority, and send me a summary email with the sheet link.

**Success Criteria:**
- Google Sheet created with tasks
- Summary email sent

**Services:** Gmail, Google Sheets  
**Complexity:** Complex

### Task 4: GitHub Bug Summary
**Instruction:**  
Find all GitHub issues assigned to me that mention 'bug' or 'error',
check if any have related Slack discussions in #bugs channel,
create a summary document with: issue title, description, and related Slack context,
and post it to #engineering channel with priority tags based on issue labels

**Success Criteria:**
- Summary doc created
- Message posted to #engineering with doc link

**Services:** GitHub, Slack, Google Docs  
**Complexity:** Complex

### Task 5: Telegram Message (Simple)
**Instruction:**  
send good morning message to +1 646-371-6198 via telegram

**Success Criteria:**
- Check API response 200 OK from Telegram

**Services:** Telegram  
**Complexity:** Simple

### Task 6: Twitter Trends (Simple)
**Instruction:**  
what are the latest trends going on twitter

**Success Criteria:**
- Check Twitter API for post results

**Services:** Twitter  
**Complexity:** Simple

### Task 7: Meeting Summary Document
**Instruction:**  
Create a meeting summary document. Get all attendees from my last Google Calendar meeting, fetch their GitHub activity from the past week, create a Google Doc summarizing the meeting and their contributions, and share it with all attendees via email.

**Success Criteria:**
- Google Doc created with meeting summary
- Shared via email with all attendees

**Services:** Google Calendar, GitHub, Google Docs, Gmail  
**Complexity:** Complex

### Task 8: Deployment Summary
**Instruction:**  
Find all Slack messages in #engineering from last week that mention 'deploy' or 'release', check corresponding GitHub pull requests, create a deployment summary in Google Sheets with columns: Date, PR, Author, Status, and notify the team in Slack.

**Success Criteria:**
- Google Sheet created with deployment summary
- Slack notification sent

**Services:** Slack, GitHub, Google Sheets  
**Complexity:** Complex

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
- Results saved to `results/benchmark_TIMESTAMP.json` (includes all 8 tasks)
- LangFuse traces for each task execution
- Per-task metrics and overall summary statistics
- Success rates by complexity (simple vs complex tasks)

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

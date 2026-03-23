# Workflow Graph Structure & Compilation Guide

## Overview

Seer workflows are directed graphs compiled to LangGraph StateGraphs. Understanding how the compiler transforms your nodes and edges into executable graphs is crucial for building correct workflows.

---

## Compilation Pipeline

The compiler transforms WorkflowSpec JSON through these stages:

1. **Parse** - Validate JSON against Pydantic schema
2. **Build Type Environment** - Infer node output types from tools/models
3. **Validate References** - Check all `${...}` expressions resolve
4. **Lower Control Flow** - Build ExecutionPlan with edge indices
5. **Emit LangGraph** - Generate executable StateGraph

---

## Graph Structure Fundamentals

### Nodes
- Each node becomes a LangGraph node with a unique ID
- Nodes are functions that transform state: `(state, config) -> state_update`
- All node outputs are stored in state under their node ID
- State updates are **merged**, not replaced (preserves all node outputs)

### Edges
- Define execution flow between nodes
- Can be **direct** (sequential) or **conditional** (routing)
- Multiple edges FROM a node → conditional routing
- Multiple edges TO a node → merge point (both paths can reach it)

### State Management
- State is a dictionary: `{node_id: output, ...}`
- State updates use a **merge reducer**: `{**left, **right}`
- All node outputs persist throughout execution
- Trace keys (`_trace_*`) track execution metadata

---

## Entry Points (START Routing)

### Single Entry Point
When workflow has no triggers, START connects to the first node:

```json
{
  "nodes": [{"id": "first", ...}],
  "edges": []
}
```

**Compiled:** `START → first`

### Trigger-Based Entry
When workflow has triggers, START routes through bootstrap node:

```json
{
  "triggers": [
    {"id": "trigger1", ...},
    {"id": "trigger2", ...}
  ],
  "edges": [
    {"source": "trigger1", "target": "node_a", "type": "trigger"},
    {"source": "trigger2", "target": "node_b", "type": "trigger"}
  ]
}
```

**Compiled:**
- `START → __trigger_bootstrap`
- `__trigger_bootstrap` reads `_trigger_id` from state
- Routes to appropriate target based on which trigger fired

---

## Exit Points (END Routing)

### Explicit END
Nodes with no outgoing edges automatically connect to END:

```json
{
  "nodes": [
    {"id": "task", ...}
  ],
  "edges": []
}
```

**Compiled:** `task → END`

### Conditional END
Control flow nodes can route to END if no target specified:

```json
{
  "id": "check",
  "type": "if",
  "condition": "${done}"
}
```

**If only one branch has a target, other routes to END**

---

## Edge Types & Routing

### Default Edges (Sequential Flow)
```json
{"source": "node_a", "target": "node_b"}
{"source": "node_a", "target": "node_b", "type": "default"}
```

**Behavior:** Direct edge, node_b executes after node_a

### Multiple Outgoing Edges
**NOT ALLOWED on regular nodes.** Only control flow nodes (if, for_each) can have multiple outgoing edges.

❌ **Invalid:**
```json
{
  "nodes": [{"id": "task", "type": "tool", ...}],
  "edges": [
    {"source": "task", "target": "a"},
    {"source": "task", "target": "b"}
  ]
}
```

✅ **Valid (use if node):**
```json
{
  "nodes": [
    {"id": "task", "type": "tool", ...},
    {"id": "route", "type": "if", "condition": "${task.result == 'A'}"},
    {"id": "a", "type": "tool", ...},
    {"id": "b", "type": "tool", ...}
  ],
  "edges": [
    {"source": "task", "target": "route"},
    {"source": "route", "target": "a", "type": "conditional_true"},
    {"source": "route", "target": "b", "type": "conditional_false"}
  ]
}
```

### Conditional Edges (If Node)
If nodes REQUIRE exactly two edges:

```json
{
  "source": "if_node",
  "target": "true_path",
  "type": "conditional_true"
},
{
  "source": "if_node",
  "target": "false_path",
  "type": "conditional_false"
}
```

**Compiled:**
- If node evaluates condition, stores result in `_if_result_{node_id}`
- Router function reads result and routes to appropriate target
- LangGraph uses `add_conditional_edges(node_id, router, path_map)`

### Loop Edges (ForEach Node)
ForEach nodes REQUIRE exactly two edges:

```json
{
  "source": "loop_node",
  "target": "body_node",
  "type": "loop_body"
},
{
  "source": "loop_node",
  "target": "exit_node",
  "type": "loop_exit"
}
```

**Compiled:**
- Loop node manages iteration state in `_loop_{node_id}`
- Router checks `has_more_iterations` to decide loop_body vs loop_exit
- Terminal nodes in loop body automatically get implicit edge back to loop node

### ⚠️ Loop Exit Target Requirements

**CRITICAL:** The `loop_exit` edge target MUST be an existing node ID in the `nodes` array.

❌ **WRONG** - "done" doesn't exist as a node:
```json
{
  "nodes": [
    {"id": "loop", "type": "for_each", "items": "${data}"},
    {"id": "process", "type": "tool", ...}
  ],
  "edges": [
    {"source": "loop", "target": "process", "type": "loop_body"},
    {"source": "loop", "target": "done", "type": "loop_exit"}
  ]
}
```
**Error:** `Edge with source 'loop' and target 'done' target not found in nodes`

✅ **CORRECT** - exit target exists in nodes array:
```json
{
  "nodes": [
    {"id": "loop", "type": "for_each", "items": "${data}"},
    {"id": "process", "type": "tool", ...},
    {"id": "complete", "type": "tool", ...}
  ],
  "edges": [
    {"source": "loop", "target": "process", "type": "loop_body"},
    {"source": "loop", "target": "complete", "type": "loop_exit"}
  ]
}
```

✅ **ALSO CORRECT** - omit loop_exit if loop is the final step:
If the loop is the last operation and nothing needs to happen after, you can omit the `loop_exit` edge. The workflow will end implicitly after the loop completes.

### Trigger Edges
```json
{
  "source": "trigger_id",
  "target": "first_node",
  "type": "trigger"
}
```

**Behavior:** Routes from trigger bootstrap to first processing node

---

## Diamond Patterns & Merging

### Diamond Pattern (If/Else Merge)
Two paths that merge back to a single node:

```json
{
  "nodes": [
    {"id": "start", "type": "tool", ...},
    {"id": "check", "type": "if", "condition": "${start.value > 0}"},
    {"id": "path_a", "type": "tool", ...},
    {"id": "path_b", "type": "tool", ...},
    {"id": "merge", "type": "tool", ...}
  ],
  "edges": [
    {"source": "start", "target": "check"},
    {"source": "check", "target": "path_a", "type": "conditional_true"},
    {"source": "check", "target": "path_b", "type": "conditional_false"},
    {"source": "path_a", "target": "merge"},
    {"source": "path_b", "target": "merge"}
  ]
}
```

**Behavior:**
- Both `path_a` and `path_b` have edges to `merge`
- LangGraph handles multiple incoming edges naturally
- `merge` node executes after either path completes
- State contains outputs from: `start`, `check`, (either `path_a` OR `path_b`), `merge`

**Important:** The merge node can reference outputs from either path using if/else logic:
```json
{
  "id": "merge",
  "type": "agent",
  "inputs": {
    "model": "qwen/qwen3-235b-a22b-2507",
    "prompt": "Result from path A: ${path_a}, Result from path B: ${path_b}"
  }
}
```

**Note:** Only ONE path executes, so only one of `${path_a}` or `${path_b}` will have a value.

---

## Multiple Incoming Edges

Multiple nodes can target the same node (merge point):

```json
{
  "edges": [
    {"source": "node_a", "target": "merge"},
    {"source": "node_b", "target": "merge"},
    {"source": "node_c", "target": "merge"}
  ]
}
```

**Valid patterns:**
1. **Sequential merge:** Different sequential paths lead to same node (OK but unusual)
2. **Conditional merge:** If/else branches merge (diamond pattern - common)
3. **Loop merge:** Loop exit leads to node that other paths also reach

**LangGraph behavior:** Node executes when ANY incoming edge is followed.

---

## Loop Body Detection

The compiler automatically detects loop body nodes:

```json
{
  "nodes": [
    {"id": "loop", "type": "for_each", "items": "${items}"},
    {"id": "process", "type": "tool", ...},
    {"id": "transform", "type": "tool", ...},
    {"id": "finalize", "type": "tool", ...}
  ],
  "edges": [
    {"source": "loop", "target": "process", "type": "loop_body"},
    {"source": "process", "target": "transform"},
    {"source": "transform", "target": "loop"},
    {"source": "loop", "target": "finalize", "type": "loop_exit"}
  ]
}
```

**Compilation:**
1. Compiler finds `loop_body` edge: `loop → process`
2. Follows `default` edges from `process`: `process → transform`
3. Detects `transform` has edge back to `loop`
4. Loop body nodes: `{process, transform}`
5. Terminal nodes: `{transform}` (already has edge back to loop)

**Implicit loop back edges:**
If terminal node doesn't have explicit edge to loop, compiler adds one:

```json
{
  "edges": [
    {"source": "loop", "target": "process", "type": "loop_body"},
    {"source": "process", "target": "transform"}
    // No edge from transform back to loop
  ]
}
```

**Compiled:** Implicit edge added: `transform → loop`

---

## State Merging & Trace Keys

### State Updates
Every node execution produces a state update dict:
```python
{
  "node_id": <node_output>,
  "_trace_node_id": {"inputs": ..., "output": ..., "timestamp": ...}
}
```

### Merge Strategy
State updates use **merge reducer** (not replace):
```python
def merge_state(left: dict, right: dict) -> dict:
    return {**left, **right}
```

**Example:**
```
Initial state: {}
After node_a: {"node_a": "result_a", "_trace_node_a": {...}}
After node_b: {"node_a": "result_a", "_trace_node_a": {...}, "node_b": "result_b", "_trace_node_b": {...}}
```

**Important:** All node outputs persist throughout execution. You can reference earlier nodes from later nodes.

---

## Key Constraints & Rules

### Node Constraints
1. ✅ **Unique IDs:** Every node must have a unique `id`
2. ✅ **Reachability:** All nodes must be reachable from START (no orphaned nodes)
3. ❌ **No Self-Edges:** Node cannot connect to itself (except through loop pattern)
4. ❌ **No Multiple Outgoing Edges:** Only if/for_each nodes can have conditional edges

### Edge Constraints
1. ✅ **Valid Source/Target:** Both must exist in nodes list
2. ✅ **If Node Edges:** Must have exactly one `conditional_true` and one `conditional_false`
3. ✅ **ForEach Node Edges:** Must have exactly one `loop_body` and one `loop_exit`
4. ✅ **Trigger Edges:** Source is trigger ID, target is node ID
5. ❌ **Circular Dependencies:** Avoid cycles (except loop pattern)

### Control Flow Constraints
1. **If Node:**
   - Must have `condition` field
   - Must have both conditional edges
   - Cannot have default edges

2. **ForEach Node:**
   - Must have `items` field
   - Must have both loop edges
   - Loop body nodes automatically detected
   - Terminal nodes get implicit back-edge

3. **Tool/Agent/MCP Nodes:**
   - Can only have default edges
   - Cannot have conditional edges
   - Can have zero or more outgoing edges (if zero, routes to END)

---

## Common Patterns

### Pattern 1: Linear Flow
```
START → node_a → node_b → node_c → END
```

### Pattern 2: Conditional Branch
```
START → check_node (if) → path_a → END
                        └→ path_b → END
```

### Pattern 3: Diamond (Branch + Merge)
```
START → node_a → check (if) → path_a → merge → END
                            └→ path_b ┘
```

### Pattern 4: Loop
```
START → loop (for_each) → process → transform → loop → END
         └──────────────────────────────────────┘
```

### Pattern 5: Nested Control Flow
```
START → outer_loop (for_each) → check (if) → path_a → outer_loop → END
                                           └→ path_b ┘
```

### Pattern 6: Multiple Triggers
```
START → __trigger_bootstrap → node_a (trigger1)
                            └→ node_b (trigger2)
                            └→ node_c (trigger3)
```

---

## Pre-Submission Validation Checklist

Before submitting a workflow, verify each item:

### Version
- [ ] `"version": "2"` (exactly, NOT "1.0", NOT "2.0")

### Nodes
- [ ] All node IDs are unique
- [ ] Tool names match exactly from `search_tools()` results
- [ ] Agent nodes have `model` and `prompt` in inputs
- [ ] ForEach `items` expression resolves to an array

### Edges
- [ ] All edge `source` values exist in nodes array (or triggers for trigger edges)
- [ ] All edge `target` values exist in nodes array
- [ ] `if` nodes have BOTH `conditional_true` AND `conditional_false` edges
- [ ] `for_each` nodes have `loop_body` AND `loop_exit` edges
- [ ] `loop_exit` targets point to an existing node (not an undefined ID like "done")

### Expressions
- [ ] Template expressions use substitution only (NO arithmetic like `${x + 1}`)
- [ ] Arithmetic/functions only appear in `if` node conditions
- [ ] All `${...}` references point to existing nodes or trigger IDs
- [ ] Trigger references use explicit ID format: `${trigger_id.data.field}` (not `${trigger.data.field}`)

### Common Mistakes to Avoid
- ❌ `"version": "1.0"` or `"version": "2.0"` → Use `"version": "2"`
- ❌ `{"source": "loop", "target": "done", ...}` where "done" isn't a node → Add the missing node
- ❌ `"range": "Sheet1!A${index + 2}"` → Arithmetic doesn't work in templates
- ❌ `${item[0]}` in for_each loop → May fail due to type inference; use agent to extract

---

## Debugging Graph Issues

### Issue: "Node X is not reachable"
**Cause:** No path from START to node X
**Fix:** Add edges connecting START (or trigger) to the orphaned node

### Issue: "If node missing conditional edges"
**Cause:** If node doesn't have both `conditional_true` and `conditional_false` edges
**Fix:** Add both edge types

### Issue: "Multiple outgoing edges from tool node"
**Cause:** Tool/Agent/MCP node has more than one outgoing edge
**Fix:** Use If node for branching, not multiple edges from tool node

### Issue: "Circular reference detected"
**Cause:** Node A references node B, which references node A
**Fix:** Ensure dependencies flow in one direction (except in loop patterns)

### Issue: "ForEach loop body not detected"
**Cause:** No `loop_body` edge or broken edge chain in loop
**Fix:** Ensure loop has `loop_body` edge and body nodes connect back to loop

---

## Best Practices

1. **Keep graphs acyclic** (except loops)
2. **Use If nodes for branching**, not multiple edges from regular nodes
3. **Merge diamond patterns explicitly** with a merge node
4. **Name nodes descriptively** for easier debugging
5. **Validate edge types** match node types (conditional for if, loop for for_each)
6. **Test diamond patterns** to ensure both paths work correctly
7. **Use trigger routing** for event-driven workflows
8. **Avoid deep nesting** of control flow (keep it simple)

---

## Quick Reference

| Pattern | Structure | Use Case |
|---------|-----------|----------|
| Linear | A → B → C | Sequential operations |
| Branch | A → if → B or C | Route based on condition |
| Diamond | A → if → B/C → D | Branch then merge |
| Loop | for_each → body → for_each | Iterate over items |
| Multi-trigger | START → bootstrap → A/B/C | Event-driven routing |

---

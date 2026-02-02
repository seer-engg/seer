---
name: workflow-compiler
description: Guide for understanding and working with the workflow compiler and runtime in src/seer/core/. Use when implementing node executors, debugging compilation/runtime issues, adding workflow features, or understanding the workflow execution architecture.
allowed-tools: Read, Grep, Glob, Bash(pytest:*)
---

# Workflow Compiler & Runtime Skill

Comprehensive guide for the workflow compiler and runtime system in `src/seer/core/`. This skill helps you understand the compilation pipeline, runtime execution, and how to extend the workflow system.

## Quick Reference

**When to use this skill:**
- Implementing new node types or node executors
- Debugging compilation or runtime errors
- Understanding workflow execution flow
- Adding features to the compiler or runtime
- Working on state management or checkpointing
- Investigating LangGraph integration issues

**Key principle:** Workflow compiler is pure compilation & runtime logic with NO database access or HTTP dependencies. API layer (`/api/workflows`) orchestrates compiler + persistence.

## Architecture Overview

```
WorkflowSpec (JSON/Dict)
    ↓
WorkflowCompiler.compile()
    ├── Validation (schema, node types, edges)
    ├── Transformation (DSL → LangGraph nodes)
    └── Graph Construction (StateGraph)
    ↓
CompiledGraph (LangGraph Runnable)
    ↓
Runtime Execution
    ├── Node executors (tool, llm, code, conditional, etc.)
    ├── State management (checkpoints)
    └── Streaming/async execution
```

## Directory Structure

```
src/seer/core/
├── README.md                   # Architecture documentation
├── __init__.py
├── compiler/                   # Compilation logic
│   ├── parse.py               # Stage 1: JSON → WorkflowSpec
│   ├── type_env.py            # Stage 2: Build type environment
│   ├── validate_refs.py       # Stage 3: Validate ${...} references
│   ├── lower_control_flow.py  # Stage 4: Transform to execution plan
│   └── emit_langgraph.py      # Stage 5: Generate LangGraph
├── runtime/                    # Execution logic
│   ├── global_compiler.py     # WorkflowCompilerSingleton
│   ├── nodes.py               # Node executors (587 lines)
│   ├── execution.py           # Execution helpers
│   ├── state.py               # State management utilities
│   ├── input_validation.py    # Runtime input validation/coercion
│   └── validate_output.py     # Runtime output schema validation
├── schema/                     # Data models
│   ├── models.py              # WorkflowSpec, NodeConfig Pydantic models
│   └── jsonschema_adapter.py  # JSON Schema validation utilities
├── expr/                       # Expression system
│   ├── evaluator.py           # Runtime ${...} evaluation
│   └── typecheck.py           # Type checking for references
├── registry/                   # Node type registry
├── triggers/                   # Trigger definitions
├── errors.py                   # Error hierarchy
└── event_loop.py              # Event loop utilities
```

## Core Components

### 1. WorkflowCompiler (compiler/compiler.py)

**Purpose:** Transform WorkflowSpec → executable LangGraph

**Usage:**
```python
from seer.core.runtime.global_compiler import WorkflowCompilerSingleton

compiler = WorkflowCompilerSingleton.instance()
graph = await compiler.compile(user, spec_dict, checkpointer=None)
```

**Compilation Steps:**
1. **Validate spec**: Check required fields, node types exist, edges valid
2. **Build state graph**: Create `StateGraph` with workflow state schema
3. **Add nodes**: Register node executors for each node in spec
4. **Add edges**: Connect nodes with conditional/normal edges
5. **Set entry/finish**: Configure graph entry point and END state
6. **Compile**: `graph.compile(checkpointer=...)`

**Caching:** Singleton maintains compiled graph cache (keyed by spec hash).

### 2. Node Executors (runtime/nodes.py)

**Purpose:** Execute workflow nodes during runtime

**Node Types:**
- **Tool Node**: Execute registered tools (e.g., Gmail, GitHub)
- **LLM Node**: Call LLM with prompts/messages
- **Code Node**: Execute Python code sandbox
- **Conditional Node**: Branch based on expression evaluation
- **Transform Node**: Map/transform data
- **Trigger Node**: Workflow entry point (receives trigger events)
- **Input Node**: Workflow inputs
- **Output Node**: Workflow outputs
- **Extract Node**: Extract structured data from text
- **If/Else Node**: Conditional branching
- **For Each Node**: Loop iteration

**Executor Pattern:**
```python
async def execute_tool_node(state, node_config):
    tool_name = node_config["tool"]
    arguments = resolve_node_inputs(state, node_config["inputs"])

    result = await execute_tool(
        tool_name=tool_name,
        user=state["user"],
        connection_id=node_config.get("connection_id"),
        arguments=arguments
    )

    return {f"node_{node_id}_output": result}
```

**Key File:** `src/seer/core/runtime/nodes.py:587` (large file, consider splitting by node type)

### 3. State Management (schema/models.py, runtime/state.py)

**WorkflowState Structure:**
```python
{
    "user": User,                    # User context
    "node_outputs": {...},           # Node execution results
    "workflow_inputs": {...},        # Initial inputs
    "messages": [...],               # LLM conversation history
    "interrupts": [...],             # Human-in-the-loop approvals
    # ... node-specific outputs
}
```

**State Updates:** Nodes return partial state updates merged by LangGraph.

**Input Resolution Types:**
- `static`: Hardcoded value
- `input`: Reference another node's output
- `expression`: Evaluate expression (e.g., `${node_1.output.result}`)
- `workflow_input`: Reference workflow input parameter

### 4. Checkpointing

**Purpose:** Save workflow state after each node execution for:
- Resume interrupted workflows
- Replay execution
- Debugging state transitions

**Checkpointer Types:**
- `MemorySaver`: In-memory (dev/testing)
- `DatabaseCheckpointer`: PostgreSQL (production)

**Usage:**
```python
checkpointer = await get_checkpointer()
graph = await compiler.compile(user, spec, checkpointer=checkpointer)
```

### 5. Expression System (expr/)

**Expression Evaluation:**
```python
from seer.core.expr.evaluator import EvaluationContext, resolve_reference

ctx = EvaluationContext(
    state={"node1": {"output": "value"}},
    inputs={"user_input": "test"},
    locals={"item": "current_item", "index": 0},
    config={"api_key": "***"}
)
```

**Supports:**
- Property access: `${node.field}`
- Index access: `${array[0]}`
- Nested access: `${node.data.items[0].name}`

**Type Checking:** `expr/typecheck.py` provides static type validation during compilation.

## Workflow Spec (DSL)

### Example Structure

```json
{
  "nodes": {
    "node_1": {
      "id": "node_1",
      "type": "tool",
      "config": {
        "tool": "gmail_send_email",
        "inputs": {
          "to": {"type": "static", "value": "user@example.com"},
          "subject": {"type": "input", "node_id": "node_0", "output_key": "subject"}
        },
        "connection_id": "conn_abc123"
      }
    },
    "node_2": {
      "type": "llm",
      "config": {
        "model": "gpt-4",
        "prompt": "Summarize: ${node_1.output}"
      }
    }
  },
  "edges": [
    {"from": "node_1", "to": "node_2"}
  ],
  "entry_point": "node_1",
  "metadata": {
    "name": "My Workflow",
    "description": "Sends email and summarizes"
  }
}
```

### Schema Models

**Key Pydantic Models** (schema/models.py):
- `WorkflowSpec`: Top-level workflow definition
- `Node`: Base class for all block types
- `InputDef`: Input parameter definition with type and default value
- `OutputContract`: Declares what a node writes (text or JSON with schema)
- `SchemaRef`: Reference to known schema
- `InlineSchema`: Inline JSON Schema

**Pattern:** All models extend `StrictModel` (extra="forbid", validate_assignment=True)

## Execution Modes

### 1. Synchronous Execution

```python
result = await graph.ainvoke(
    {"workflow_inputs": {...}},
    config={"configurable": {"thread_id": run_id}}
)
```

Returns final state after completion.

### 2. Streaming Execution

```python
async for event in graph.astream(
    {"workflow_inputs": {...}},
    config={"configurable": {"thread_id": run_id}}
):
    # event = {
    #   "event": "on_chain_start|on_chain_stream|on_chain_end",
    #   "data": {...}
    # }
```

Yields events as workflow executes.

### 3. Async Background Execution

Orchestrated by Worker (see `src/seer/worker/`):

```python
# API enqueues job
await execute_saved_workflow_task.kiq(run_id)

# Worker executes
async def execute_saved_workflow(run_id):
    run = await WorkflowRun.get(id=run_id)
    graph = await compiler.compile(run.user, run.spec)
    result = await graph.ainvoke(...)
    await WorkflowRun.filter(id=run_id).update(status=COMPLETED, outputs=result)
```

## Error Handling

### Error Hierarchy (errors.py)

```
WorkflowCompilerError (base)
├── ValidationPhaseError (Stage 1-3: structural/reference validation)
├── TypeEnvironmentError (Stage 2: type env construction)
├── LoweringError (Stage 4: lowering failures)
└── ExecutionError (runtime: tool execution, schema validation)
    └── EvaluationError (runtime: expression evaluation)
```

### Compilation Errors

```python
from seer.core.errors import WorkflowCompilerError

try:
    graph = await compiler.compile(user, spec)
except WorkflowCompilerError as e:
    # e.message = "Node 'node_1' references unknown node 'node_0'"
    # e.node_id = "node_1"
    _raise_problem(
        type_uri=COMPILE_PROBLEM,
        title="Compilation failed",
        detail=str(e),
        status=400
    )
```

### Runtime Errors

```python
try:
    result = await graph.ainvoke(inputs)
except Exception as e:
    # Node execution errors, tool failures, etc.
    _raise_problem(
        type_uri=RUN_PROBLEM,
        title="Execution failed",
        detail=str(e),
        status=500
    )
```

## LangGraph Integration

**Why LangGraph?**
- Built-in checkpointing (resume/replay)
- Streaming execution
- Human-in-the-loop support (interrupts)
- Mature graph orchestration
- LangChain tool ecosystem

**State Graph Pattern:**
```python
from langgraph.graph import StateGraph

graph = StateGraph(WorkflowState)
graph.add_node("node_1", execute_node_1)
graph.add_node("node_2", execute_node_2)
graph.add_edge("node_1", "node_2")
graph.set_entry_point("node_1")
compiled = graph.compile(checkpointer=checkpointer)
```

## Adding New Features

### Adding a New Node Type

1. **Define the node model** in `schema/models.py`
   ```python
   class MyNewNode(Node):
       type: Literal["my_new_node"]
       config: MyNewNodeConfig
   ```

2. **Add to Node union type** in `schema/models.py`
   ```python
   Node = Union[ToolNode, LLMNode, ..., MyNewNode]
   ```

3. **Implement executor** in `runtime/nodes.py`
   ```python
   async def execute_my_new_node(state: WorkflowState, node_config: dict):
       # Implementation
       return {f"node_{node_id}_output": result}
   ```

4. **Register in compiler** in `compiler/compiler.py`
   ```python
   if node_type == "my_new_node":
       graph.add_node(node_id, lambda s: execute_my_new_node(s, node_config))
   ```

5. **Add validation** in `compiler/validate_refs.py` if needed

6. **Write tests**
   - Unit tests for executor
   - Integration tests with full workflow spec
   - Validation error tests

### Adding Validation Logic

1. **Compilation-time validation:** Add to `compiler/validate_refs.py`
2. **Runtime validation:** Add to `runtime/input_validation.py` or `runtime/validate_output.py`
3. **Schema validation:** Use `schema/jsonschema_adapter.py` utilities

## Testing

### Running Tests

```bash
# Run all workflow compiler tests
uv run pytest tests/core/

# Run specific test files
uv run pytest tests/core/test_compiler.py
uv run pytest tests/core/test_nodes.py

# Run with verbose output
uv run pytest -v tests/core/

# Run tests for specific node type
uv run pytest tests/core/ -k "test_tool_node"
```

### Test Structure

- Unit tests for individual executors
- Full JSON spec tests for integration
- Validation error tests
- Regression tests for bug fixes

**Important:** For every change related to `src/seer/core`, add concerned unit tests and full JSON spec tests, then validate that changes pass all tests (regression testing).

## Common Debugging Scenarios

### Compilation Fails

1. **Check validation stage** - Look at error type:
   - `ValidationPhaseError`: Structural/reference issue → Check `compiler/parse.py` or `compiler/validate_refs.py`
   - `TypeEnvironmentError`: Type environment issue → Check `compiler/type_env.py`
   - `LoweringError`: Control flow issue → Check `compiler/lower_control_flow.py`

2. **Enable verbose logging**
   ```python
   import logging
   logging.getLogger("seer.core").setLevel(logging.DEBUG)
   ```

3. **Examine spec structure** - Validate JSON structure matches schema models

### Runtime Execution Fails

1. **Check node executor** - Add logging to specific executor in `runtime/nodes.py`
2. **Inspect state** - Log workflow state before/after node execution
3. **Validate expressions** - Check `${...}` references resolve correctly
4. **Check tool execution** - Verify tool registry and credentials

### Cache Issues

**Problem:** Compiled graph cache doesn't invalidate on spec changes

**Workaround:**
```python
# Clear cache manually
compiler._cache.clear()

# Or restart the service
```

## Best Practices

1. **Separation of concerns**: Keep compiler pure (no DB/HTTP dependencies)
2. **Error messages**: Provide clear, actionable error messages with context
3. **Type safety**: Use Pydantic models for all spec validation
4. **Testing**: Add both unit and integration tests for all changes
5. **Documentation**: Update README.md when adding new node types or features
6. **Validation layers**: Use 5-stage compilation pipeline for progressive error catching
7. **State immutability**: Nodes should return new state updates, not mutate existing state
8. **Expression syntax**: Use `${...}` for references, maintain consistent syntax

## Integration Points

### API Layer → Compiler

**API Layer** (`src/seer/api/workflows/`):
- Handles HTTP requests
- Persists workflows to database
- Manages authentication
- Calls compiler for execution

**Compiler** (`src/seer/core/`):
- Pure compilation logic
- No awareness of API layer
- Testable in isolation

**Flow:**
```python
# API endpoint
@router.post("/workflows/{id}/run")
async def run_workflow(id: str):
    workflow = await Workflow.get(id=id)
    compiler = WorkflowCompilerSingleton.instance()
    graph = await compiler.compile(user, workflow.spec)
    result = await graph.ainvoke(workflow.inputs)
    return result
```

### Compiler → Tools

**Tools** (`src/seer/tools/`):
- Provides BaseTool registry
- Handles credential resolution
- Executes tool actions

**Compiler**:
- Calls tools via tool executor
- Passes connection_id for credentials
- Handles tool execution errors

### Compiler → Worker

**Worker** (`src/seer/worker/`):
- Background task execution
- Trigger polling
- Async workflow runs

**Compiler**:
- Executed by worker for async runs
- Uses checkpointing for long-running workflows

## Known Issues & Future Improvements

- [ ] **Cache invalidation**: Compiled graph cache doesn't invalidate on spec changes
- [ ] **Node executor separation**: 587-line `nodes.py` could be split by node type
- [ ] **Error context**: Compilation errors should include line/field context
- [ ] **Validation layer**: More robust pre-compilation validation (detect cycles, unreachable nodes)
- [ ] **Performance**: Optimize compilation for large workflows (100+ nodes)
- [ ] **Type inference**: Better type inference for expressions

## Quick Checklist

When working on workflow compiler:
- [ ] Read `src/seer/core/README.md` for architecture overview
- [ ] Check existing node executors in `runtime/nodes.py` for patterns
- [ ] Use schema models from `schema/models.py` for type safety
- [ ] Add validation in appropriate compilation stage
- [ ] Write unit tests and full JSON spec tests
- [ ] Update error handling with clear messages
- [ ] Verify no database/HTTP dependencies in core
- [ ] Test with checkpointing enabled
- [ ] Validate expression evaluation works correctly
- [ ] Run full test suite: `uv run pytest tests/core/`

## Related Documentation

- [Workflows API](../../api/workflows/README.md) - Workflow CRUD & execution orchestration
- [Database Models](../../database/README.md) - Workflow/WorkflowRun persistence
- [Tools System](../../tools/README.md) - Tool execution in tool nodes
- [Worker](../../worker/README.md) - Background async execution
- [Validation Skill](./../workflow-validation/SKILL.md) - Detailed validation guide

## Example: Complete Workflow Execution

```python
# 1. Define spec
spec = {
    "nodes": {
        "send_email": {
            "type": "tool",
            "config": {
                "tool": "gmail_send_email",
                "inputs": {
                    "to": {"type": "workflow_input", "key": "recipient"},
                    "subject": {"type": "static", "value": "Hello"},
                    "body": {"type": "static", "value": "Test email"}
                },
                "connection_id": "conn_123"
            }
        }
    },
    "edges": [],
    "entry_point": "send_email"
}

# 2. Compile
compiler = WorkflowCompilerSingleton.instance()
graph = await compiler.compile(user, spec, checkpointer=checkpointer)

# 3. Execute
result = await graph.ainvoke(
    {"workflow_inputs": {"recipient": "user@example.com"}},
    config={"configurable": {"thread_id": "run_123"}}
)

# 4. Result
# {
#   "node_send_email_output": {"message_id": "msg_xyz"},
#   "workflow_inputs": {"recipient": "user@example.com"},
#   ...
# }
```

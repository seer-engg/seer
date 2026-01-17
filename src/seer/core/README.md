# Workflow Compiler

**Purpose**: Compile workflow specs (DSL) into executable LangGraph graphs and orchestrate runtime execution.

## Architecture

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

## Core Components

### 1. Workflow Compiler (`workflow_compiler/compiler.py`)

**Responsibilities**: Transform WorkflowSpec → executable LangGraph

```python
from seer.core.runtime.global_compiler import WorkflowCompilerSingleton

compiler = WorkflowCompilerSingleton.instance()
graph = await compiler.compile(user, spec_dict, checkpointer=None)
```

**Compilation Steps**:
1. **Validate spec**: Check required fields, node types exist, edges valid
2. **Build state graph**: Create `StateGraph` with workflow state schema
3. **Add nodes**: Register node executors for each node in spec
4. **Add edges**: Connect nodes with conditional/normal edges
5. **Set entry/finish**: Configure graph entry point and END state
6. **Compile**: `graph.compile(checkpointer=...)`

**Caching**: Singleton maintains compiled graph cache (keyed by spec hash).

### 2. Runtime Node Executors (`runtime/nodes.py`)

**Node Types**:
- **Tool Node**: Execute registered tools (e.g., Gmail, GitHub)
- **LLM Node**: Call LLM with prompts/messages
- **Code Node**: Execute Python code sandbox
- **Conditional Node**: Branch based on expression evaluation
- **Transform Node**: Map/transform data
- **Trigger Node**: Workflow entry point (receives trigger events)
- **Input Node**: Workflow inputs
- **Output Node**: Workflow outputs
- **Extract Node**: Extract structured data from text

**Example Tool Node Execution**:
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

### 3. State Management (`schema/models.py`)

**WorkflowState**: LangGraph state dict containing:

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

**State Updates**: Nodes return partial state updates merged by LangGraph.

### 4. Checkpointing

**Purpose**: Save workflow state after each node execution for:
- Resume interrupted workflows
- Replay execution
- Debugging state transitions

**Checkpointer Types**:
- `MemorySaver`: In-memory (dev/testing)
- `DatabaseCheckpointer`: PostgreSQL (production)

**Usage**:
```python
checkpointer = await get_checkpointer()
graph = await compiler.compile(user, spec, checkpointer=checkpointer)
```

## Workflow Spec (DSL)

### Structure

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
        "prompt": "Summarize: {{node_1.output}}"
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

### Node Input Resolution

**Types**:
- `static`: Hardcoded value
- `input`: Reference another node's output
- `expression`: Evaluate expression (e.g., `{{node_1.output.result}}`)
- `workflow_input`: Reference workflow input parameter

**Resolution**:
```python
def resolve_node_inputs(state, input_config):
    if input_config["type"] == "static":
        return input_config["value"]
    elif input_config["type"] == "input":
        node_id = input_config["node_id"]
        return state[f"node_{node_id}_output"]
    # ...
```

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

Orchestrated by Worker (see [Worker README](../worker/README.md)):

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

## Key Files

```
workflow_compiler/
├── compiler.py                 # Main compiler logic
├── schema/
│   └── models.py              # WorkflowSpec, NodeConfig Pydantic models
├── runtime/
│   ├── global_compiler.py     # WorkflowCompilerSingleton
│   ├── nodes.py               # Node executors (587 lines)
│   ├── execution.py           # Execution helpers
│   └── state.py               # State management utilities
└── errors.py                  # WorkflowCompilerError hierarchy
```

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

## Separation: Compiler vs API Layer

**Workflow Compiler** (this module):
- ✅ Pure compilation & runtime logic
- ✅ No database access
- ✅ No HTTP/FastAPI dependencies
- ✅ Testable in isolation

**API Layer** (`/api/workflows`):
- Database persistence (Workflow, WorkflowRun, WorkflowVersion)
- HTTP endpoint handling
- User authentication
- Analytics tracking
- Orchestrates compiler + persistence

**Clean boundary**: API layer calls compiler, compiler is unaware of API layer.

## LangGraph Integration

Seer uses [LangGraph](https://github.com/langchain-ai/langgraph) for workflow execution:

**Why LangGraph?**
- Built-in checkpointing (resume/replay)
- Streaming execution
- Human-in-the-loop support (interrupts)
- Mature graph orchestration
- LangChain tool ecosystem

**State Graph Pattern**:
```python
from langgraph.graph import StateGraph

graph = StateGraph(WorkflowState)
graph.add_node("node_1", execute_node_1)
graph.add_node("node_2", execute_node_2)
graph.add_edge("node_1", "node_2")
graph.set_entry_point("node_1")
compiled = graph.compile(checkpointer=checkpointer)
```

## Known Issues & Improvements Planned

- [ ] **Cache invalidation**: Compiled graph cache doesn't invalidate on spec changes
- [ ] **Node executor separation**: 587-line `nodes.py` could be split by node type
- [ ] **Error context**: Compilation errors should include line/field context
- [ ] **Validation layer**: More robust pre-compilation validation (detect cycles, unreachable nodes)

## Related Documentation

- [Workflows API](../api/workflows/README.md) - Workflow CRUD & execution orchestration
- [Database Models](../shared/database/README.md) - Workflow/WorkflowRun persistence
- [Tools System](../shared/tools/README.md) - Tool execution in tool nodes
- [Worker](../worker/README.md) - Background async execution

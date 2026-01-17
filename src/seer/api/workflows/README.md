# Workflows API Module

**Purpose**: Core workflow management system - CRUD operations, execution, versioning, triggers, and run history.

## Architecture

```
HTTP Request (router.py)
    ↓
Service Layer (services/)
    ├── lifecycle.py     - Create, read, update, delete, version management
    ├── execution.py     - Workflow run execution (sync/async)
    ├── triggers.py      - Trigger subscriptions
    ├── history.py       - Run history & checkpoint traversal
    ├── catalog.py       - Tool/model/node type catalogs
    ├── expression.py    - Expression validation
    └── shared.py        - Common utilities (_raise_problem, _get_workflow)
    ↓
Workflow Compiler (/workflow_compiler)
    ├── Compilation      - Validate & transform spec
    └── Runtime          - LangGraph execution engine
    ↓
Worker (/worker/tasks/workflows.py) - Async execution via Taskiq
```

## Data Model

### Workflow Lifecycle

```
Workflow (workflow_id, name, description)
    ├── WorkflowDraft (mutable spec, revision counter)
    │   ↓ Create version
    ├── WorkflowVersion[] (immutable spec snapshot, status)
    │   ├── DRAFT (test runs)
    │   ├── PUBLISHED (active version)
    │   └── ARCHIVED (old versions)
    └── WorkflowRun[] (execution records)
        ├── inputs, outputs, status
        └── checkpoints (LangGraph state snapshots)
```

**Key relationships**:
- `Workflow` ➜ 1 `WorkflowDraft` (current editable spec)
- `Workflow` ➜ N `WorkflowVersion` (immutable snapshots)
- `Workflow` ➜ `published_version` (active version for triggers)
- `WorkflowVersion` ➜ N `WorkflowRun` (execution history)

See [Database Models](../../shared/database/README.md) for full schema.

## Service Layer Modules

### 1. Lifecycle Service (`services/lifecycle.py` - 507 lines)

**Responsibilities**: Workflow CRUD, draft management, versioning

**Key Functions**:
```python
# Workflow CRUD
await create_workflow(user, payload)
await get_workflow(user, workflow_id)
await list_workflows(user)
await update_workflow(user, workflow_id, payload)
await delete_workflow(user, workflow_id)

# Draft management
await update_draft(user, workflow_id, payload)

# Version management
await create_version(user, workflow_id, payload)
await publish_version(user, workflow_id, version_id)
await list_versions(user, workflow_id)
```

**Draft Update Flow**:
1. Validate `WorkflowSpec` via Pydantic
2. Increment draft revision
3. Update `WorkflowDraft.spec`
4. Return updated workflow

**Versioning Flow**:
1. Hash draft spec (`_hash_spec`)
2. Check if version already exists (deduplication)
3. Create `WorkflowVersion` (status=DRAFT or PUBLISHED)
4. Optionally set as `published_version`

### 2. Execution Service (`services/execution.py` - 457 lines)

**Responsibilities**: Workflow run orchestration, sync/async execution

**Key Functions**:
```python
# Execution
await run_saved_workflow(user, workflow_id, payload, source=MANUAL)
await run_adhoc_workflow(user, spec_dict, payload)
await resume_workflow(user, run_id, payload)

# Run management
await get_run(user, run_id)
await list_runs(user, workflow_id=None)
await cancel_run(user, run_id)
```

**Execution Flow (Synchronous)**:
1. Fetch workflow + draft
2. Create `WorkflowVersion` (if not exists)
3. Create `WorkflowRun` record (status=QUEUED)
4. Compile workflow via `WorkflowCompiler`
5. Execute via `graph.ainvoke()` or `graph.astream()`
6. Update run record (status=COMPLETED/FAILED)
7. Track analytics (`workflow_executed`)

**Execution Flow (Asynchronous)**:
1. Same steps 1-3 as synchronous
2. Enqueue Taskiq job: `execute_saved_workflow_task.kiq(run_id)`
3. Return immediately with `status=QUEUED`
4. Worker picks up job, executes in background

**Streaming Execution**:
```python
# Returns async generator
async for chunk in run_saved_workflow(..., stream=True):
    # chunk = {"event": "node", "data": {...}}
```

### 3. Triggers Service (`services/triggers.py` - 518 lines)

**Responsibilities**: Trigger catalog, subscription management

**Key Functions**:
```python
# Catalog
await list_triggers()

# Subscriptions
await create_trigger_subscription(user, payload)
await update_trigger_subscription(user, subscription_id, payload)
await delete_trigger_subscription(user, subscription_id)
await test_trigger_subscription(user, subscription_id, payload)
```

**Subscription Lifecycle**:
1. User creates subscription linking workflow ➜ trigger
2. Polling engine detects events (see [Triggers](../triggers/README.md))
3. Events trigger workflow execution via `execution.run_saved_workflow(..., source=TRIGGER)`

### 4. History Service (`services/history.py` - 517 lines)

**Responsibilities**: Run history, checkpoint traversal, output retrieval

**Key Functions**:
```python
await list_runs(user, workflow_id=None)
await get_run(user, run_id)
await get_run_output(user, run_id)
await list_run_checkpoints(user, run_id)
await get_checkpoint_state(user, run_id, checkpoint_id)
```

**Checkpoint System**:
- LangGraph saves state after each node execution
- Checkpoints stored in `workflow_run_checkpoints` table
- Enables: resume, replay, state inspection, debugging

**Traversal**:
```python
# Get all checkpoints for run
checkpoints = await list_run_checkpoints(user, run_id)

# Inspect state at specific checkpoint
state = await get_checkpoint_state(user, run_id, checkpoint_id)
# state = {"messages": [...], "node_outputs": {...}}
```

### 5. Catalog Service (`services/catalog.py` - 281 lines)

**Responsibilities**: Expose tool/model/node type registries to frontend

**Key Functions**:
```python
await list_tools()         # ToolRegistry
await list_models()        # ModelRegistry
await list_node_types()    # NodeTypeRegistry
```

**Response Format**:
```json
{
  "tools": [
    {
      "name": "gmail_send_email",
      "description": "...",
      "parameters": {...},
      "required_scopes": [...]
    }
  ],
  "models": [...],
  "node_types": [...]
}
```

### 6. Shared Utilities (`services/shared.py` - 108 lines)

**Common helpers used across all services**:

```python
# Error handling (RFC 7807)
_raise_problem(
    type_uri="https://seer.errors/workflows/validation",
    title="Workflow not found",
    detail=f"Workflow '{workflow_id}' not found",
    status=404
)

# Workflow fetching with validation
workflow = await _get_workflow(user, workflow_id)

# Spec serialization
spec_dict = _spec_to_dict(spec)

# Spec hashing (version deduplication)
spec_hash = _hash_spec(spec_dict)

# Run config building (ensures thread_id consistency)
config = _build_run_config(run, user_config)
```

## Execution Paths

### Path 1: Manual Execution (API)

```
User → POST /v1/workflows/{id}/runs
    ↓
router.run_saved_workflow()
    ↓
services.execution.run_saved_workflow(user, workflow_id, payload, source=MANUAL)
    ↓
WorkflowCompiler.compile(spec)
    ↓
graph.ainvoke() or graph.astream()
    ↓
Return outputs + status
```

### Path 2: Async Execution (Worker)

```
User → POST /v1/workflows/{id}/runs?mode=async
    ↓
services.execution.run_saved_workflow(..., mode=async)
    ↓
Create WorkflowRun (status=QUEUED)
    ↓
execute_saved_workflow_task.kiq(run_id)  # Enqueue Taskiq job
    ↓
Return {run_id, status: "queued"}

[Background Worker]
    ↓
worker/tasks/workflows.py:execute_saved_workflow()
    ↓
Fetch run, compile, execute
    ↓
Update WorkflowRun (status=COMPLETED/FAILED)
```

### Path 3: Trigger Execution (Polling)

```
Polling Engine (background)
    ↓
Detects TriggerEvent
    ↓
services.execution.run_saved_workflow(
    user,
    workflow_id,
    payload={"trigger_event": {...}},
    source=TRIGGER
)
    ↓
Same as Path 1 (sync execution)
```

### Path 4: Chat Agent Execution

```
User → POST /v1/agent/sessions/{id}/messages
    ↓
LangGraph agent (agents/workflow/router.py)
    ↓
Agent creates workflow spec
    ↓
services.execution.run_adhoc_workflow(user, spec, inputs)
    ↓
Execute ephemeral workflow (no Workflow record)
```

## API Endpoints

### Workflow CRUD

- `POST /v1/workflows` - Create workflow
- `GET /v1/workflows` - List workflows
- `GET /v1/workflows/{id}` - Get workflow
- `PATCH /v1/workflows/{id}` - Update metadata (name, description)
- `DELETE /v1/workflows/{id}` - Delete workflow

### Draft Management

- `PATCH /v1/workflows/{id}/draft` - Update draft spec

### Version Management

- `POST /v1/workflows/{id}/versions` - Create version
- `GET /v1/workflows/{id}/versions` - List versions
- `POST /v1/workflows/{id}/versions/{version_id}/publish` - Publish version
- `GET /v1/workflows/{id}/versions/{version_id}` - Get version

### Execution

- `POST /v1/workflows/{id}/runs` - Execute workflow
  - Query params: `mode=sync|async`, `stream=true|false`
- `POST /v1/workflows/{id}/runs/adhoc` - Execute adhoc spec
- `POST /v1/runs/{run_id}/resume` - Resume interrupted run

### Run History

- `GET /v1/runs` - List all runs (optionally filter by workflow_id)
- `GET /v1/runs/{run_id}` - Get run details
- `GET /v1/runs/{run_id}/output` - Get run output
- `POST /v1/runs/{run_id}/cancel` - Cancel running workflow

### Checkpoints

- `GET /v1/runs/{run_id}/checkpoints` - List checkpoints
- `GET /v1/runs/{run_id}/checkpoints/{checkpoint_id}` - Get checkpoint state

### Catalogs

- `GET /v1/builder/node-types` - List available node types
- `GET /v1/registries/tools` - List available tools
- `GET /v1/registries/models` - List available models

## Error Handling

### Standard Pattern

All services use `_raise_problem` (RFC 7807):

```python
# Not found
_raise_problem(
    type_uri=f"{PROBLEM_BASE}/validation",
    title="Workflow not found",
    detail=f"Workflow '{workflow_id}' not found",
    status=404
)

# Validation error
_raise_problem(
    type_uri=VALIDATION_PROBLEM,
    title="Invalid workflow spec",
    detail="Workflow must have at least one node",
    status=400,
    errors=[ProblemError(field="nodes", message="Required")]
)

# Compilation error
_raise_problem(
    type_uri=COMPILE_PROBLEM,
    title="Workflow compilation failed",
    detail=str(compiler_error),
    status=400,
    errors=[...]
)

# Runtime error
_raise_problem(
    type_uri=RUN_PROBLEM,
    title="Workflow execution failed",
    detail=str(runtime_error),
    status=500
)
```

### Error URIs

```python
PROBLEM_BASE = "https://seer.errors/workflows"
VALIDATION_PROBLEM = f"{PROBLEM_BASE}/validation"
COMPILE_PROBLEM = f"{PROBLEM_BASE}/compile"
RUN_PROBLEM = f"{PROBLEM_BASE}/run"
```

## Analytics Events

```python
# Workflow execution
analytics.track(user.id, "workflow_executed", {
    "workflow_id": workflow.public_id,
    "mode": "sync|async",
    "source": "manual|trigger|agent",
    "duration_ms": 123.45,
    "status": "completed|failed"
})

# Workflow created
analytics.track(user.id, "workflow_created", {
    "workflow_id": workflow.public_id
})

# Version published
analytics.track(user.id, "workflow_version_published", {
    "workflow_id": workflow.public_id,
    "version_number": version.version_number
})
```

## Common Patterns

### Fetching Workflow with Prefetch

```python
workflow = await Workflow.filter(id=pk, user=user)\
    .prefetch_related("draft", "published_version")\
    .first()
```

### Spec Validation

```python
try:
    spec = WorkflowSpec.model_validate(payload.spec)
except ValidationError as e:
    _raise_problem(
        type_uri=VALIDATION_PROBLEM,
        title="Invalid workflow spec",
        detail="Spec validation failed",
        status=400,
        errors=[...]
    )
```

### Version Deduplication

```python
spec_hash = _hash_spec(spec_dict)
existing = await WorkflowVersion.filter(
    workflow=workflow,
    spec_hash=spec_hash,
    status=WorkflowVersionStatus.DRAFT
).first()
if existing:
    return existing  # Don't create duplicate
```

## Known Issues & Improvements Planned

- [ ] **Service consolidation**: 7 files → 4 (reduce circular imports)
- [ ] **Duplicate helpers**: `_require_user` duplicated in router.py and agents/workflow/router.py
- [ ] **Mixed responsibilities**: execution.py handles orchestration + analytics + persistence (should split)
- [ ] **Repository layer**: Abstract ORM queries for testability
- [ ] **Analytics middleware**: Centralize event tracking vs scattered calls

## Related Documentation

- [API Layer](../README.md) - Router pattern, error handling conventions
- [Workflow Compiler](../../workflow_compiler/README.md) - Compilation & runtime engine
- [Database Models](../../shared/database/README.md) - Workflow/Draft/Version/Run schemas
- [Triggers](../triggers/README.md) - Polling system & trigger subscriptions
- [Agents](../../agents/README.md) - Chat-based workflow creation
- [Worker](../../worker/README.md) - Background job execution

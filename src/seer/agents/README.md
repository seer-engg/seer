# Agents Module

**Purpose**: LangGraph-based chat agent for conversational workflow creation, editing, and execution.

## Architecture

```
User Message
    ↓
Chat Session (persisted in checkpointer)
    ↓
LangGraph Agent (agent graph with tools)
    ├── Analyze user intent
    ├── Read/modify workflow spec
    ├── Execute workflow
    └── Propose changes
    ↓
Stream response + proposals
```

## Core Components

### 1. Workflow Chat Agent (`workflow/router.py` - 1,049 lines)

**Purpose**: Conversational interface for workflow management

**Key Endpoints**:
- `POST /v1/agent/sessions` - Create chat session
- `POST /v1/agent/sessions/{id}/messages` - Send message
- `GET /v1/agent/sessions/{id}/history` - Get conversation history
- `POST /v1/agent/proposals/{id}/apply` - Apply proposed changes

**Agent Capabilities**:
- Create workflows from natural language
- Modify existing workflows
- Execute workflows adhoc
- Explain workflow logic
- Debug execution issues

### 2. Agent Graph Architecture

**LangGraph Agent**:
```python
# Simplified structure
agent = create_react_agent(
    model=llm,
    tools=[
        read_workflow_tool,
        update_workflow_tool,
        execute_workflow_tool,
        list_tools_catalog,
        create_proposal_tool
    ],
    checkpointer=checkpointer
)
```

**State**:
```python
{
    "messages": [...],              # Conversation history
    "workflow_id": "wf_abc",        # Current workflow context
    "proposals": [...],             # Pending change proposals
    "execution_results": {...}      # Adhoc execution outputs
}
```

### 3. Checkpointer System (`checkpointer.py`)

**Purpose**: Persist agent conversation state for resumability

**Implementations**:
- `MemoryCheckpointer` - In-memory (dev)
- `DatabaseCheckpointer` - PostgreSQL (production)

**Database Schema**:
```python
class AgentCheckpoint(Model):
    thread_id: str              # Session ID
    checkpoint_id: str          # State snapshot ID
    parent_checkpoint_id: str   # Previous state
    checkpoint_data: dict       # Serialized state
    created_at: datetime
```

**Usage**:
```python
from seer.api.agents.checkpointer import get_checkpointer

checkpointer = await get_checkpointer()

# Agent automatically saves checkpoints after each turn
agent = create_agent(..., checkpointer=checkpointer)

# Resume conversation
result = await agent.ainvoke(
    {"messages": [new_message]},
    config={"configurable": {"thread_id": session_id}}
)
```

**Health Checks**: Router includes extensive checkpointer health checking with retry logic (lines 286-500+).

### 4. Workflow Proposals (`workflow/router.py`)

**Purpose**: Preview workflow changes before applying

**Flow**:
1. User: "Add a step to send email notifications"
2. Agent analyzes current workflow spec
3. Agent creates `WorkflowProposal` with modified spec
4. Frontend displays diff/preview
5. User approves → `POST /v1/agent/proposals/{id}/apply`
6. Proposal applied to workflow draft

**Proposal Model**:
```python
class WorkflowProposal(Model):
    id: int
    session_id: str
    workflow: ForeignKey[Workflow]
    proposed_spec: dict         # Modified workflow spec
    description: str            # Natural language explanation
    created_at: datetime
```

**Apply Proposal**:
```python
POST /v1/agent/proposals/{proposal_id}/apply

# Updates WorkflowDraft.spec with proposed_spec
# Increments draft revision
```

## Chat Sessions

### Create Session

```
POST /v1/agent/sessions
{
  "workflow_id": "wf_abc123"  # Optional: context for existing workflow
}

Response:
{
  "session_id": "sess_xyz",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Send Message

```
POST /v1/agent/sessions/{session_id}/messages
{
  "message": "Create a workflow that sends me an email every morning with unread Gmail count",
  "stream": true
}

Response (streaming):
event: message
data: {"content": "I'll create a workflow with a Gmail tool..."}

event: proposal
data: {"proposal_id": 123, "description": "Created workflow with Gmail count tool"}
```

### Get History

```
GET /v1/agent/sessions/{session_id}/history

Response:
{
  "messages": [
    {"role": "user", "content": "Create a workflow..."},
    {"role": "assistant", "content": "I'll help you..."},
    ...
  ]
}
```

## Agent Tools

**Tools available to LangGraph agent**:

### 1. `read_workflow`
```python
async def read_workflow(workflow_id: str) -> dict:
    """Get current workflow spec"""
    workflow = await Workflow.get(workflow_id=workflow_id)
    return workflow.draft.spec
```

### 2. `update_workflow`
```python
async def update_workflow(workflow_id: str, spec: dict, description: str):
    """Create proposal with modified spec"""
    proposal = await WorkflowProposal.create(
        workflow_id=workflow_id,
        proposed_spec=spec,
        description=description
    )
    return {"proposal_id": proposal.id}
```

### 3. `execute_adhoc_workflow`
```python
async def execute_adhoc_workflow(spec: dict, inputs: dict):
    """Execute workflow without saving"""
    result = await run_adhoc_workflow(user, spec, inputs)
    return result
```

### 4. `list_tools_catalog`
```python
async def list_tools_catalog():
    """List available tools for workflow building"""
    return ToolRegistry.list_all()
```

## Example Conversations

### Create Workflow

```
User: Create a workflow that checks my Gmail every hour and sends me Slack notifications for important emails

Agent: I'll create a workflow with:
1. Trigger: Cron schedule (hourly)
2. Gmail tool: List messages with query "is:important is:unread"
3. Slack tool: Send notification

[Creates proposal]

User: Apply this

Agent: Applied! Your workflow is ready to publish.
```

### Modify Workflow

```
User: Add a filter to only notify me about emails from my boss

Agent: I'll modify the Gmail query to include from:boss@company.com
[Shows diff of spec change]

User: Looks good

Agent: Updated! The workflow now filters for your boss's emails.
```

### Debug Execution

```
User: My workflow failed with "Invalid credentials"

Agent: Let me check the workflow spec... I see you're using a Gmail tool but haven't connected a Google account. You need to:
1. Go to Integrations
2. Connect your Google account
3. Update the workflow to use that connection

Would you like me to show you how to add the connection ID to the workflow?
```

## Implementation Details

### Session Management

**Thread ID = Session ID**: LangGraph thread_id used as session identifier

```python
config = {
    "configurable": {
        "thread_id": session_id,
        "user": user
    }
}
```

### Checkpointer Health & Retry Logic

Router includes complex retry logic for database checkpointer issues (lines 286-500+):

```python
# Retry with exponential backoff
for attempt in range(max_retries):
    try:
        result = await agent.ainvoke(...)
        break
    except CheckpointerError:
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)
            continue
        raise
```

**Known issue**: Checkpointer health logic should be extracted to separate module.

### Proposal Creation & Validation

```python
# Agent tool creates proposal
proposal = await WorkflowProposal.create(
    session_id=session_id,
    workflow=workflow,
    proposed_spec=modified_spec,
    description="Added email notification step"
)

# Validate proposed spec before storing
try:
    WorkflowSpec.model_validate(proposed_spec)
except ValidationError as e:
    return {"error": "Invalid workflow spec"}
```

## Streaming Response

**Server-Sent Events (SSE)** format:

```
event: message
data: {"role": "assistant", "content": "Let me help you..."}

event: message_chunk
data: {"content": " create"}

event: proposal
data: {"proposal_id": 123, "description": "..."}

event: done
data: {}
```

**Frontend handling**:
```javascript
const eventSource = new EventSource(`/v1/agent/sessions/${id}/messages`)
eventSource.addEventListener('message', (event) => {
  const data = JSON.parse(event.data)
  appendToChat(data.content)
})
```

## Analytics

```python
analytics.track(user.id, "agent_message_sent", {
    "session_id": session_id,
    "message_length": len(message),
    "has_workflow_context": workflow_id is not None
})

analytics.track(user.id, "agent_proposal_created", {
    "proposal_id": proposal.id,
    "workflow_id": workflow.workflow_id
})

analytics.track(user.id, "agent_proposal_applied", {
    "proposal_id": proposal.id
})
```

## Known Issues & Improvements Planned

- [ ] **File bloat**: `workflow/router.py` is 1,049 lines - split into:
  - `chat_endpoints.py` (HTTP handling)
  - `proposal_management.py` (proposal CRUD)
  - `session_management.py` (session lifecycle)
  - `checkpointer_health.py` (health checks & retry logic)
- [ ] **Error handling**: Uses raw `HTTPException` (should use `_raise_problem`)
- [ ] **Checkpointer health**: Complex retry logic should be middleware
- [ ] **Agent tools**: Should be more discoverable and documented
- [ ] **Proposal diff**: No structured diff view (just full spec)

## Related Documentation

- [Workflows API](../api/workflows/README.md) - Workflow CRUD & execution
- [Workflow Compiler](../workflow_compiler/README.md) - Spec validation & compilation
- [Database Models](../shared/database/README.md) - WorkflowProposal, AgentCheckpoint schemas
- [Tools System](../shared/tools/README.md) - Tools available for workflow building

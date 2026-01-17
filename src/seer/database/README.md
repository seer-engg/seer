# Database Models

**Purpose**: Tortoise ORM data models defining application schema and relationships.

## Architecture

**ORM**: Tortoise ORM (async SQLAlchemy alternative)
**Database**: PostgreSQL
**Migrations**: Aerich (Tortoise migration tool)

## Model Files

```
shared/database/
├── models.py                   # Core models (User, ApiKey)
├── workflow_models.py          # Workflow system models
├── models_oauth.py             # OAuth connections
├── models_integrations.py      # Integration resources & secrets
└── models_triggers.py          # Trigger subscriptions
```

## Core Entities

### 1. User (`models.py`)

```python
class User(Model):
    id: int                     # Primary key
    user_id: str                # External user ID (from Clerk)
    email: str
    created_at: datetime
```

**Relationships**:
- 1:N Workflow
- 1:N OAuthConnection
- 1:N TriggerSubscription

### 2. Workflow System (`workflow_models.py`)

#### Workflow

```python
class Workflow(Model):
    id: int                     # Internal PK
    workflow_id: str            # Public ID (wf_...)
    user: ForeignKey[User]
    name: str
    description: str
    created_at: datetime
    updated_at: datetime

    # Relationships
    draft: ReverseRelation[WorkflowDraft]                # 1:1
    published_version: ForeignKey[WorkflowVersion]       # 1:1 (nullable)
    versions: ReverseRelation[WorkflowVersion]           # 1:N
    runs: ReverseRelation[WorkflowRun]                   # 1:N
```

**Public ID Format**: `wf_{uuid}` (e.g., `wf_a1b2c3d4`)

#### WorkflowDraft

```python
class WorkflowDraft(Model):
    id: int
    workflow: OneToOne[Workflow]
    spec: dict                  # WorkflowSpec JSON
    revision: int               # Incremented on each update
    updated_at: datetime
```

**Lifecycle**:
- Created when Workflow is created
- Updated via `PATCH /v1/workflows/{id}/draft`
- Immutable snapshots created as WorkflowVersion

#### WorkflowVersion

```python
class WorkflowVersion(Model):
    id: int
    workflow: ForeignKey[Workflow]
    version_number: int         # Auto-incremented
    status: WorkflowVersionStatus  # DRAFT, PUBLISHED, ARCHIVED
    spec: dict                  # Immutable spec snapshot
    spec_hash: str              # SHA256 hash (deduplication)
    created_from_draft_revision: int
    created_by: ForeignKey[User]
    created_at: datetime

    # Relationships
    runs: ReverseRelation[WorkflowRun]
```

**Deduplication**: Same spec hash → reuse existing version

**Status Transitions**:
- DRAFT → PUBLISHED (via publish endpoint)
- PUBLISHED → ARCHIVED (when new version published)

#### WorkflowRun

```python
class WorkflowRun(Model):
    id: int
    run_id: str                 # Public ID (run_...)
    thread_id: str              # LangGraph thread ID (same as run_id)
    user: ForeignKey[User]
    workflow: ForeignKey[Workflow]  # Nullable (adhoc runs)
    workflow_version: ForeignKey[WorkflowVersion]
    spec: dict                  # Spec snapshot (may differ from version)
    inputs: dict                # Execution inputs
    outputs: dict               # Execution outputs
    config: dict                # LangGraph config
    status: WorkflowRunStatus   # QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED
    source: WorkflowRunSource   # MANUAL, TRIGGER, AGENT, FORM
    error: str                  # Error message (if failed)
    started_at: datetime
    completed_at: datetime
    created_at: datetime

    # Relationships
    checkpoints: ReverseRelation[WorkflowRunCheckpoint]
```

**Run Status Flow**:
```
QUEUED → RUNNING → COMPLETED
                ↓ FAILED
                ↓ CANCELLED
```

### 3. OAuth System (`models_oauth.py`)

#### OAuthConnection

```python
class OAuthConnection(Model):
    id: int
    connection_id: str          # Public ID (conn_...)
    user: ForeignKey[User]
    provider: str               # google, github, etc.
    access_token: str           # Encrypted OAuth token
    refresh_token: str          # Encrypted refresh token
    scopes: list[str]           # Granted scopes
    expires_at: datetime
    created_at: datetime
    updated_at: datetime
```

**Token Refresh**: Handled by `shared/tools/oauth_manager.py`

### 4. Integrations (`models_integrations.py`)

#### IntegrationResource

```python
class IntegrationResource(Model):
    id: int
    resource_id: str            # Public ID (res_...)
    user: ForeignKey[User]
    name: str                   # User-friendly name ("Production DB")
    resource_type: str          # Resource type identifier
    provider: str               # Integration provider
    connection: ForeignKey[OAuthConnection]  # Nullable
    metadata: dict              # Resource-specific metadata
    created_at: datetime
```

**Use case**: Persist "My Gmail Account", "Production Database" references in workflows

#### IntegrationSecret

```python
class IntegrationSecret(Model):
    id: int
    resource: ForeignKey[IntegrationResource]
    key: str                    # Secret key name
    value: str                  # Encrypted secret value
    created_at: datetime
```

**Encryption**: Values encrypted at rest (implementation in credentials resolver)

### 5. Triggers (`models_triggers.py`)

#### TriggerSubscription

```python
class TriggerSubscription(Model):
    id: int
    user: ForeignKey[User]
    workflow: ForeignKey[Workflow]
    trigger_type: str           # cron_schedule, gmail_email_received, etc.
    config: dict                # Trigger-specific config
    enabled: bool
    created_at: datetime
    updated_at: datetime
```

#### TriggerEvent

```python
class TriggerEvent(Model):
    id: int
    subscription: ForeignKey[TriggerSubscription]
    event_data: dict            # Event payload
    event_hash: str             # SHA256 hash (deduplication)
    processed: bool
    created_at: datetime
```

**Deduplication**: Same event hash within 24h → skip processing

## Relationships Diagram

```
User
├── Workflow[]
│   ├── WorkflowDraft (1:1)
│   ├── WorkflowVersion[]
│   │   └── WorkflowRun[]
│   ├── published_version → WorkflowVersion (nullable)
│   └── WorkflowRun[]
├── OAuthConnection[]
│   └── IntegrationResource[]
│       └── IntegrationSecret[]
└── TriggerSubscription[]
    └── TriggerEvent[]
```

## Common Patterns

### Fetching with Relationships

```python
# Prefetch related objects
workflow = await Workflow.filter(id=pk, user=user)\
    .prefetch_related("draft", "published_version", "versions")\
    .first()

# Access prefetched
draft = workflow.draft  # No additional query
```

### Filtering

```python
# Get user's workflows
workflows = await Workflow.filter(user=user).all()

# Get runs for workflow
runs = await WorkflowRun.filter(workflow=workflow).order_by("-created_at").all()
```

### Creating Records

```python
workflow = await Workflow.create(
    user=user,
    name="My Workflow",
    description="..."
)

# Related record
draft = await WorkflowDraft.create(
    workflow=workflow,
    spec={},
    revision=1
)
```

### Updating Records

```python
await Workflow.filter(id=workflow.id).update(
    name="Updated Name",
    updated_at=datetime.now(timezone.utc)
)

# Or via instance
workflow.name = "Updated Name"
await workflow.save()
```

## Public ID Formats

All user-facing IDs use prefixed UUIDs:

```python
from seer.database.workflow_models import (
    make_workflow_public_id,
    parse_workflow_public_id
)

# Create public ID
public_id = make_workflow_public_id(123)  # "wf_a1b2c3d4..."

# Parse back to internal ID
internal_id = parse_workflow_public_id("wf_a1b2c3d4...")  # 123
```

**Formats**:
- Workflow: `wf_{uuid}`
- Run: `run_{uuid}`
- Connection: `conn_{uuid}`
- Resource: `res_{uuid}`

## Migrations

**Tool**: Aerich (Tortoise migration framework)

```bash
# Generate migration
aerich migrate --name "add_workflow_tags"

# Apply migrations
aerich upgrade

# Rollback
aerich downgrade
```

**Migration files**: `/migrations/models/`

## Database Configuration

```python
# shared/config.py
DATABASE_URL = "postgres://user:pass@host:5432/seer"

# Tortoise ORM config
TORTOISE_ORM = {
    "connections": {"default": DATABASE_URL},
    "apps": {
        "models": {
            "models": [
                "shared.database.models",
                "shared.database.workflow_models",
                "shared.database.models_oauth",
                "shared.database.models_integrations",
                "shared.database.models_triggers",
                "aerich.models"
            ],
            "default_connection": "default",
        }
    }
}
```

## Known Issues & Improvements Planned

- [ ] **No repository layer**: Direct ORM queries scattered throughout services
- [ ] **Missing indexes**: Add indexes for common queries (user_id, workflow_id, status)
- [ ] **Unused fields**: `WorkflowProposal.preview_graph`, `applied_graph` appear unused
- [ ] **Secret encryption**: Standardize encryption implementation across IntegrationSecret

## Related Documentation

- [Workflows API](../../api/workflows/README.md) - Workflow CRUD using these models
- [Tools System](../tools/README.md) - OAuth connection & resource usage
- [Triggers](../../api/triggers/README.md) - Trigger subscription system

# API Layer

**Purpose**: FastAPI-based REST API layer handling HTTP requests, authentication, and business logic orchestration.

## Architecture Pattern

This layer follows a **three-tier architecture**:

```
Router Layer (router.py)
    ↓ validates input, extracts user
Service Layer (services/)
    ↓ business logic, orchestration
Database Layer (shared/database/)
    ↓ ORM models via Tortoise
```

### Key Principles

1. **Routers** handle HTTP concerns:
   - Request/response validation (Pydantic models)
   - User authentication via `_require_user(request)`
   - Endpoint routing and status codes
   - NO business logic

2. **Services** contain business logic:
   - Workflow orchestration
   - Database transactions
   - External API calls
   - Validation and error handling

3. **Models** (`api/models/`) define API contracts:
   - Request/response schemas
   - Validation rules
   - Separate from database models

## Error Handling

### Standard Pattern: RFC 7807 Problem Details

Use `_raise_problem()` for all API errors:

```python
from seer.api.workflows.services.shared import _raise_problem

_raise_problem(
    type_uri="https://seer.errors/workflows/validation",
    title="Workflow not found",
    detail=f"Workflow '{workflow_id}' not found",
    status=404,
    errors=[...]  # Optional list of ProblemError
)
```

**Response format**:
```json
{
  "type": "https://seer.errors/workflows/validation",
  "title": "Workflow not found",
  "status": 404,
  "detail": "Workflow 'wf_123' not found",
  "errors": []
}
```

⚠️ **Note**: Currently being standardized across codebase. Some endpoints still use raw `HTTPException` (being migrated).

## Authentication

### User Extraction

```python
def _require_user(request: Request) -> User:
    """Extract authenticated user from request.state.db_user (set by middleware)"""
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user
```

**Usage**:
```python
@router.get("/workflows")
async def list_workflows(request: Request):
    user = _require_user(request)
    return await services.list_workflows(user)
```

⚠️ **Note**: `_require_user` is duplicated in multiple routers. Migration to dependency injection planned.

## Module Structure

```
api/
├── main.py                    # FastAPI app, router registration
├── middleware/                # Auth, analytics, error handling
│   ├── analytics.py          # PostHog event tracking
│   ├── auth.py               # Clerk JWT validation
│   └── timing.py             # Request timing headers
├── models/                    # API request/response schemas
│   ├── workflow_models.py
│   ├── integration_models.py
│   └── ...
├── workflows/                 # Workflow CRUD & execution
│   ├── router.py             # HTTP endpoints
│   ├── models.py             # Request/response models
│   └── services/             # Business logic (7 modules)
├── integrations/              # OAuth & resource browsing
│   ├── router.py             # OAuth callback, resource endpoints
│   ├── services.py           # OAuth flow, scope validation
│   ├── providers/            # OAuth provider implementations
│   └── resource_providers/   # Resource browsing (files, repos)
├── triggers/                  # Trigger catalog & polling
│   ├── router.py
│   ├── services.py
│   └── polling/              # Background polling engine
├── agents/                    # LangGraph-based workflow chat
│   └── workflow/
│       └── router.py         # Chat endpoints
├── tools/                     # Tool execution endpoints
│   └── router.py
├── forms/                     # Form trigger management
│   └── router.py
└── webhooks/                  # Webhook receivers
    └── router.py
```

## Common Patterns

### Service Delegation Pattern

Routers delegate immediately to service layer:

```python
# router.py
@router.get("/workflows/{workflow_id}")
async def get_workflow(request: Request, workflow_id: str):
    user = _require_user(request)
    return await services.get_workflow(user, workflow_id)

# services/lifecycle.py
async def get_workflow(user: User, workflow_id: str):
    workflow = await _get_workflow(user, workflow_id)  # Helper validates & fetches
    return api_models.WorkflowResponse.from_db(workflow)
```

### Database Access Pattern

Services use Tortoise ORM directly:

```python
workflow = await Workflow.filter(id=pk, user=user)\
    .prefetch_related("draft", "published_version")\
    .first()
```

⚠️ **Note**: No repository layer yet. Direct ORM queries throughout services.

### Analytics Pattern

Track events via `shared.analytics.analytics`:

```python
from seer.analytics import analytics

analytics.track(
    str(user.id),
    "workflow_executed",
    {"workflow_id": workflow.public_id, "mode": "sync"}
)
```

## Entry Points

### Main Application

`api/main.py` creates FastAPI app and registers all routers:

```python
app = FastAPI(title="Seer API")

# Middleware
app.add_middleware(AuthMiddleware)
app.add_middleware(AnalyticsMiddleware)

# Routers
app.include_router(workflows_router)
app.include_router(integrations_router)
app.include_router(triggers_router)
# ...
```

### Running Locally

```bash
uvicorn api.main:app --reload --port 8000
```

### Production (via Docker)

```bash
docker compose up api
```

## Key Endpoints

### Workflows
- `GET /v1/workflows` - List workflows
- `POST /v1/workflows` - Create workflow
- `GET /v1/workflows/{id}` - Get workflow
- `POST /v1/workflows/{id}/runs` - Execute workflow
- `PATCH /v1/workflows/{id}/draft` - Update draft

### Integrations
- `GET /v1/integrations/{provider}/authorize` - Start OAuth
- `GET /v1/integrations/{provider}/callback` - OAuth callback
- `GET /v1/integrations/{provider}/resources` - Browse resources

### Triggers
- `GET /v1/triggers` - List trigger types
- `POST /v1/trigger-subscriptions` - Create subscription
- `POST /v1/trigger-subscriptions/{id}/test` - Test trigger

### Chat Agent
- `POST /v1/agent/sessions` - Create chat session
- `POST /v1/agent/sessions/{id}/messages` - Send message

## Related Documentation

- [Workflows Module](./workflows/README.md) - Workflow system details
- [Integrations Module](./integrations/README.md) - OAuth & resource providers
- [Triggers Module](./triggers/README.md) - Trigger polling system
- [Workflow Compiler](/workflow_compiler/README.md) - Compilation & execution engine
- [Database Models](/shared/database/README.md) - Data model relationships

## Development Guidelines

### Adding New Endpoints

1. **Define models** in `api/models/` or module-specific `models.py`
2. **Add router** in module's `router.py`
3. **Implement service** in module's `services.py` or `services/`
4. **Use `_raise_problem`** for errors (not raw `HTTPException`)
5. **Track analytics** for key user actions
6. **Add tests** in `tests/api/`

### Error Handling Best Practices

```python
# ✅ DO: Use _raise_problem with structured errors
_raise_problem(
    type_uri=f"{PROBLEM_BASE}/validation",
    title="Invalid input",
    detail="Workflow name cannot be empty",
    status=400,
    errors=[ProblemError(field="name", message="Required")]
)

# ❌ DON'T: Use raw HTTPException
raise HTTPException(status_code=400, detail="Workflow name cannot be empty")
```

### Service Layer Best Practices

```python
# ✅ DO: Keep routers thin
@router.post("/workflows")
async def create_workflow(request: Request, payload: CreateRequest):
    user = _require_user(request)
    return await services.create_workflow(user, payload)

# ❌ DON'T: Put business logic in routers
@router.post("/workflows")
async def create_workflow(request: Request, payload: CreateRequest):
    user = _require_user(request)
    # 50 lines of validation, database queries, compilation...
```

## Known Issues & Improvements Planned

- [ ] **Error handling standardization**: Migrate all `HTTPException` → `_raise_problem`
- [ ] **Dependency injection**: Replace `_require_user(request)` with FastAPI `Depends()`
- [ ] **Repository layer**: Abstract ORM queries for testability
- [ ] **Service consolidation**: Reduce workflow services from 7 → 4 files
- [ ] **Analytics middleware**: Centralize event tracking vs scattered calls

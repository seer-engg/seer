# Tool System

**Purpose**: Extensible tool registry with OAuth credential resolution, resource binding, and unified execution interface.

## Architecture

```
Tool Registration
    ↓
ToolRegistry (base.py)
    ↓
Tool Execution (executor.py)
    ↓
Credential Resolution (credential_resolver.py)
    ├── OAuth Connection (oauth_manager.py)
    ├── Resource Binding (models_integrations.IntegrationResource)
    └── Secrets (models_integrations.IntegrationSecret)
    ↓
Tool.execute(access_token, arguments, credentials)
```

## Core Components

### 1. BaseTool (`base.py`)

Abstract interface all tools must implement:

```python
class BaseTool(ABC):
    name: str                      # Tool identifier (e.g., "gmail_send_email")
    description: str               # Human-readable description
    required_scopes: List[str]     # OAuth scopes (empty for non-OAuth tools)
    integration_type: str          # Integration type (gmail, github, etc.)
    provider: Optional[str]        # OAuth provider (google, github, etc.)

    @abstractmethod
    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        credentials: Optional[ResolvedCredentials] = None
    ) -> Any:
        """Execute the tool and return result"""

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for tool parameters"""
```

### 2. Tool Registry (`registry.py`)

Global registry for tool discovery:

```python
from seer.tools.base import get_tool, list_tools

# Get tool by name
tool = get_tool("gmail_send_email")

# List all tools
tools = list_tools()  # Returns List[BaseTool]
```

**Registration**: Tools auto-register via decorators or explicit calls during module import.

### 3. Tool Executor (`executor.py`)

Unified execution interface with credential resolution:

```python
from seer.tools.executor import execute_tool

result = await execute_tool(
    tool_name="gmail_send_email",
    user=user,
    connection_id="conn_abc123",  # Optional OAuth connection
    arguments={"to": "user@example.com", "subject": "Hello", "body": "..."}
)
```

**Features**:
- OAuth token management (fetch + refresh)
- Resource binding resolution
- Secret decryption
- Analytics tracking (duration, success/failure)
- Error handling and logging

### 4. Credential Resolver (`credential_resolver.py`)

Resolves all runtime credentials for tool execution:

```python
@dataclass
class ResolvedCredentials:
    connection: Optional[OAuthConnection]      # OAuth connection object
    access_token: Optional[str]                # OAuth access token (refreshed if needed)
    resource: Optional[IntegrationResource]    # Bound resource (if using resource_binding_id)
    secrets: Dict[str, str]                    # Decrypted secrets {key: value}
    secret_records: Dict[str, IntegrationSecret]  # Secret metadata
```

**Resolution flow**:
1. **OAuth Connection**: Fetch by `connection_id` or infer from `tool.provider`
2. **Token Refresh**: Automatically refresh if expired via `oauth_manager.py`
3. **Resource Binding**: Resolve `integration_resource_id` from arguments
4. **Secrets**: Decrypt secrets from resource or connection
5. **Scope Validation**: Ensure connection has required scopes

## Tool Structure

### Directory Layout

```
shared/tools/
├── base.py                     # BaseTool abstract class, registry
├── registry.py                 # Global tool registry
├── executor.py                 # Tool execution with credential resolution
├── credential_resolver.py      # OAuth/resource/secret resolution
├── oauth_manager.py            # Token fetch & refresh
├── scope_validator.py          # Scope validation logic
├── loader.py                   # Dynamic tool loading
├── google/
│   ├── gmail.py               # 15+ Gmail tools (1,597 lines)
│   ├── gdrive.py              # 10+ Drive tools (1,208 lines)
│   └── gsheets.py             # Sheets tools (1,137 lines)
├── github/
│   ├── repos.py
│   ├── issues.py
│   └── pulls.py
├── supabase/
│   ├── database.py
│   ├── auth.py
│   └── storage.py
└── postgres.py                # PostgreSQL tools (786 lines)
```

## Adding a New Tool

### Step 1: Create Tool Class

```python
# shared/tools/my_integration/my_tool.py
from seer.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_integration_do_something"
    description = "Does something useful"
    required_scopes = ["my_scope"]  # OAuth scopes, or [] for non-OAuth
    integration_type = "my_integration"
    provider = "my_provider"  # OAuth provider (google, github, etc.)

    def get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "input_text": {
                    "type": "string",
                    "description": "Text to process"
                }
            },
            "required": ["input_text"]
        }

    async def execute(self, access_token, arguments, credentials=None):
        input_text = arguments.get("input_text")

        # Use access_token for OAuth API calls
        # Or use credentials.secrets for non-OAuth

        return {"result": f"Processed: {input_text}"}
```

### Step 2: Register Tool

```python
# In tool file or __init__.py
from seer.tools.base import register_tool

register_tool(MyTool())
```

### Step 3: Tool Auto-Discovery

Tools are auto-loaded via `loader.py` which imports all tool modules. Ensure your tool module is imported.

## Resource Picker System

Tools can use **resource pickers** for rich UI parameter selection (browse files, repos, etc.).

### Declaring Resource Picker in Schema

```python
def get_parameters_schema(self):
    return {
        "type": "object",
        "properties": {
            "spreadsheet_id": {
                "type": "string",
                "description": "Google Sheets spreadsheet ID",
                "x-resource-picker": {
                    "resource_type": "google_drive_file",
                    "filter": {"mimeType": "application/vnd.google-apps.spreadsheet"},
                    "display_field": "name",
                    "value_field": "id",
                    "search_enabled": True,
                    "hierarchy": True  # Enable folder navigation
                }
            }
        }
    }
```

**Frontend behavior**: UI renders `ResourcePicker` component calling `/api/integrations/{provider}/resources/{resource_type}`.

See [Integrations README](../../api/integrations/README.md) for resource provider implementation.

## OAuth Integration

### Tool Execution with OAuth

```python
# Tool requires OAuth scopes
class GmailSendEmail(BaseTool):
    required_scopes = ["https://www.googleapis.com/auth/gmail.send"]
    provider = "google"

    async def execute(self, access_token, arguments, credentials=None):
        # access_token is automatically refreshed if expired
        headers = {"Authorization": f"Bearer {access_token}"}
        # Make API call...
```

### OAuth Flow

1. User connects integration via `/api/integrations/{provider}/authorize`
2. OAuth callback stores `OAuthConnection` with tokens
3. Tool execution:
   - `CredentialResolver` fetches connection
   - `oauth_manager.get_oauth_token()` refreshes if expired
   - Token passed to `tool.execute()`

### Scope Validation

Before execution, `scope_validator.py` ensures connection has required scopes:

```python
validate_scopes(connection, tool.required_scopes)  # Raises HTTPException if missing
```

## Resource Binding

Tools can reference **persisted resources** (files, repos, databases) via `integration_resource_id`:

```python
# Arguments include resource_binding_id
arguments = {
    "integration_resource_id": "res_abc123",
    "sheet_name": "Sheet1"
}

# CredentialResolver fetches IntegrationResource
resolved = await resolver.resolve(arguments)
resolved.resource  # IntegrationResource with metadata, secrets, connection
```

**Use case**: User saves "My Production Database" resource, workflows reference by ID instead of hardcoding credentials.

## Secrets Management

Tools can access encrypted secrets via resource bindings:

```python
async def execute(self, access_token, arguments, credentials=None):
    # Secrets automatically decrypted in credentials.secrets
    api_key = credentials.secrets.get("api_key")
    database_url = credentials.secrets.get("database_url")
```

**Storage**: Secrets stored in `IntegrationSecret` table, encrypted at rest.

## Analytics

Tool execution automatically tracked via PostHog:

```python
analytics.capture(
    distinct_id=user.user_id,
    event="tool_executed",
    properties={
        "tool_name": tool_name,
        "duration_ms": 123.45,
        "success": True,
        "error_type": None
    }
)
```

## Error Handling

### In Tools

```python
async def execute(self, access_token, arguments, credentials=None):
    if not arguments.get("required_field"):
        raise HTTPException(status_code=400, detail="required_field is required")

    try:
        # API call
    except SomeAPIError as e:
        raise HTTPException(status_code=502, detail=f"API error: {e}")
```

### In Executor

```python
# Executor wraps all errors in HTTPException with 500 status
try:
    result = await execute_tool(...)
except HTTPException as e:
    # Original error preserved
    raise
except Exception as e:
    # Wrapped in HTTPException(500)
    raise
```

## Tool Examples

### Simple Non-OAuth Tool

```python
class TextTransform(BaseTool):
    name = "text_uppercase"
    description = "Converts text to uppercase"
    required_scopes = []  # No OAuth needed
    integration_type = "text"
    provider = None

    def get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"]
        }

    async def execute(self, access_token, arguments, credentials=None):
        return {"result": arguments["text"].upper()}
```

### OAuth Tool with API Call

```python
class GitHubGetRepo(BaseTool):
    name = "github_get_repo"
    description = "Fetches GitHub repository details"
    required_scopes = ["repo"]
    integration_type = "github"
    provider = "github"

    async def execute(self, access_token, arguments, credentials=None):
        owner = arguments["owner"]
        repo = arguments["repo"]

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()
            return response.json()
```

## Known Issues & Improvements Planned

- [ ] **Tool file size**: Gmail (1,597 lines), Drive (1,208 lines), Sheets (1,137 lines) - split into multiple files
- [ ] **Base class for Google tools**: Extract common HTTP client, OAuth refresh, error handling
- [ ] **Pydantic parameter schemas**: Replace verbose JSON schema with Pydantic auto-generation
- [ ] **Defensive type checking**: Remove 30+ line type gymnastics in Gmail tools, use validation
- [ ] **OAuth consolidation**: Unify `oauth_manager.py` and parts of `credential_resolver.py`
- [ ] **Tool signature inconsistency**: Some tools don't accept `credentials` kwarg (backwards compatibility hack in executor.py:103-109)

## Related Documentation

- [API Integrations](../../api/integrations/README.md) - OAuth providers, resource browsing
- [Database Models](../database/README.md) - OAuthConnection, IntegrationResource, IntegrationSecret schemas
- [Workflow Compiler](/workflow_compiler/README.md) - How tools are invoked from workflows

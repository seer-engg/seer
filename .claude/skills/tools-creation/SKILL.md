---
name: tools-creation
description: Context about the tools system architecture in seer. Use when understanding BaseTool, tool registry, execution patterns, or debugging tool issues.
allowed-tools: Read, Grep, Glob
---

# Tools System Architecture

This skill explains how tools work in seer - the extensible framework for integrating external services with workflows.

## Architecture Overview

```
Tool Registration (at import time)
    ↓
register_tool(MyTool())
    ↓
_TOOL_REGISTRY["tool_name"] = tool_instance
    ↓
Workflow Runtime calls execute_tool()
    ↓
CredentialResolver.resolve()
    ├── OAuth connection lookup
    ├── Token refresh if needed
    ├── Resource binding
    └── Secrets decryption
    ↓
tool.execute(access_token, arguments, credentials)
    ↓
Result returned to workflow
```

## Core Components

### BaseTool (`src/seer/tools/base.py`)

Abstract base class all tools inherit from:

```python
class BaseTool(ABC):
    # Required class attributes
    name: str                              # Tool identifier (e.g., "gmail_send_email")
    description: str                       # Human-readable description
    required_scopes: List[str] = []        # OAuth scopes needed
    integration_type: Optional[str] = None # e.g., "gmail", "github"
    provider: Optional[str] = None         # OAuth provider (e.g., "google")
    required_secrets: List[str] = []       # Secrets needed (e.g., "api_key")
    default_resource: Optional[DefaultResourceRequirement] = None

    @abstractmethod
    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        """Execute the tool with resolved credentials."""

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for tool parameters."""

    def get_resource_pickers(self) -> Dict[str, ResourcePickerConfig]:
        """Define parameters that support resource browsing."""

    def get_output_schema(self) -> Dict[str, Any]:
        """Return JSON schema for tool output."""

    def get_metadata(self) -> Dict[str, Any]:
        """Aggregate all metadata for API responses."""
```

### Tool Registry

Global registry for tool lookup:

```python
_TOOL_REGISTRY: Dict[str, BaseTool] = {}

def register_tool(tool: BaseTool) -> None:
    """Register a tool (idempotent)."""
    if tool.name not in _TOOL_REGISTRY:
        _TOOL_REGISTRY[tool.name] = tool

def get_tool(name: str) -> Optional[BaseTool]:
    """Get tool by name."""
    return _TOOL_REGISTRY.get(name)

def list_tools() -> List[BaseTool]:
    """List all registered tools."""
    return list(_TOOL_REGISTRY.values())
```

### Tool Executor (`src/seer/tools/executor.py`)

Unified execution interface:

```python
async def execute_tool(
    tool_name: str,
    user: User,
    connection_id: Optional[str],
    arguments: Dict[str, Any],
) -> Any:
    tool = get_tool(tool_name)
    if not tool:
        raise HTTPException(404, f"Tool '{tool_name}' not found")

    resolver = CredentialResolver(user=user, tool=tool, connection_id=connection_id)
    credentials = await resolver.resolve(arguments)

    return await tool.execute(
        credentials.access_token,
        arguments,
        credentials=credentials,  # Full credentials object
    )
```

## Tool Attributes

### Authentication Attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `required_scopes` | `List[str]` | OAuth scopes tool needs |
| `provider` | `str` | OAuth provider name ("google", "github") |
| `integration_type` | `str` | Integration category ("gmail", "github") |
| `required_secrets` | `List[str]` | Secret names needed (e.g., "api_key") |

### Resource Binding

```python
default_resource: Optional[DefaultResourceRequirement] = {
    "resource_type": "supabase_project",
    "provider": "supabase",
    "required": True,  # Tool fails without bound resource
}
```

## Parameter Schemas

JSON Schema for tool inputs:

```python
def get_parameters_schema(self) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "Email address to send to"
            },
            "subject": {
                "type": "string",
                "description": "Email subject line"
            },
            "body": {
                "type": "string",
                "description": "Email body content"
            }
        },
        "required": ["recipient", "subject", "body"]
    }
```

## Resource Picker Integration

Tools can declare parameters that support resource browsing:

```python
def get_resource_pickers(self) -> Dict[str, ResourcePickerConfig]:
    return {
        "spreadsheet_id": {
            "resource_type": "google_spreadsheet",
            "display_field": "name",
            "value_field": "id",
            "search_enabled": True,
            "hierarchy": False,
        },
        "sheet_name": {
            "resource_type": "google_sheet_tab",
            "depends_on": "spreadsheet_id",  # Nested dependency
        }
    }
```

The `get_metadata()` method injects these as `x-resource-picker` in the schema.

## Provider Base Classes

### GoogleAPIClient (`src/seer/tools/google/base.py`)

Abstract base for all Google tools:

```python
class GoogleAPIClient(BaseTool, ABC):
    provider = "google"

    async def _make_request(
        self,
        method: str,
        url: str,
        *,
        credentials: ResolvedCredentials,
        **kwargs,
    ) -> httpx.Response:
        """Unified request method with error handling."""
        self._validate_token(credentials)
        headers = self._build_headers(credentials.access_token)

        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            if response.status_code >= 400:
                self._handle_api_error(response)
            return response

    def _handle_api_error(self, response: httpx.Response):
        """Translate Google API errors to HTTPException."""
        # 401 → auth error
        # 403 → permission error
        # 404 → not found
        # 429 → rate limit
```

### DiscordAPIClient (`src/seer/tools/discord/base.py`)

Abstract base for Discord tools:

```python
class DiscordAPIClient(BaseTool, ABC):
    provider = "discord"
    required_permissions: int = 0  # Permission bitfield

    @property
    def bot_token(self) -> str:
        return config.discord_bot_token

    async def _make_request(self, method: str, endpoint: str, **kwargs):
        """Discord API request with bot token auth."""
```

## Tool Organization

```
src/seer/tools/
├── base.py                    # BaseTool, registry
├── registry.py                # get_tools_by_integration
├── executor.py                # execute_tool
├── credential_resolver.py     # CredentialResolver
├── oauth_manager.py           # Token management
├── __init__.py                # Tool auto-registration
├── google/
│   ├── base.py                # GoogleAPIClient
│   ├── gmail/
│   │   ├── send.py            # GmailSendEmailTool
│   │   ├── read.py            # GmailReadTool
│   │   └── ...
│   ├── gdrive/
│   │   ├── list.py            # GoogleDriveListTool
│   │   └── ...
│   └── gsheets/
│       └── ...
├── github/
│   └── github.py              # GitHub tools
├── discord/
│   ├── base.py                # DiscordAPIClient
│   ├── messages.py            # Message tools
│   └── ...
├── supabase/
│   ├── database.py            # Query, insert, update
│   └── ...
└── ...
```

## Registration Pattern

Tools auto-register when their modules are imported:

```python
# src/seer/tools/google/gmail/send.py
class GmailSendEmailTool(GoogleAPIClient):
    name = "gmail_send_email"
    description = "Send an email via Gmail"
    integration_type = "gmail"
    required_scopes = ["https://www.googleapis.com/auth/gmail.send"]
    ...

# Register at module level
from seer.tools.base import register_tool
register_tool(GmailSendEmailTool())
```

```python
# src/seer/tools/__init__.py
# Import triggers registration
import seer.tools.google.gmail.send  # noqa
import seer.tools.github.github      # noqa
import seer.tools.discord.messages   # noqa
...
```

## Execution Signature

Tools receive credentials in two ways:

**Legacy** (access_token only):
```python
async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
```

**Modern** (full credentials):
```python
async def execute(
    self,
    access_token: Optional[str],
    arguments: Dict[str, Any],
    credentials: Optional[ResolvedCredentials] = None,  # Includes resource, secrets
) -> Any:
```

## Workflow Integration

Tools are invoked by the workflow runtime:

```python
# In workflow block execution
from seer.tools.executor import execute_tool

result = await execute_tool(
    tool_name="gmail_send_email",
    user=workflow_run.user,
    connection_id=block_config.get("connection_id"),
    arguments={
        "recipient": "user@example.com",
        "subject": "Hello",
        "body": "Message content",
    },
)
```

## Key Design Patterns

### Inheritance Hierarchy

Provider-specific base classes abstract common patterns:
- `GoogleAPIClient` → All Google tools
- `DiscordAPIClient` → All Discord tools

### Idempotent Registration

`register_tool()` is idempotent - safe to call multiple times:
```python
if tool.name in _TOOL_REGISTRY:
    if _TOOL_REGISTRY[tool.name] is tool:
        return  # Same instance, skip
```

### Credential Abstraction

Tools don't manage credentials directly:
- Executor handles OAuth lookup
- Token refresh is automatic
- Scope validation happens before execution

### Provider Inference

When provider isn't explicit, it's inferred:
1. From `tool.provider`
2. From `tool.integration_type`
3. From OAuth connection
4. From resource binding

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/seer/tools/base.py` | BaseTool, registry functions |
| `src/seer/tools/registry.py` | `get_tools_by_integration()` |
| `src/seer/tools/executor.py` | `execute_tool()` unified execution |
| `src/seer/tools/__init__.py` | Tool import/registration |
| `src/seer/tools/google/base.py` | GoogleAPIClient base |
| `src/seer/tools/discord/base.py` | DiscordAPIClient base |

## Related Skills

- `/credential-resolution` - How credentials are resolved at runtime
- `/resource-browser` - Resource picker integration
- `/add-integration` - Step-by-step instructions for adding tools

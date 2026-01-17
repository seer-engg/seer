# Integrations API Module

**Purpose**: OAuth connection management and resource browsing for external integrations (Google, GitHub, Supabase).

## Architecture

```
OAuth Flow
    ↓
IntegrationProvider (authorize → callback → token storage)
    ↓
OAuthConnection (stored in database)
    ↓
Resource Browsing
    ↓
ResourceProvider (list files, repos, databases)
    ↓
Tool Execution (uses OAuth token from connection)
```

## Core Components

### 1. Integration Providers (`providers/`)

**Purpose**: Handle OAuth authorization flow

```python
# providers/base.py
class IntegrationProvider(Protocol):
    provider_name: str

    def get_authorize_url(state: str) -> str
    async def handle_callback(code: str) -> OAuthTokenResponse
    async def refresh_token(refresh_token: str) -> OAuthTokenResponse
```

**Implementations**:
- `providers/google.py` - Google OAuth (gmail, drive, sheets)
- `providers/github.py` - GitHub OAuth
- `providers/supabase.py` - Supabase (API key based, not OAuth)

**Registration**:
```python
from seer.api.integrations.providers.registry import ProviderRegistry

provider = ProviderRegistry.get("google")
```

### 2. Resource Providers (`resource_providers/`)

**Purpose**: Browse external resources (files, repos, databases)

```python
# resource_providers/base.py
class ResourceProvider(Protocol):
    provider_name: str
    resource_types: list[str]

    async def list_resources(
        user: User,
        resource_type: str,
        connection_id: str,
        parent_id: str = None,
        filters: dict = None
    ) -> ResourceListResponse
```

**Implementations**:
- `resource_providers/google.py` - Drive files/folders, Sheets
- `resource_providers/github.py` - Repos, branches, files
- `resource_providers/supabase.py` - Databases, tables

**Resource Types**:
- Google: `google_drive_file`, `google_drive_folder`, `google_sheets`
- GitHub: `github_repo`, `github_branch`, `github_file`
- Supabase: `supabase_database`, `supabase_table`

### 3. Services Layer (`services.py` - 677 lines)

**Responsibilities**: OAuth flow orchestration, scope validation, resource management

**Key Functions**:
```python
# OAuth
await initiate_oauth(user, provider)
await handle_oauth_callback(user, provider, code, state)
await list_connections(user, provider=None)
await delete_connection(user, connection_id)

# Resource browsing
await browse_resources(user, provider, resource_type, connection_id, filters)

# Scope management
await get_required_scopes(provider, tool_names=[])
await validate_connection_scopes(connection, required_scopes)
```

## OAuth Flow

### Step 1: Initiate Authorization

```
User → GET /v1/integrations/{provider}/authorize
    ↓
Generate state token (CSRF protection)
    ↓
Redirect to provider OAuth page
```

```python
provider = ProviderRegistry.get("google")
authorize_url = provider.get_authorize_url(state=csrf_token)
# Redirects to: https://accounts.google.com/o/oauth2/auth?...
```

### Step 2: OAuth Callback

```
Provider → GET /v1/integrations/{provider}/callback?code=...&state=...
    ↓
Validate state token
    ↓
Exchange code for tokens
    ↓
Store OAuthConnection
    ↓
Redirect to frontend
```

```python
token_response = await provider.handle_callback(code)
connection = await OAuthConnection.create(
    user=user,
    provider=provider_name,
    access_token=encrypt(token_response.access_token),
    refresh_token=encrypt(token_response.refresh_token),
    scopes=token_response.scopes,
    expires_at=token_response.expires_at
)
```

### Step 3: Token Usage

Tools automatically use OAuth token via `CredentialResolver`:

```python
# In tool execution
resolver = CredentialResolver(user=user, tool=tool, connection_id=connection_id)
resolved = await resolver.resolve(arguments)
# resolved.access_token is automatically refreshed if expired
```

## Resource Browsing

### List Resources

```
User → GET /v1/integrations/{provider}/resources/{resource_type}?connection_id=...
    ↓
Fetch ResourceProvider
    ↓
Call provider.list_resources(...)
    ↓
Return resource list
```

**Example: Browse Google Drive**

```
GET /v1/integrations/google/resources/google_drive_file?connection_id=conn_123&parent_id=root
```

Response:
```json
{
  "resources": [
    {
      "id": "file_abc",
      "name": "My Document.docx",
      "type": "file",
      "mime_type": "application/vnd.google-apps.document",
      "parent_id": "root",
      "has_children": false
    },
    {
      "id": "folder_xyz",
      "name": "Projects",
      "type": "folder",
      "has_children": true
    }
  ]
}
```

### Resource Picker Integration

Frontend uses `x-resource-picker` schema extension to render resource browser:

```python
# In tool parameter schema
{
    "spreadsheet_id": {
        "type": "string",
        "x-resource-picker": {
            "resource_type": "google_drive_file",
            "filter": {"mimeType": "application/vnd.google-apps.spreadsheet"}
        }
    }
}
```

UI calls `/v1/integrations/google/resources/google_drive_file` and renders picker.

## Scope Validation

### Required Scopes per Tool

```python
# GET /v1/integrations/{provider}/scopes?tools=gmail_send_email,gmail_read_email

{
  "required_scopes": [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly"
  ]
}
```

### Connection Scope Validation

Before tool execution, validate connection has required scopes:

```python
from seer.tools.scope_validator import validate_scopes

validate_scopes(connection, tool.required_scopes)
# Raises HTTPException(403) if scopes missing
```

**Prompt for reauthorization** if scopes insufficient.

## API Endpoints

### OAuth Management

- `GET /v1/integrations/{provider}/authorize` - Initiate OAuth
- `GET /v1/integrations/{provider}/callback` - OAuth callback
- `GET /v1/integrations/connections` - List user's connections
- `DELETE /v1/integrations/connections/{connection_id}` - Disconnect

### Resource Browsing

- `GET /v1/integrations/{provider}/resources/{resource_type}` - List resources
  - Query params: `connection_id`, `parent_id`, `filters`

### Scope Management

- `GET /v1/integrations/{provider}/scopes` - Get required scopes for tools
  - Query params: `tools` (comma-separated tool names)

## Supported Integrations

### Google

**OAuth Scopes**:
- Gmail: `gmail.send`, `gmail.readonly`, `gmail.modify`
- Drive: `drive.readonly`, `drive.file`
- Sheets: `spreadsheets.readonly`, `spreadsheets`

**Resources**:
- `google_drive_file` - Files and folders
- `google_sheets` - Spreadsheets

**Tools**: See `/shared/tools/google/`

### GitHub

**OAuth Scopes**:
- `repo` - Full repository access
- `read:user` - User profile

**Resources**:
- `github_repo` - Repositories
- `github_branch` - Branches
- `github_file` - File tree

**Tools**: See `/shared/tools/github/`

### Supabase

**Authentication**: API key (not OAuth)

**Resources**:
- `supabase_database` - Database instances
- `supabase_table` - Tables

**Tools**: See `/shared/tools/supabase/`

## Adding New Integration

### Step 1: Create Provider

```python
# providers/my_provider.py
from seer.api.integrations.providers.base import IntegrationProvider

class MyProvider(IntegrationProvider):
    provider_name = "my_provider"

    def get_authorize_url(self, state: str) -> str:
        return f"https://oauth.myprovider.com/authorize?state={state}&..."

    async def handle_callback(self, code: str):
        # Exchange code for token
        response = await httpx.post("https://oauth.myprovider.com/token", ...)
        return OAuthTokenResponse(
            access_token=response["access_token"],
            refresh_token=response["refresh_token"],
            scopes=response["scope"].split(" "),
            expires_at=...
        )

    async def refresh_token(self, refresh_token: str):
        # Refresh logic
        ...
```

### Step 2: Register Provider

```python
# providers/__init__.py
from seer.api.integrations.providers.registry import ProviderRegistry
from .my_provider import MyProvider

ProviderRegistry.register(MyProvider())
```

### Step 3: Create Resource Provider (optional)

```python
# resource_providers/my_provider.py
from seer.api.integrations.resource_providers.base import ResourceProvider

class MyResourceProvider(ResourceProvider):
    provider_name = "my_provider"
    resource_types = ["my_resource_type"]

    async def list_resources(self, user, resource_type, connection_id, **kwargs):
        connection = await OAuthConnection.get(connection_id=connection_id, user=user)
        # Fetch resources from seer.api using connection.access_token
        return ResourceListResponse(resources=[...])
```

### Step 4: Register Resource Provider

```python
# resource_providers/__init__.py
from seer.api.integrations.resource_providers.registry import ResourceProviderRegistry
from .my_provider import MyResourceProvider

ResourceProviderRegistry.register(MyResourceProvider())
```

### Step 5: Add Tools

See [Tools README](../../shared/tools/README.md) for tool creation guide.

## Known Issues & Improvements Planned

- [ ] **File bloat**: `router.py` (711 lines), `services.py` (677 lines) - split into oauth/resources
- [ ] **Error handling**: Inconsistent use of `HTTPException` vs `_raise_problem`
- [ ] **Scope validation duplication**: Logic spread across services.py and scope_validator.py
- [ ] **OAuth consolidation**: Token management in 3 places (oauth_manager, credential_resolver, services)

## Related Documentation

- [Tools System](../../shared/tools/README.md) - Tool execution using OAuth tokens
- [Database Models](../../shared/database/README.md) - OAuthConnection, IntegrationResource schemas
- [API Layer](../README.md) - Error handling, authentication patterns

---
name: add-integration
description: Guide for adding new external integrations (OAuth or API key-based) to seer. Use when implementing new service integrations like Google, GitHub, Slack, etc.
allowed-tools: Read, Grep, Glob, Edit, Write, Bash(uv:*, pytest:*)
---

# Add Integration Skill

Guides you through adding a new external integration to seer, supporting both OAuth-based and API key-based authentication. Use when implementing integrations with third-party services.

## Integration Architecture

Integrations follow a modular pattern with clear separation of concerns:

```
OAuth Flow
    ↓
IntegrationProvider (authorization logic)
    ↓
OAuthConnection (database storage)
    ↓
ResourceBrowser (optional - UI resource picker)
    ↓
Tools (execute actions using credentials)
```

## Key Components

### 1. Integration Provider (`src/seer/services/integrations/providers/`)

**Purpose**: Handle authentication flow (OAuth or API key)

**Base Class**: `IntegrationProvider` from `base.py`

**Required Implementation**:
```python
from seer.services.integrations.providers.base import IntegrationProvider, OAuthAuthorizeContext
from typing import Any, Dict

class MyServiceProvider(IntegrationProvider):
    provider = "myservice"  # Primary identifier
    aliases = {"myservice_alias"}  # Optional alternative names
    resource_types = {"myservice_resource"}  # Optional resource types

    # OAuth-specific methods (skip for API key-based integrations)

    def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
        """
        Transform requested scopes into provider-specific format.

        Args:
            context: Contains user, requested_scopes, existing_connection

        Returns:
            Scope string formatted for OAuth provider
        """
        # Example: Join scopes with space
        return " ".join(context.requested_scopes)

        # Example: Provider-specific formatting
        # return ",".join(f"provider:{scope}" for scope in context.requested_scopes)

    def build_authorize_kwargs(
        self, context: OAuthAuthorizeContext, *, state: str, scope: str
    ) -> Dict[str, Any]:
        """
        Customize OAuth authorization parameters.

        Returns:
            Dictionary of parameters for authorize_redirect call
        """
        # Base implementation (most providers):
        return {"state": state, "scope": scope}

        # Example: Add provider-specific params
        # return {
        #     "state": state,
        #     "scope": scope,
        #     "access_type": "offline",  # For refresh tokens
        #     "prompt": "consent",        # Force consent screen
        # }

    def resolve_granted_scopes(
        self, *, token: Dict[str, Any], state_data: Dict[str, Any]
    ) -> str:
        """
        Extract granted scopes from OAuth token response.

        Args:
            token: OAuth token response from provider
            state_data: State data from authorization initiation

        Returns:
            Space-separated scope string to persist
        """
        # Standard implementation (try token, fall back to requested):
        return token.get("scope") or state_data.get("requested_scope") or ""

        # Example: Custom scope extraction
        # return " ".join(token.get("scopes", []))

    async def fetch_user_profile(
        self, *, client: Any, token: Dict[str, Any], state_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fetch user profile from provider API.

        Returns:
            User profile dict with at least {"id": "..."}
        """
        # Example: Userinfo in token (Google)
        # return token.get("userinfo") or {}

        # Example: Fetch from API (GitHub pattern)
        import httpx
        access_token = token.get("access_token")
        if not access_token:
            raise HTTPException(500, "Missing access token")

        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(
                "https://api.myservice.com/user",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()
```

### 2. OAuth Configuration (`src/seer/config.py`)

**Add OAuth Client Credentials**:
```python
# In config.py, add environment variables:

class Config(BaseSettings):
    # ... existing config ...

    # OAuth credentials
    myservice_client_id: Optional[str] = Field(None, env="MYSERVICE_CLIENT_ID")
    myservice_client_secret: Optional[str] = Field(None, env="MYSERVICE_CLIENT_SECRET")
```

**Configure OAuth Client** (`src/seer/services/integrations/auth/oauth.py`):
```python
# Add to OAuth registry configuration

def _get_oauth_config() -> Dict[str, Dict[str, Any]]:
    return {
        # ... existing providers ...

        "myservice": {
            "client_id": config.myservice_client_id,
            "client_secret": config.myservice_client_secret,
            "authorize_url": "https://oauth.myservice.com/authorize",
            "access_token_url": "https://oauth.myservice.com/token",
            "redirect_uri": f"{config.api_base_url}/v1/integrations/myservice/callback",
            "client_kwargs": {
                "scope": "read write",  # Default scopes
            },
        },
    }
```

### 3. Provider Registration

**Register in `src/seer/services/integrations/providers/__init__.py`**:
```python
from .myservice import MyServiceProvider

_registry = ProviderRegistry()
# ... existing registrations ...
_registry.register(MyServiceProvider())
```

### 4. Resource Browsing (Optional)

**If your integration needs a resource picker UI** (e.g., file selector, repo picker):

**Add Resource Browser** (`src/seer/services/integrations/resource_browser.py`):
```python
async def _list_myservice_resources(
    access_token: str,
    resource_type: str,
    options: ResourceListOptions,
) -> Dict[str, Any]:
    """
    List resources from MyService API.

    Args:
        access_token: OAuth access token
        resource_type: Type of resource to list
        options: Query, pagination, filters

    Returns:
        {
            "items": [{"id": "...", "name": "...", "type": "...", ...}],
            "next_page_token": "..." or None,
            "supports_search": bool,
            "supports_hierarchy": bool,
        }
    """
    async with httpx.AsyncClient() as client:
        # Build API request
        params = {}
        if options.query:
            params["q"] = options.query
        if options.page_token:
            params["cursor"] = options.page_token
        params["limit"] = options.page_size

        resp = await client.get(
            f"https://api.myservice.com/{resource_type}",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()

        # Transform to standard format
        items = [
            {
                "id": item["id"],
                "name": item["name"],
                "display_name": item.get("display_name") or item["name"],
                "type": resource_type,
                "metadata": {...},  # Optional extra data
            }
            for item in data.get("items", [])
        ]

        return {
            "items": items,
            "next_page_token": data.get("next_cursor"),
            "supports_search": True,
            "supports_hierarchy": False,
        }

# Register in RESOURCE_TYPE_HANDLERS
RESOURCE_TYPE_HANDLERS: Dict[str, ResourceTypeHandler] = {
    # ... existing handlers ...
    "myservice_file": _list_myservice_resources,
    "myservice_project": _list_myservice_resources,
}

# Register in RESOURCE_TYPE_INFO
RESOURCE_TYPE_INFO: Dict[str, Dict[str, Any]] = {
    # ... existing types ...
    "myservice_file": {
        "label": "Files",
        "supports_search": True,
        "supports_hierarchy": False,
        "depends_on": [],  # Or ["project_id"] if hierarchical
    },
}
```

### 5. Tools Implementation

**Create tools in `src/seer/tools/myservice/`**:

**Example: API Action Tool**
```python
# src/seer/tools/myservice/send_message.py

from typing import Any, Dict
from seer.tools.base import BaseTool, ToolMetadata
from seer.tools.oauth_manager import get_oauth_token
import httpx

class MyServiceSendMessage(BaseTool):
    metadata = ToolMetadata(
        name="myservice_send_message",
        description="Send a message via MyService",
        provider="myservice",
        integration_type="myservice",
        requires_oauth=True,
        required_scopes=["messages.send"],
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Channel ID to send message to",
                    "x-resource-picker": {  # Optional: Enable resource picker UI
                        "resource_type": "myservice_channel",
                    },
                },
                "message": {
                    "type": "string",
                    "description": "Message content",
                },
            },
            "required": ["channel_id", "message"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        # Get OAuth token automatically
        _, access_token = await get_oauth_token(
            self.user,
            provider=self.metadata.provider,
            connection_id=self.connection_id,
        )

        # Call API
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.myservice.com/messages",
                headers={"Authorization": f"Bearer {access_token}"},
                json={
                    "channel": arguments["channel_id"],
                    "text": arguments["message"],
                },
                timeout=30.0,
            )
            resp.raise_for_status()

        return {"success": True, "message_id": resp.json().get("id")}

# Register tool
from seer.tools.registry import register_tool
register_tool(MyServiceSendMessage)
```

**Ensure tools are imported** in `src/seer/tools/__init__.py`:
```python
# Import to trigger registration
import seer.tools.myservice.send_message  # noqa: F401
```

### 6. Database Models

**OAuth connections are handled automatically** via existing models:
- `OAuthConnection` - Stores access/refresh tokens
- `IntegrationResource` - Stores persisted resources (optional)
- `IntegrationSecret` - Stores API keys/secrets (optional)

**If you need custom resource binding** (like Supabase projects):
```python
# In your provider class
async def bind_resource(
    self,
    context: ProviderContext,
    user: User,
    resource_type: str,
    **kwargs,
):
    """Custom resource binding logic."""
    if resource_type != "myservice_project":
        raise HTTPException(400, f"Unsupported resource type: {resource_type}")

    project_id = kwargs.get("project_id")
    if not project_id:
        raise HTTPException(400, "project_id is required")

    # Store resource
    resource = await context.upsert_resource(
        user=user,
        oauth_connection=None,  # Or link to OAuth connection
        provider=self.provider,
        resource_type=resource_type,
        resource_id=project_id,
        resource_key=project_id,
        name=kwargs.get("name") or project_id,
        metadata={"project_id": project_id, ...},
    )

    # Optionally store secrets
    if api_key := kwargs.get("api_key"):
        await context.upsert_secret(
            user=user,
            provider=self.provider,
            name="myservice_api_key",
            secret_type="api_key",
            value_enc=api_key,
            resource=resource,
            oauth_connection=None,
            metadata={},
        )

    return resource
```

## Implementation Checklist

When adding a new integration:

### Phase 1: Provider Setup
- [ ] Create provider class in `src/seer/services/integrations/providers/myservice.py`
- [ ] Extend `IntegrationProvider` base class
- [ ] Implement OAuth lifecycle methods (or skip for API key-based)
- [ ] Add OAuth config to `config.py` and environment variables
- [ ] Configure OAuth client in `auth/oauth.py`
- [ ] Register provider in `providers/__init__.py`

### Phase 2: Resource Browsing (Optional)
- [ ] Add resource type handlers in `resource_browser.py`
- [ ] Add resource type info/metadata
- [ ] Test resource listing endpoints
- [ ] Add resource browsing route in `resource_router.py` (if custom)

### Phase 3: Tools
- [ ] Create tool directory `src/seer/tools/myservice/`
- [ ] Implement tool classes extending `BaseTool`
- [ ] Define parameter schemas with `get_parameters_schema()`
- [ ] Implement `execute()` method with API calls
- [ ] Register tools in `tools/__init__.py`
- [ ] Test tool execution with valid credentials

### Phase 4: Testing
- [ ] Add OAuth flow tests (authorization, callback, token refresh)
- [ ] Add resource browsing tests (if applicable)
- [ ] Add tool execution tests
- [ ] Test error handling (missing tokens, invalid scopes, API errors)
- [ ] Test scope validation and upgrade flows
- [ ] Integration tests with real API (if possible)

### Phase 5: Documentation
- [ ] Update integration README with provider details
- [ ] Document required scopes for each tool
- [ ] Document resource types and capabilities
- [ ] Add example usage in docs

## Common Patterns by Authentication Type

### OAuth 2.0 (Most Common)

**Example Providers**: Google, GitHub, Slack, Microsoft

**Required Methods**:
- `get_oauth_scope()` - Format scopes for provider
- `build_authorize_kwargs()` - Customize authorization params
- `resolve_granted_scopes()` - Extract scopes from token
- `fetch_user_profile()` - Get user identity

**OAuth Configuration**:
```python
# config.py
myservice_client_id: str
myservice_client_secret: str

# auth/oauth.py
"myservice": {
    "client_id": config.myservice_client_id,
    "client_secret": config.myservice_client_secret,
    "authorize_url": "...",
    "access_token_url": "...",
    "redirect_uri": "...",
}
```

### API Key Authentication

**Example Providers**: Supabase (management API), OpenAI

**Pattern**: Store API key in `IntegrationSecret`, skip OAuth methods

**Implementation**:
```python
class MyServiceProvider(IntegrationProvider):
    provider = "myservice"

    # No OAuth methods needed

    async def bind_resource(self, context, user, resource_type, **kwargs):
        """Store API key during resource binding."""
        api_key = kwargs.get("api_key")
        resource = await context.upsert_resource(...)
        await context.upsert_secret(
            user=user,
            provider=self.provider,
            name="myservice_api_key",
            secret_type="api_key",
            value_enc=api_key,  # Encrypted automatically
            resource=resource,
        )
        return resource
```

### OAuth with Incremental Authorization

**Example**: Google (request additional scopes later)

**Pattern**: Use `include_granted_scopes=true` in authorization

```python
def build_authorize_kwargs(self, context, *, state, scope):
    kwargs = {"state": state, "scope": scope}

    # Request incremental auth if adding scopes to existing connection
    if context.existing_connection and context.existing_connection.scopes:
        kwargs["include_granted_scopes"] = "true"
        kwargs["prompt"] = "consent"  # Force consent for new scopes

    return kwargs
```

### Custom Token Exchange

**Example**: Discord (bot installation flow)

**Pattern**: Override token handling in callback

```python
def resolve_granted_scopes(self, *, token, state_data):
    """Discord uses custom scope format."""
    # Extract permissions instead of scopes
    permissions = token.get("permissions", 0)
    return f"permissions:{permissions}"
```

## Scope Management

**Define scopes in tool metadata**:
```python
class MyServiceReadTool(BaseTool):
    metadata = ToolMetadata(
        name="myservice_read",
        required_scopes=["read"],  # Will validate connection has this scope
    )
```

**Scope validation happens automatically** during tool execution via `CredentialResolver`.

**Requesting multiple scopes**:
```python
# Frontend sends scope parameter in /connect request
GET /v1/integrations/myservice/connect?scope=read+write+admin
```

## Error Handling

**Use `raise_problem()` for API errors**:
```python
from seer.api.core.errors import INTEGRATION_PROBLEM, raise_problem

if not access_token:
    raise_problem(
        type_uri=INTEGRATION_PROBLEM,
        title="No active connection",
        detail="Please connect your MyService account first.",
        status=401,
    )
```

**Handle provider API errors gracefully**:
```python
try:
    resp = await client.get(...)
    resp.raise_for_status()
except httpx.HTTPStatusError as exc:
    logger.error("MyService API error: %s", exc.response.text)
    raise_problem(
        type_uri=INTEGRATION_PROBLEM,
        title="MyService API error",
        detail=f"API returned {exc.response.status_code}",
        status=exc.response.status_code,
    )
```

## Testing Your Integration

**Run integration tests**:
```bash
# Run all integration tests
uv run pytest tests/api/integrations/

# Run provider-specific tests
uv run pytest tests/api/integrations/test_myservice_provider.py

# Test OAuth flow end-to-end
uv run pytest tests/api/integrations/test_oauth_flow.py -k myservice
```

**Manual testing checklist**:
1. Start dev server: `uv run python -m seer.api.main`
2. Initiate OAuth: `GET /v1/integrations/myservice/connect?scope=read`
3. Complete authorization in browser
4. Verify connection: `GET /v1/integrations/`
5. Test tool execution via workflow or API
6. Test token refresh (wait for expiry or force refresh)
7. Test scope upgrade (request additional scopes)

## Reference Implementation

**See existing providers for patterns**:
- **Google** (`providers/google.py`) - OAuth with userinfo in token
- **GitHub** (`providers/github.py`) - OAuth with API profile fetch
- **Supabase** (`providers/supabase.py`) - API key + OAuth hybrid
- **Discord** (`providers/discord.py`) - Bot installation flow

## Common Issues

### "Provider not configured"
- Check provider is registered in `providers/__init__.py`
- Verify OAuth config in `auth/oauth.py`
- Ensure environment variables are set

### "Missing required scopes"
- Check tool `required_scopes` matches OAuth config
- Verify frontend sends correct scope in `/connect`
- Check `resolve_granted_scopes()` persists scopes correctly

### "Token refresh failed"
- Ensure provider supports refresh tokens (`access_type=offline`)
- Check `access_token_url` is correct
- Verify client secret is configured

### "Resource picker not working"
- Add resource type to `RESOURCE_TYPE_HANDLERS`
- Register resource type info in `RESOURCE_TYPE_INFO`
- Ensure OAuth connection has required scopes

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/seer/services/integrations/providers/base.py` | Base classes and protocols |
| `src/seer/services/integrations/providers/<provider>.py` | Provider implementation |
| `src/seer/services/integrations/providers/__init__.py` | Provider registration |
| `src/seer/services/integrations/auth/oauth.py` | OAuth client configuration |
| `src/seer/services/integrations/resource_browser.py` | Resource browsing logic |
| `src/seer/api/integrations/router.py` | OAuth flow endpoints |
| `src/seer/api/integrations/resource_router.py` | Resource browsing endpoints |
| `src/seer/tools/<provider>/` | Tool implementations |
| `src/seer/config.py` | Configuration and env vars |
| `src/seer/database/models.py` | Database models |

## Best Practices

1. **Fail fast with clear errors**: Provide actionable error messages
2. **Handle token refresh gracefully**: Use existing `get_oauth_token()` helper
3. **Validate scopes early**: Check scopes before making API calls
4. **Support incremental authorization**: Allow adding scopes to existing connections
5. **Log API errors**: Include provider name, status code, and error details
6. **Test with real APIs**: Use test accounts and sandbox environments
7. **Document required scopes**: Make it clear what permissions each tool needs
8. **Use type hints**: Enable better IDE support and catch errors early

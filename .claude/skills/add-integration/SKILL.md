---
name: add-integration
description: Guide for adding new external integrations (OAuth or API key-based) to seer. Use when implementing new service integrations like Google, GitHub, Slack, etc.
allowed-tools: Read, Grep, Glob, Edit, Write, Bash(uv:*, pytest:*)
---

# Add Integration Guide

Step-by-step instructions for adding a new external integration to seer. This skill focuses on **implementation** - for understanding how each system works, see the context skills below.

## Context Skills (for deeper understanding)

> Before implementing, you may want to understand how each module works:
> - `/oauth-integration` - How OAuth authentication works in seer
> - `/integration-providers` - IntegrationProvider abstraction and lifecycle methods
> - `/integration-metadata` - How INTEGRATION_CONFIGS drives frontend UI
> - `/resource-browser` - Resource picker/browser system
> - `/tools-creation` - Tools system architecture
> - `/credential-resolution` - Runtime credential resolution

## Integration Architecture

```
Integration Metadata (frontend display config)
    ↓
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

**Important**: The frontend is fully dynamic - adding integration metadata in the backend is sufficient for UI rendering.

---

## Implementation Checklist

### Phase 1: Integration Metadata

- [ ] Add integration config to `INTEGRATION_CONFIGS` in `src/seer/services/integrations/metadata.py`
- [ ] Configure display_name, oauth_provider, requires_oauth
- [ ] Add icon URL (use Logo.dev: `https://img.logo.dev/<domain>?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA`)
- [ ] Set brand_color (optional)
- [ ] Define default_scopes and available scopes with display names
- [ ] Add detection_patterns for tool matching
- [ ] Update `get_oauth_provider()` in `auth/oauth.py` to map integration type to provider

### Phase 2: Provider Setup

- [ ] Create provider class in `src/seer/services/integrations/providers/myservice.py`
- [ ] Extend `IntegrationProvider` base class
- [ ] Implement OAuth lifecycle methods (or skip for API key-based)
- [ ] Add OAuth config to `config.py` and environment variables
- [ ] Configure OAuth client in `auth/oauth.py`
- [ ] Register provider in `providers/__init__.py`

### Phase 3: Resource Browsing (Optional)

- [ ] Add resource type handlers in `resource_providers/`
- [ ] Add resource type info/metadata
- [ ] Test resource listing endpoints

### Phase 4: Tools

- [ ] Create tool directory `src/seer/tools/myservice/`
- [ ] Implement tool classes extending `BaseTool`
- [ ] Set `integration_type` and `provider` attributes on tools
- [ ] Define parameter schemas with `get_parameters_schema()`
- [ ] Implement `execute()` method with API calls
- [ ] Register tools in `tools/__init__.py`

### Phase 5: Testing

- [ ] Verify integration appears in `GET /api/integrations/metadata` response
- [ ] Add OAuth flow tests
- [ ] Add tool execution tests
- [ ] Test error handling

---

## Step-by-Step Implementation

### 1. Integration Metadata (`src/seer/services/integrations/metadata.py`)

Add configuration to `INTEGRATION_CONFIGS`:

```python
INTEGRATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ... existing integrations ...

    "myservice": {
        "display_name": "My Service",
        "oauth_provider": "myservice",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/myservice.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#1DA1F2",
        "default_scopes": ["read", "write"],
        "scopes": [
            {
                "value": "read",
                "display_name": "Read access",
                "description": "View your MyService data"
            },
            {
                "value": "write",
                "display_name": "Write access",
                "description": "Create and modify content on your behalf"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["myservice_"],
            "scope_keywords": ["myservice"]
        }
    },
}
```

Update OAuth provider mapping in `auth/oauth.py`:

```python
def get_oauth_provider(integration_type: str) -> str:
    myservice_integrations = ['myservice', 'myservice_messages']
    if integration_type in myservice_integrations:
        return 'myservice'
    # ... existing mappings ...
```

### 2. Integration Provider (`src/seer/services/integrations/providers/myservice.py`)

```python
from seer.services.integrations.providers.base import IntegrationProvider, OAuthAuthorizeContext
from typing import Any, Dict

class MyServiceProvider(IntegrationProvider):
    provider = "myservice"
    aliases = {"myservice_alias"}  # Optional
    resource_types = {"myservice_resource"}  # Optional

    def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
        """Format scopes for provider."""
        return " ".join(context.requested_scopes)

    def build_authorize_kwargs(
        self, context: OAuthAuthorizeContext, *, state: str, scope: str
    ) -> Dict[str, Any]:
        """Customize OAuth authorization parameters."""
        return {"state": state, "scope": scope}

    def resolve_granted_scopes(
        self, *, token: Dict[str, Any], state_data: Dict[str, Any]
    ) -> str:
        """Extract granted scopes from token response."""
        return token.get("scope") or state_data.get("requested_scope") or ""

    async def fetch_user_profile(
        self, *, client: Any, token: Dict[str, Any], state_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch user profile from provider API."""
        import httpx
        access_token = token.get("access_token")

        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(
                "https://api.myservice.com/user",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()
```

### 3. OAuth Configuration

Add to `src/seer/config.py`:

```python
class Config(BaseSettings):
    myservice_client_id: Optional[str] = Field(None, env="MYSERVICE_CLIENT_ID")
    myservice_client_secret: Optional[str] = Field(None, env="MYSERVICE_CLIENT_SECRET")
```

Register OAuth client in `src/seer/services/integrations/auth/oauth.py`:

```python
if config.myservice_client_id and config.myservice_client_secret:
    oauth.register(
        name='myservice',
        client_id=config.myservice_client_id,
        client_secret=config.myservice_client_secret,
        authorize_url='https://oauth.myservice.com/authorize',
        access_token_url='https://oauth.myservice.com/token',
        api_base_url='https://api.myservice.com/',
        client_kwargs={'scope': 'read write'},
    )
```

Register provider in `src/seer/services/integrations/providers/__init__.py`:

```python
from .myservice import MyServiceProvider

_registry.register(MyServiceProvider())
```

### 4. Tools Implementation (`src/seer/tools/myservice/`)

```python
# src/seer/tools/myservice/send_message.py

from typing import Any, Dict
from seer.tools.base import BaseTool, register_tool

class MyServiceSendMessage(BaseTool):
    name = "myservice_send_message"
    description = "Send a message via MyService"
    integration_type = "myservice"
    provider = "myservice"
    required_scopes = ["write"]

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Channel ID to send message to",
                },
                "message": {
                    "type": "string",
                    "description": "Message content",
                },
            },
            "required": ["channel_id", "message"],
        }

    async def execute(self, access_token, arguments, credentials=None):
        import httpx

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
register_tool(MyServiceSendMessage())
```

Import in `src/seer/tools/__init__.py`:

```python
import seer.tools.myservice.send_message  # noqa: F401
```

### 5. Resource Browsing (Optional)

Add resource provider in `src/seer/services/integrations/resource_providers/myservice.py`:

```python
from .base import ResourceProvider, ResourceContext

class MyServiceResourceProvider(ResourceProvider):
    provider = "myservice"
    resource_configs = {
        "myservice_channel": {
            "display_field": "name",
            "supports_search": True,
            "supports_hierarchy": False,
        }
    }

    async def list_resources(
        self, *, resource_type, query, page_token, page_size, context, **kwargs
    ):
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.myservice.com/channels",
                headers={"Authorization": f"Bearer {context.access_token}"},
                params={"q": query, "limit": page_size},
            )
            resp.raise_for_status()
            data = resp.json()

        return {
            "items": [
                {"id": ch["id"], "name": ch["name"], "type": "myservice_channel"}
                for ch in data.get("channels", [])
            ],
            "next_page_token": data.get("next_cursor"),
            "supports_search": True,
            "supports_hierarchy": False,
        }
```

---

## Common Patterns

### OAuth 2.0 (Standard)

Most common pattern. Required methods:
- `get_oauth_scope()` - Format scopes
- `build_authorize_kwargs()` - Customize auth params
- `resolve_granted_scopes()` - Extract granted scopes
- `fetch_user_profile()` - Get user identity

### API Key Authentication

Skip OAuth methods. Use `bind_resource()` to store API key:

```python
async def bind_resource(self, context, user, resource_type, **kwargs):
    api_key = kwargs.get("api_key")
    resource = await context.upsert_resource(
        user=user, provider=self.provider, resource_type=resource_type, ...
    )
    await context.upsert_secret(
        user=user, provider=self.provider, name="api_key",
        secret_type="api_key", value_enc=api_key, resource=resource, ...
    )
    return resource
```

### Incremental Authorization (Google pattern)

Request only new scopes when adding to existing connection:

```python
def build_authorize_kwargs(self, context, *, state, scope):
    kwargs = {"state": state, "scope": scope, "access_type": "offline", "prompt": "consent"}

    if context.existing_connection and context.existing_connection.scopes:
        existing = set(context.existing_connection.scopes.split())
        requested = set(context.requested_scopes)
        if not requested.issubset(existing):
            kwargs["include_granted_scopes"] = "true"

    return kwargs
```

---

## Troubleshooting

### Integration not appearing in frontend

- Check `INTEGRATION_CONFIGS` in `metadata.py` has your integration
- Verify `GET /api/integrations/metadata` returns your integration
- Check icon URL is accessible

### "Provider not configured"

- Check provider is registered in `providers/__init__.py`
- Verify OAuth config in `auth/oauth.py`
- Ensure environment variables are set
- Verify `get_oauth_provider()` maps your integration type

### "Missing required scopes"

- Check tool `required_scopes` matches OAuth config
- Verify frontend sends correct scope in `/connect`
- Check `resolve_granted_scopes()` persists scopes

### "Token refresh failed"

- Ensure provider supports refresh tokens (`access_type=offline`)
- Check `access_token_url` is correct
- Verify client secret is configured

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/seer/services/integrations/metadata.py` | Integration metadata config |
| `src/seer/services/integrations/providers/base.py` | Base classes |
| `src/seer/services/integrations/providers/<provider>.py` | Provider implementation |
| `src/seer/services/integrations/providers/__init__.py` | Provider registration |
| `src/seer/services/integrations/auth/oauth.py` | OAuth client config + provider mapping |
| `src/seer/tools/<provider>/` | Tool implementations |
| `src/seer/config.py` | Configuration and env vars |

---

## Reference Implementations

See existing providers for patterns:
- **Google** (`providers/google.py`) - OAuth with userinfo in token, incremental auth
- **GitHub** (`providers/github.py`) - OAuth with API profile fetch
- **Supabase** (`providers/supabase.py`) - API key + OAuth hybrid
- **Discord** (`providers/discord.py`) - Bot installation flow

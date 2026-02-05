---
name: integration-providers
description: Context about the IntegrationProvider abstraction in seer. Use when understanding how providers work, their lifecycle methods, or debugging provider-specific issues.
allowed-tools: Read, Grep, Glob
---

# Integration Provider Architecture

This skill explains how the IntegrationProvider abstraction works in seer - the layer between OAuth and integration-specific logic.

## Architecture Overview

```
OAuth Flow (router.py)
    ↓
ProviderRegistry.get(provider_name)
    ↓
IntegrationProvider instance
    ├── get_oauth_scope()      → Format scopes
    ├── build_authorize_kwargs() → Customize OAuth params
    ├── resolve_granted_scopes() → Extract scopes from token
    ├── fetch_user_profile()   → Get user identity
    └── bind_resource()        → Optional resource binding
    ↓
OAuthConnection stored
```

## Core Classes

### IntegrationProvider (`src/seer/services/integrations/providers/base.py`)

Base class with **optional override hooks**. Default implementations work for most providers.

**Class Attributes**:
```python
class IntegrationProvider:
    provider: str              # Primary identifier (e.g., "google", "github")
    aliases: Set[str] = set()  # Alternative names (e.g., {"gmail", "googledrive"})
    resource_types: Set[str] = set()  # Supported resource types
```

**Lifecycle Methods** (all have default implementations):

| Method | Purpose | Default Behavior |
|--------|---------|------------------|
| `get_oauth_scope(context)` | Format scopes for provider | Join with spaces |
| `build_authorize_kwargs(context, state, scope)` | Customize OAuth URL params | `{"state": state, "scope": scope}` |
| `resolve_granted_scopes(token, state_data)` | Extract scopes from token response | Use `requested_scope` or `token.scope` |
| `fetch_user_profile(client, token, state_data)` | Get user identity | Return `token.userinfo` |
| `bind_resource(context, user, resource_type, **kwargs)` | Optional resource binding | Raises HTTPException |

### OAuthAuthorizeContext

Context passed to provider methods during authorization:

```python
@dataclass
class OAuthAuthorizeContext:
    user: User                              # Current user
    oauth_provider: str                     # Provider name
    integration_type: Optional[str]         # Original integration type
    requested_scopes: List[str]             # Scopes being requested
    existing_connection: Optional[OAuthConnection]  # Existing connection if re-authorizing
    helpers: OAuthHelpers                   # Helper functions (has_required_scopes)
```

### ProviderContext

Context for resource/secret operations:

```python
@dataclass
class ProviderContext:
    upsert_resource: _ResourceUpserter  # Callback to persist resources
    upsert_secret: _SecretUpserter      # Callback to persist secrets
```

### ProviderRegistry

Singleton pattern for provider lookup:

```python
_registry = ProviderRegistry()
_registry.register(GoogleProvider())
_registry.register(GitHubProvider())

# Lookup by name or alias
provider = _registry.get("google")    # Returns GoogleProvider
provider = _registry.get("gmail")     # Also returns GoogleProvider (via alias)
```

## Provider Implementation Patterns

### Google Provider (`google.py`)

**Characteristics**:
- Uses OpenID Connect (userinfo embedded in token)
- Enforces base scopes: `openid`, `email`, `profile`
- Supports incremental authorization

**Key Overrides**:
```python
def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
    # Always include OpenID scopes
    scopes = {"openid", "email", "profile"}
    scopes.update(context.requested_scopes)
    return " ".join(sorted(scopes))

def build_authorize_kwargs(self, context, *, state, scope):
    kwargs = {
        "state": state,
        "scope": scope,
        "access_type": "offline",  # Get refresh token
        "prompt": "consent",        # Force consent screen
    }
    # Only use incremental auth when adding NEW scopes
    if context.existing_connection and context.existing_connection.scopes:
        existing = set(context.existing_connection.scopes.split())
        requested = set(context.requested_scopes)
        if not requested.issubset(existing):
            kwargs["include_granted_scopes"] = "true"
    return kwargs

async def fetch_user_profile(self, *, client, token, state_data):
    # Google embeds userinfo in token
    return token.get("userinfo") or {}
```

### GitHub Provider (`github.py`)

**Characteristics**:
- Simple OAuth (no OpenID Connect)
- Requires manual profile fetch from API
- Comma-separated scopes

**Key Overrides**:
```python
def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
    # GitHub uses comma-separated scopes
    return ",".join(context.requested_scopes)

async def fetch_user_profile(self, *, client, token, state_data):
    # Must fetch profile from API
    access_token = token.get("access_token")
    async with httpx.AsyncClient() as http_client:
        resp = await http_client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        return resp.json()
```

### Discord Provider (`discord.py`)

**Characteristics**:
- Bot installation flow (not user OAuth)
- Permissions instead of scopes
- Stores guild resources after auth

**Key Overrides**:
```python
def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
    # Discord uses "bot" scope with permissions parameter
    return "bot"

def build_authorize_kwargs(self, context, *, state, scope):
    # Calculate permissions bitfield from requested tools
    permissions = self._calculate_permissions(context)
    return {
        "state": state,
        "scope": scope,
        "permissions": str(permissions),
    }

def resolve_granted_scopes(self, *, token, state_data):
    # Discord returns permissions, not scopes
    permissions = token.get("permissions", 0)
    return f"permissions:{permissions}"
```

### Supabase Provider (`supabase.py`)

**Characteristics**:
- Hybrid OAuth + API key
- Management API OAuth for project listing
- Service role keys for database access

Supports `bind_resource()` for project binding:
```python
async def bind_resource(self, context, user, resource_type, **kwargs):
    project_id = kwargs.get("project_id")
    resource = await context.upsert_resource(
        user=user,
        provider=self.provider,
        resource_type="supabase_project",
        resource_id=project_id,
        ...
    )
    # Store service role key as secret
    if service_key := kwargs.get("service_role_key"):
        await context.upsert_secret(
            user=user,
            provider=self.provider,
            name="service_role_key",
            value_enc=service_key,
            resource=resource,
            ...
        )
    return resource
```

## Provider Registration

Providers are registered in `src/seer/services/integrations/providers/__init__.py`:

```python
from .google import GoogleProvider
from .github import GitHubProvider
from .discord import DiscordProvider
from .supabase import SupabaseProvider
from .linkedin import LinkedInProvider

_registry = ProviderRegistry()
_registry.register(GoogleProvider())
_registry.register(GitHubProvider())
_registry.register(DiscordProvider())
_registry.register(SupabaseProvider())
_registry.register(LinkedInProvider())

def get_provider(name: str) -> Optional[IntegrationProvider]:
    return _registry.get(name)
```

## Key Design Patterns

### Default + Override

Base class provides sensible defaults. Providers only override what's different:
- Google: Overrides everything (OpenID Connect, incremental auth)
- GitHub: Only overrides `fetch_user_profile()` (manual API fetch)
- Simple providers: May not need any overrides

### Alias Support

Multiple integration types can share one provider:
```python
class GoogleProvider(IntegrationProvider):
    provider = "google"
    aliases = {"gmail", "googledrive", "googlesheets"}
```

Looking up "gmail" returns the same GoogleProvider instance.

### Context Injection

Providers receive rich context for customization:
- User info for user-specific logic
- Existing connection for incremental auth decisions
- Helper functions for scope validation

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/seer/services/integrations/providers/base.py` | Base classes, context dataclasses, registry |
| `src/seer/services/integrations/providers/__init__.py` | Provider registration |
| `src/seer/services/integrations/providers/google.py` | Google/Gmail/Drive/Sheets provider |
| `src/seer/services/integrations/providers/github.py` | GitHub provider |
| `src/seer/services/integrations/providers/discord.py` | Discord bot provider |
| `src/seer/services/integrations/providers/supabase.py` | Supabase provider |
| `src/seer/services/integrations/providers/linkedin.py` | LinkedIn provider |

## Related Skills

- `/oauth-integration` - OAuth flow and token management
- `/add-integration` - Step-by-step instructions for adding providers

---
name: credential-resolution
description: Context about runtime credential resolution in seer. Use when understanding how OAuth tokens, resources, and secrets are resolved for tool execution.
allowed-tools: Read, Grep, Glob
---

# Credential Resolution Architecture

This skill explains how credentials are resolved at runtime before tool execution - the bridge between OAuth connections and actual tool invocation.

## Architecture Overview

```
execute_tool(tool_name, user, connection_id, arguments)
    ↓
CredentialResolver(user, tool, connection_id)
    ↓
resolver.resolve(arguments)
    ↓
┌─────────────────────────────────────────┐
│ 1. _resolve_connection()                │
│    ├── Lookup by connection_id          │
│    ├── Or infer provider from tool      │
│    ├── get_oauth_token() with refresh   │
│    └── validate_scopes()                │
├─────────────────────────────────────────┤
│ 2. _resolve_resource()                  │
│    ├── Extract resource_id from args    │
│    └── Or find default_resource         │
├─────────────────────────────────────────┤
│ 3. _resolve_secrets()                   │
│    ├── Lookup by resource              │
│    ├── Or by connection                 │
│    └── Or standalone                    │
└─────────────────────────────────────────┘
    ↓
ResolvedCredentials(connection, access_token, resource, secrets)
    ↓
tool.execute(access_token, arguments, credentials=credentials)
```

## Core Components

### ResolvedCredentials (`src/seer/tools/credential_resolver.py`)

Dataclass containing all resolved credentials:

```python
@dataclass
class ResolvedCredentials:
    connection: Optional[OAuthConnection] = None  # OAuth connection
    access_token: Optional[str] = None            # Decrypted token
    resource: Optional[IntegrationResource] = None # Bound resource
    secrets: Dict[str, str] = field(default_factory=dict)  # Decrypted secrets
    secret_records: Dict[str, IntegrationSecret] = field(default_factory=dict)  # Metadata
```

### CredentialResolver

Main resolver class:

```python
class CredentialResolver:
    def __init__(
        self,
        *,
        user: User,
        tool: BaseTool,
        connection_id: Optional[str] = None,
    ) -> None: ...

    async def resolve(self, arguments: Dict[str, Any]) -> ResolvedCredentials:
        """Resolve all credentials needed for tool execution."""
        connection, access_token = await self._resolve_connection()
        resource = await self._resolve_resource(arguments, connection)
        secrets, secret_records = await self._resolve_secrets(resource, connection)
        return ResolvedCredentials(...)
```

## Resolution Flow

### 1. Connection Resolution (`_resolve_connection`)

**When required**: Only if tool has `required_scopes`:

```python
async def _resolve_connection(self):
    if not self.tool.required_scopes:
        return None, None  # Non-OAuth tool

    provider = self._infer_provider()
    connection, access_token = await get_oauth_token(
        self.user,
        connection_id=self.connection_id,
        provider=provider,
    )

    # Validate scopes
    is_valid, missing_scope = validate_scopes(connection, self.tool.required_scopes)
    if not is_valid:
        raise HTTPException(403, f"Missing required scope '{missing_scope}'")

    return connection, access_token
```

### 2. Resource Resolution (`_resolve_resource`)

**Resource ID extraction** from arguments:

```python
RESOURCE_ARGUMENT_KEYS = ("integration_resource_id", "resource_id", "resource_binding_id")

def _extract_resource_id(arguments: Dict) -> Optional[int]:
    for key in RESOURCE_ARGUMENT_KEYS:
        if key in arguments and arguments[key] is not None:
            return int(arguments[key])
    return None
```

**Lookup flow**:
1. If resource_id in arguments → lookup by ID
2. If tool has `default_resource` → find matching resource
3. Otherwise → None

```python
async def _resolve_resource(self, arguments, connection):
    resource_id = self._extract_resource_id(arguments)

    if resource_id:
        resource = await IntegrationResource.get_or_none(
            id=resource_id,
            user=self.user,
            status="active",
        )
        # Validate provider and connection match
        ...
        return resource

    elif self.tool.default_resource:
        return await self._find_default_resource(provider, connection)

    return None
```

### 3. Secret Resolution (`_resolve_secrets`)

**Lookup priority**:
1. Secrets bound to the resource
2. Secrets bound to the connection
3. Standalone secrets (no resource/connection)

```python
async def _resolve_secrets(self, resource, connection):
    if not self.tool.required_secrets:
        return {}, {}

    secrets = {}
    for name in self.tool.required_secrets:
        secret = await self._find_secret(provider, name, resource, connection)
        if not secret:
            raise HTTPException(404, f"Missing required secret '{name}'")
        secrets[name] = secret.value_enc  # Decrypted value
    return secrets, secret_records
```

## Token Management (`src/seer/tools/oauth_manager.py`)

### get_oauth_token

Resolves connection with automatic refresh:

```python
async def get_oauth_token(
    user: User,
    connection_id: Optional[str] = None,
    provider: Optional[str] = None,
) -> Tuple[OAuthConnection, str]:
    # Lookup connection
    if connection_id:
        connection = await OAuthConnection.get(id=..., user=user, status="active")
    elif provider:
        connection = await OAuthConnection.get_or_none(user=user, provider=provider, status="active")

    # Check expiration
    if connection.expires_at and connection.expires_at < datetime.now(timezone.utc):
        connection = await refresh_oauth_token(connection)

    return connection, connection.access_token_enc
```

### refresh_oauth_token

Provider-specific token refresh:

```python
async def refresh_oauth_token(connection: OAuthConnection) -> OAuthConnection:
    if connection.provider in ["google", "googledrive", "gmail"]:
        refresh_url = "https://oauth2.googleapis.com/token"
        refresh_data = {
            "client_id": config.google_client_id,
            "client_secret": config.google_client_secret,
            "refresh_token": connection.refresh_token_enc,
            "grant_type": "refresh_token",
        }
    elif connection.provider == "github":
        refresh_url = "https://github.com/login/oauth/access_token"
        ...
    elif connection.provider in ["supabase", "supabase_mgmt"]:
        refresh_url = "https://api.supabase.com/v1/oauth/token"
        ...

    # POST to refresh endpoint
    response = await client.post(refresh_url, data=refresh_data)
    token_data = response.json()

    # Update connection
    connection.access_token_enc = token_data["access_token"]
    connection.expires_at = datetime.now(timezone.utc) + timedelta(seconds=token_data["expires_in"])
    await connection.save()

    return connection
```

## Scope Validation (`src/seer/tools/scope_validator.py`)

Validates that connection has required scopes:

```python
def validate_scopes(
    connection: OAuthConnection,
    required_scopes: List[str],
) -> Tuple[bool, Optional[str]]:
    """
    Returns (is_valid, missing_scope).
    missing_scope is None if all scopes are satisfied.
    """
    granted = parse_scopes(connection.scopes)

    for required in required_scopes:
        if required in granted:
            continue
        # For Google APIs, check hierarchy
        if "googleapis.com" in required:
            if not _any_scope_satisfies(granted, required):
                return False, required
        else:
            return False, required

    return True, None
```

**Google scope hierarchy**:
- `gmail` satisfies `gmail.readonly`, `gmail.send`, etc.
- Narrower scopes do NOT satisfy broader scopes

## Provider Inference

When provider isn't explicitly provided:

```python
def _infer_provider(self, connection=None, resource=None) -> Optional[str]:
    # Priority:
    # 1. Resource provider
    if resource and resource.provider:
        return resource.provider
    # 2. Connection provider
    if connection and connection.provider:
        return connection.provider
    # 3. Tool attributes
    return self.tool.provider or self.tool.integration_type
```

## Default Resource Resolution

For tools with `default_resource`:

```python
async def _find_default_resource(self, provider, connection):
    config = self.tool.default_resource
    resource_type = config.get("resource_type")

    filters = {
        "user": self.user,
        "provider": provider,
        "resource_type": resource_type,
        "status": "active",
    }

    queryset = IntegrationResource.filter(**filters)
    if connection:
        queryset = queryset.filter(oauth_connection=connection)

    # Return most recently updated
    return await queryset.order_by("-updated_at").first()
```

## Error Handling

Clear error messages for common failures:

| Scenario | HTTP Status | Message |
|----------|-------------|---------|
| Tool requires OAuth, no user | 401 | "Tool requires OAuth authentication" |
| No connection_id or provider | 400 | "connection_id must be provided" |
| Connection not found | 404 | "OAuth connection not found" |
| Missing required scope | 403 | "Missing required scope 'X'" |
| Resource not found | 404 | "Integration resource not found" |
| Missing required secret | 404 | "Missing required secret 'X'" |
| Token expired, no refresh token | 401 | "Please reconnect your account" |

## Key Design Patterns

### Lazy Resolution

Only resolves what's needed:
- No `required_scopes` → skip OAuth
- No `required_secrets` → skip secrets
- No `default_resource` → skip resource lookup

### Fail Fast

Validates early, before tool execution:
- Scope validation before API calls
- Resource ownership check
- Secret availability check

### Token Refresh on Demand

Tokens refreshed only when expired and accessed:
- Check `expires_at < now`
- Refresh automatically
- Update database

### Multi-Source Secrets

Secrets can come from multiple locations:
1. Resource-specific (Supabase project key)
2. Connection-specific (OAuth client secret)
3. User-level (API keys)

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/seer/tools/credential_resolver.py` | CredentialResolver, ResolvedCredentials |
| `src/seer/tools/oauth_manager.py` | get_oauth_token, refresh_oauth_token |
| `src/seer/tools/scope_validator.py` | validate_scopes, scope parsing |
| `src/seer/tools/executor.py` | execute_tool (uses resolver) |

## Related Skills

- `/oauth-integration` - OAuth flow and token storage
- `/tools-creation` - Tool attributes affecting resolution
- `/add-integration` - Step-by-step instructions for adding integrations

---
name: credential-resolution
description: Context about runtime credential resolution in seer. Use when understanding how OAuth tokens, resources, and secrets are resolved for tool execution.
allowed-tools: Read, Grep, Glob
---

# Credential Resolution Architecture

Bridge between OAuth connections and tool invocation. Lives in `src/seer/tools/credential_resolver.py`.

## Resolution Flow

`execute_tool(tool_name, user, connection_id, arguments)` → `CredentialResolver(user, tool, connection_id)` → `resolver.resolve(arguments)` → three phases → `ResolvedCredentials` → `tool.execute(access_token, arguments, credentials=credentials)`

## Three Resolution Phases

**1. Connection** — Skipped if tool has no `required_scopes`. Otherwise lookup by `connection_id` or infer provider from tool, call `get_oauth_token()` (auto-refreshes on expiry), validate scopes. Raises 403 on missing scope.

**2. Resource** — Extract `resource_id` from arguments (keys: `integration_resource_id`, `resource_id`, `resource_binding_id`). If found, lookup by ID. If tool has `default_resource`, find matching active resource. Otherwise None.

**3. Secrets** — Skipped if tool has no `required_secrets`. Lookup priority: resource-bound → connection-bound → standalone. Raises 404 on missing secret.

## Error Table

| Scenario | Status |
|----------|--------|
| No connection_id or provider | 400 |
| Connection not found | 404 |
| Missing required scope | 403 |
| Resource not found | 404 |
| Missing required secret | 404 |
| Token expired, no refresh token | 401 |

## Key Design Patterns
- **Lazy resolution** — only resolves what the tool needs
- **Fail fast** — validates before tool execution
- **Token refresh on demand** — checks `expires_at`, refreshes automatically
- **Multi-source secrets** — resource-specific, connection-specific, or user-level

## Key Files

| File | Purpose |
|------|---------|
| `src/seer/tools/credential_resolver.py` | `CredentialResolver`, `ResolvedCredentials` |
| `src/seer/tools/oauth_manager.py` | `get_oauth_token`, `refresh_oauth_token` |
| `src/seer/tools/scope_validator.py` | `validate_scopes`, scope parsing |
| `src/seer/tools/executor.py` | `execute_tool` (uses resolver) |

## Related Skills
- `/oauth-integration` · `/tools-creation` · `/add-integration`

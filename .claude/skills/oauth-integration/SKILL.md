---
name: oauth-integration
description: Context about how OAuth authentication works in seer. Use when understanding OAuth flows, token management, scope validation, or debugging OAuth issues.
allowed-tools: Read, Grep, Glob
---

# OAuth Integration Architecture

## Flow

`Frontend connect button` → `GET /{provider}/connect?scope=...` → build base64-encoded state → `provider.build_authorize_kwargs()` → Authlib redirect to OAuth provider → user authorizes → `GET /{provider}/callback?code=...&state=...` → manual httpx code exchange → `provider.resolve_granted_scopes()` → `provider.fetch_user_profile()` → `store_oauth_connection()` → redirect to frontend

## Key Components

**OAuth Client Registration** (`auth/oauth.py`) — Authlib Starlette integration. Google uses OpenID Connect metadata URL. GitHub/Discord use manual authorize + token URLs. `get_oauth_provider()` maps integration types to providers (gmail/googlesheets/googledrive → google).

**Helper Utilities** (`auth/helpers.py`) — Scope parsing (space or comma-separated), scope validation with Google hierarchy (broad scopes satisfy narrow ones), scope merging on reconnect (accumulative), account ID extraction (provider-specific).

**OAuthConnection model** (`database/models_oauth.py`) — stores user, provider, provider_account_id, encrypted access/refresh tokens, expires_at, scopes string, status. Unique constraint: `(user, provider, provider_account_id)`.

**Token Management** (`tools/oauth_manager.py`) — `get_oauth_token()` resolves by connection_id or provider, checks expiry, auto-refreshes. `refresh_oauth_token()` posts to provider-specific endpoint and updates the connection record.

## Design Patterns
- **Stateless OAuth** — state is base64-encoded JSON in callback URL (no server sessions, works multi-worker)
- **Incremental authorization** — re-connect with `include_granted_scopes=true` to add scopes without revoking existing
- **Lazy token refresh** — refreshed on-demand when `expires_at < now`, not proactively
- **Scope accumulation** — existing scopes are always preserved when re-connecting

## Key Files

| File | Purpose |
|------|---------|
| `src/seer/services/integrations/auth/oauth.py` | OAuth client registration, provider mapping |
| `src/seer/services/integrations/auth/helpers.py` | Scope parsing, validation, storage, merging |
| `src/seer/api/integrations/router.py` | `/connect`, `/callback`, `/disconnect` endpoints |
| `src/seer/tools/oauth_manager.py` | Token resolution and refresh |
| `src/seer/database/models_oauth.py` | `OAuthConnection` model |

## Related Skills
- `/integration-providers` · `/credential-resolution` · `/add-integration`

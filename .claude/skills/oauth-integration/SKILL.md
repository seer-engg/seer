---
name: oauth-integration
description: Context about how OAuth authentication works in seer. Use when understanding OAuth flows, token management, scope validation, or debugging OAuth issues.
allowed-tools: Read, Grep, Glob
---

# OAuth Integration Architecture

This skill provides context about how OAuth authentication works in seer. It explains the architecture, key components, and design patterns - not implementation instructions.

## Architecture Overview

```
Frontend (connect button)
    ↓
GET /{provider}/connect?scope=...
    ↓
Build state (base64-encoded, stateless)
    ↓
Provider.build_authorize_kwargs()
    ↓
Authlib authorize_redirect → OAuth Provider
    ↓
User authorizes
    ↓
GET /{provider}/callback?code=...&state=...
    ↓
Exchange code for tokens (manual httpx POST)
    ↓
Provider.resolve_granted_scopes()
    ↓
Provider.fetch_user_profile()
    ↓
store_oauth_connection() → OAuthConnection table
    ↓
Redirect to frontend with success
```

## Key Components

### 1. OAuth Client Registration (`src/seer/services/integrations/auth/oauth.py`)

Uses **Authlib's Starlette integration** to register OAuth clients:

- **Google**: Uses OpenID Connect metadata URL, minimal default scopes (`openid email profile`)
- **GitHub**: Manual authorize/token URLs, minimal default (`user:email`)
- **Supabase**: Management API OAuth (`read:projects`)
- **Discord**: Bot installation flow (`bot` scope)
- **LinkedIn**: OpenID Connect (`openid profile email`)

The `get_oauth_provider()` function maps integration types to OAuth providers:
- `gmail`, `googlesheets`, `googledrive` → `google`
- `github`, `pull_request` → `github`
- Others map to themselves

### 2. OAuth Helper Utilities (`src/seer/services/integrations/auth/helpers.py`)

**Scope Parsing**: Handles both formats:
- Google: whitespace-separated (`scope1 scope2 scope3`)
- GitHub: comma-separated (`scope1,scope2,scope3`)

**Scope Validation with Hierarchy**:
- For Google APIs, broader scopes satisfy narrower ones
- `gmail` scope satisfies `gmail.readonly`, `gmail.send`, `gmail.modify`
- Narrower scopes do NOT satisfy broader scopes

**Scope Merging**: When re-connecting, new scopes are merged with existing ones (accumulative model).

**Account ID Extraction**: Provider-specific logic:
- Google: `sub` or `email` from profile
- GitHub: `id` from profile
- Discord: `id` from profile
- LinkedIn: `sub` (OpenID Connect)

### 3. Database Model (`src/seer/database/models_oauth.py`)

**OAuthConnection** table stores:
- `id`, `user_id`, `provider`, `provider_account_id`
- `access_token_enc`, `refresh_token_enc` (encrypted)
- `expires_at` (timestamp)
- `scopes` (space-separated string)
- `token_type` (usually "Bearer")
- `provider_metadata` (JSON for provider-specific data)
- `status` ("active", "revoked", "error")

**Unique constraint**: `(user, provider, provider_account_id)` - one connection per provider account per user.

### 4. Token Management (`src/seer/tools/oauth_manager.py`)

**`get_oauth_token(user, connection_id, provider)`**:
- Resolves connection by ID or provider
- Checks if token expired (`expires_at < now`)
- Triggers refresh if expired
- Returns `(connection, access_token)` tuple

**`refresh_oauth_token(connection)`**:
- Provider-specific token endpoints
- Constructs refresh request with client credentials
- Updates connection with new token and expiration

## Design Patterns

### Stateless OAuth (Multi-Worker Support)

OAuth state is **base64-encoded** in the callback URL instead of server-side sessions:

```python
state_data = {
    "user_id": user.user_id,
    "oauth_provider": oauth_provider,
    "requested_scope": requested_scope,
    "redirect_to": redirect_to,
}
state = base64.urlsafe_b64encode(json.dumps(state_data).encode()).decode()
```

This allows any worker to handle the callback without shared session storage.

### Incremental Authorization

When requesting additional scopes for an existing connection:
1. Check if existing connection already has required scopes
2. If not, build new authorization URL with `include_granted_scopes=true` (Google)
3. After callback, merge new scopes with existing

### Lazy Token Refresh

Tokens are refreshed **on-demand** (when accessed), not proactively:
1. Tool execution requests token via `get_oauth_token()`
2. If `expires_at < now`, refresh is triggered
3. Updated token is persisted

### Scope Accumulation

Once a user grants a scope, it's retained:
- Existing scopes + new scopes = merged scopes
- Re-connecting with subset of scopes doesn't revoke others

## OAuth Flow Details

### Authorization Initiation (`GET /{provider}/connect`)

1. **Validate scope parameter** - Frontend must specify scopes
2. **Check existing scopes** - Skip OAuth if already satisfied
3. **Build state object** - user_id, requested_scopes, redirect_to
4. **Call provider hooks**:
   - `provider.get_oauth_scope()` - Format scopes for provider
   - `provider.build_authorize_kwargs()` - Customize OAuth params
5. **Redirect to provider** via Authlib's `authorize_redirect()`

### Callback Handling (`GET /{provider}/callback`)

1. **Validate state** - Decode base64, verify user_id matches
2. **Exchange code** - Manual httpx POST (bypasses Authlib session validation)
3. **Handle expires_in** - Convert to expires_at timestamp
4. **Resolve scopes** - `provider.resolve_granted_scopes()`
5. **Fetch profile** - `provider.fetch_user_profile()`
6. **Store connection** - `store_oauth_connection()` with scope merging
7. **Redirect to frontend** - `?connected={provider}`

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/seer/services/integrations/auth/oauth.py` | OAuth client registration, provider mapping |
| `src/seer/services/integrations/auth/helpers.py` | Scope parsing, validation, storage, merging |
| `src/seer/api/integrations/router.py` | OAuth endpoints (`/connect`, `/callback`, `/disconnect`) |
| `src/seer/tools/oauth_manager.py` | Token resolution and refresh |
| `src/seer/database/models_oauth.py` | OAuthConnection model |

## Related Skills

- `/integration-providers` - Provider abstraction and lifecycle methods
- `/credential-resolution` - How credentials are resolved at runtime
- `/add-integration` - Step-by-step instructions for adding integrations

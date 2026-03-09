---
name: integration-providers
description: Context about the IntegrationProvider abstraction in seer. Use when understanding how providers work, their lifecycle methods, or debugging provider-specific issues.
allowed-tools: Read, Grep, Glob
---

# Integration Provider Architecture

Layer between OAuth flow and integration-specific logic. Lives in `src/seer/services/integrations/providers/`.

## Flow

`OAuth router` → `ProviderRegistry.get(name)` → `IntegrationProvider` lifecycle hooks → `OAuthConnection` stored

## Lifecycle Methods (all have defaults)

| Method | Purpose | Default |
|--------|---------|---------|
| `get_oauth_scope(context)` | Format scopes for provider | Join with spaces |
| `build_authorize_kwargs(context, state, scope)` | Customize OAuth URL params | `{state, scope}` |
| `resolve_granted_scopes(token, state_data)` | Extract scopes from token | `requested_scope` or `token.scope` |
| `fetch_user_profile(client, token, state_data)` | Get user identity | `token.userinfo` |
| `bind_resource(context, user, resource_type, **kwargs)` | Optional resource binding | Raises HTTPException |

## Provider Characteristics

- **Google** — OpenID Connect (userinfo in token), enforces `openid email profile` base scopes, incremental authorization via `include_granted_scopes=true`, `access_type=offline` for refresh token
- **GitHub** — Simple OAuth, must fetch profile from API, comma-separated scopes
- **Discord** — Bot installation flow, `bot` scope + permissions bitfield, stores guild resources after auth
- **Supabase** — Hybrid OAuth + API key, Management API OAuth for project listing, service role key stored as secret via `bind_resource()`

## Key Patterns
- **Default + Override** — only override what differs from base
- **Alias support** — `aliases = {"gmail", "googledrive"}` lets multiple integration types share one provider
- **Context injection** — providers receive user, existing connection, and helper functions for customization

## Key Files

| File | Purpose |
|------|---------|
| `src/seer/services/integrations/providers/base.py` | `IntegrationProvider`, context dataclasses, `ProviderRegistry` |
| `src/seer/services/integrations/providers/__init__.py` | Provider registration |
| `src/seer/services/integrations/providers/google.py` | Google/Gmail/Drive/Sheets |
| `src/seer/services/integrations/providers/github.py` | GitHub |
| `src/seer/services/integrations/providers/discord.py` | Discord bot |
| `src/seer/services/integrations/providers/supabase.py` | Supabase |

## Related Skills
- `/oauth-integration` · `/add-integration`

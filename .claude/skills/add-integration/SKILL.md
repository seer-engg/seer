---
name: add-integration
description: Guide for adding new external integrations (OAuth or API key-based) to seer. Use when implementing new service integrations like Google, GitHub, Slack, etc.
allowed-tools: Read, Grep, Glob, Edit, Write, Bash(uv:*, pytest:*)
---

# Add Integration Guide

The frontend is fully dynamic — adding backend metadata is sufficient for UI rendering. No frontend code changes needed.

**Architecture:** `Metadata config → OAuth flow → IntegrationProvider → OAuthConnection → ResourceBrowser (optional) → Tools`

## Implementation Checklist

**Phase 1: Metadata** — Add entry to `INTEGRATION_CONFIGS` in `src/seer/services/integrations/metadata.py`. Set display_name, oauth_provider, icon (Logo.dev URL), brand_color, default_scopes, available scopes, detection_patterns. Update `get_oauth_provider()` in `auth/oauth.py`.

**Phase 2: Provider** — Create provider class in `src/seer/services/integrations/providers/myservice.py` extending `IntegrationProvider`. Add OAuth config to `config.py` and env vars. Register OAuth client in `auth/oauth.py`. Register provider in `providers/__init__.py`.

**Phase 3: Resource Browsing (optional)** — Add resource type handlers in `resource_providers/`. Test listing endpoints.

**Phase 4: Tools** — Create `src/seer/tools/myservice/` directory. Implement tool classes extending `BaseTool` with `integration_type`, `provider`, `required_scopes`, `get_parameters_schema()`, and `execute()`. Register in `tools/__init__.py`.

**Phase 5: Testing** — Verify `GET /api/integrations/metadata`, add OAuth + tool execution tests.

## Key Files

| File | Purpose |
|------|---------|
| `src/seer/services/integrations/metadata.py` | `INTEGRATION_CONFIGS` |
| `src/seer/services/integrations/providers/base.py` | Base classes |
| `src/seer/services/integrations/providers/<provider>.py` | Provider implementation |
| `src/seer/services/integrations/providers/__init__.py` | Provider registration |
| `src/seer/services/integrations/auth/oauth.py` | OAuth client config + provider mapping |
| `src/seer/tools/<provider>/` | Tool implementations |
| `src/seer/config.py` | Env vars |

## Reference Implementations
- **Google** (`providers/google.py`) — OpenID Connect, incremental auth
- **GitHub** (`providers/github.py`) — OAuth with API profile fetch
- **Supabase** (`providers/supabase.py`) — API key + OAuth hybrid
- **Discord** (`providers/discord.py`) — Bot installation flow

## Troubleshooting
- Integration not in frontend → check `INTEGRATION_CONFIGS` and icon URL accessibility
- "Provider not configured" → check registration in `providers/__init__.py` and env vars
- "Missing required scopes" → check tool `required_scopes` matches OAuth config
- "Token refresh failed" → ensure `access_type=offline` and correct `access_token_url`

## Related Skills
- `/oauth-integration` · `/integration-providers` · `/integration-metadata` · `/resource-browser` · `/tools-creation` · `/credential-resolution`

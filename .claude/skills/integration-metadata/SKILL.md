---
name: integration-metadata
description: Context about how integration metadata configures frontend display in seer. Use when understanding INTEGRATION_CONFIGS, icons, scopes, or how the dynamic UI system works.
allowed-tools: Read, Grep, Glob
---

# Integration Metadata Architecture

Frontend is fully dynamic — no hardcoded integration lists. Everything comes from `GET /api/integrations/metadata`.

## Flow

`INTEGRATION_CONFIGS (metadata.py)` → `get_all_integration_metadata()` → `GET /api/integrations/metadata` → frontend renders integration cards, scope selectors, connect buttons

## INTEGRATION_CONFIGS Fields

| Field | Purpose |
|-------|---------|
| `display_name` | Human-readable name |
| `oauth_provider` | Maps to OAuth provider (e.g. `"google"` for gmail/drive/sheets) |
| `requires_oauth` | True for OAuth, False for API key only |
| `icon` | `{type: "url"\|"lucide"\|"svg", value: "..."}` — prefer Logo.dev URL |
| `brand_color` | Hex color for UI accents |
| `default_scopes` | Scopes requested by default |
| `scopes` | Array of `{value, display_name, description}` |
| `detection_patterns` | `{tool_name_patterns, scope_keywords}` for tool-to-integration matching |

## Provider Mapping

`get_oauth_provider()` in `auth/oauth.py` maps integration types to OAuth providers. Multiple integration types can share one provider (gmail/googledrive/googlesheets → google). Integration types are user-facing; providers are auth system identifiers.

## Dynamic Metadata Assembly

`get_all_integration_metadata()` collects from `INTEGRATION_CONFIGS`, discovers additional integration types from registered tools, merges scope info, and builds `provider_to_types` map. Tools with scopes not in static config get those scopes auto-added. Integration types with tools but no static config get fallback metadata (lucide "Wrench" icon, title-cased name).

## Key Files

| File | Purpose |
|------|---------|
| `src/seer/services/integrations/metadata.py` | `INTEGRATION_CONFIGS` + metadata functions |
| `src/seer/api/integrations/metadata_models.py` | Pydantic models for API response |
| `src/seer/services/integrations/auth/oauth.py` | `get_oauth_provider()` mapping |
| `src/seer/api/integrations/router.py` | Metadata API endpoint |

## Related Skills
- `/oauth-integration` · `/add-integration`

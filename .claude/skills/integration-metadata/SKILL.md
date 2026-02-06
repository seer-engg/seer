---
name: integration-metadata
description: Context about how integration metadata configures frontend display in seer. Use when understanding INTEGRATION_CONFIGS, icons, scopes, or how the dynamic UI system works.
allowed-tools: Read, Grep, Glob
---

# Integration Metadata Architecture

This skill explains how integration metadata drives the frontend's dynamic UI rendering. No frontend code changes are needed for new integrations - it's all backend configuration.

## Architecture Overview

```
INTEGRATION_CONFIGS (metadata.py)
    ↓
get_all_integration_metadata()
    ↓
GET /api/integrations/metadata
    ↓
Frontend renders integrations dynamically
    ├── Icons (from config)
    ├── Names (from config)
    ├── Scopes (from config)
    └── Connect buttons
```

## Key Component: INTEGRATION_CONFIGS

Located in `src/seer/services/integrations/metadata.py`:

```python
INTEGRATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gmail": {
        "display_name": "Gmail",
        "oauth_provider": "google",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/gmail.com?token=..."},
        "brand_color": "#EA4335",
        "default_scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
        "scopes": [
            {
                "value": "https://www.googleapis.com/auth/gmail.readonly",
                "display_name": "Read emails",
                "description": "Read email messages and labels"
            },
            # ... more scopes
        ],
        "detection_patterns": {
            "tool_name_patterns": ["gmail_"],
            "scope_keywords": ["gmail"]
        }
    },
    # ... more integrations
}
```

## Configuration Fields

### Core Identity

| Field | Type | Purpose |
|-------|------|---------|
| `display_name` | string | Human-readable name shown in UI |
| `oauth_provider` | string | OAuth provider name (maps to `oauth.py` registration) |
| `requires_oauth` | bool | True for OAuth, False for API key only |

### Visual Configuration

**Icon Types**:

```python
# URL (recommended - use Logo.dev)
"icon": {"type": "url", "value": "https://img.logo.dev/gmail.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"}

# Lucide icon name
"icon": {"type": "lucide", "value": "Mail"}

# Inline SVG
"icon": {"type": "svg", "value": "<svg>...</svg>"}
```

**Brand Color**: Hex color for UI accents:
```python
"brand_color": "#EA4335"  # Gmail red
```

### Scope Configuration

**default_scopes**: Scopes requested by default when connecting:
```python
"default_scopes": ["https://www.googleapis.com/auth/gmail.readonly"]
```

**scopes**: Array of available scopes with metadata:
```python
"scopes": [
    {
        "value": "https://www.googleapis.com/auth/gmail.readonly",
        "display_name": "Read emails",           # Shown in UI
        "description": "Read email messages..."  # Tooltip/details
    }
]
```

### Detection Patterns

Used to match tools to integrations:
```python
"detection_patterns": {
    "tool_name_patterns": ["gmail_"],  # Tool names starting with "gmail_"
    "scope_keywords": ["gmail"]         # Scopes containing "gmail"
}
```

## Provider Mapping

The `get_oauth_provider()` function in `auth/oauth.py` maps integration types to OAuth providers:

```python
def get_oauth_provider(integration_type: str) -> str:
    google_integrations = ['gmail', 'googlesheets', 'googledrive', 'google']
    if integration_type in google_integrations:
        return 'google'

    github_integrations = ['github', 'pull_request']
    if integration_type in github_integrations:
        return 'github'

    # For others, integration type = provider
    return integration_type
```

**Why this exists**: Multiple integration types (Gmail, Drive, Sheets) share the same OAuth provider (Google) and connection.

## Dynamic Metadata Assembly

The `get_all_integration_metadata()` function:

1. **Collects from static config**: Iterates INTEGRATION_CONFIGS
2. **Discovers from tools**: Finds integration types from registered tools
3. **Merges scope information**: Adds scopes discovered from tools not in static config
4. **Builds provider mapping**: Creates `provider_to_types` map

```python
def get_all_integration_metadata() -> Dict[str, Any]:
    integrations = []
    all_types = _discover_integration_types()

    for int_type in sorted(all_types):
        metadata = _get_integration_metadata(int_type)
        if metadata:
            integrations.append(metadata)
        else:
            # Fallback for tools without config
            integrations.append(_create_fallback_metadata(int_type))

    return {
        "integrations": integrations,
        "provider_to_types": _build_provider_to_types_map()
    }
```

## Scope Discovery from Tools

Scopes are collected from registered tools to ensure UI shows all available scopes:

```python
def _collect_scopes_from_tools(integration_type: str) -> Set[str]:
    scopes: Set[str] = set()
    for tool in list_tools():
        if tool.integration_type == integration_type:
            scopes.update(tool.required_scopes or [])
    return scopes
```

If a tool uses a scope not in `INTEGRATION_CONFIGS`, it's auto-added with a generated display name.

## API Response

**Endpoint**: `GET /api/integrations/metadata`

**Response Structure**:
```json
{
    "integrations": [
        {
            "type": "gmail",
            "display_name": "Gmail",
            "oauth_provider": "google",
            "requires_oauth": true,
            "icon": {"type": "url", "value": "..."},
            "brand_color": "#EA4335",
            "default_scopes": ["..."],
            "scopes": [...],
            "detection_patterns": {...}
        }
    ],
    "provider_to_types": {
        "google": ["gmail", "google_drive", "google_sheets"],
        "github": ["github", "pull_request"]
    }
}
```

## Frontend Usage

The frontend calls this endpoint on startup and uses it to:
1. **Render integration cards** with icons and names
2. **Show scope selection** when connecting
3. **Match tools to integrations** via detection patterns
4. **Group integrations by provider** for OAuth connection sharing

## Fallback Metadata

For integration types with tools but no static config:

```python
def _create_fallback_metadata(integration_type: str) -> Dict[str, Any]:
    return {
        "type": integration_type,
        "display_name": integration_type.replace("_", " ").title(),
        "oauth_provider": provider,  # Inferred from tools
        "icon": {"type": "lucide", "value": "Wrench"},  # Default icon
        "scopes": [...],  # Collected from tools
        "detection_patterns": {
            "tool_name_patterns": [f"{integration_type}_"],
            "scope_keywords": [integration_type]
        }
    }
```

## Design Patterns

### Backend-Driven UI

The frontend is completely dynamic:
- No hardcoded integration lists
- No hardcoded icons or colors
- Everything comes from the metadata API

### Scope Accumulation

Scopes from multiple sources are merged:
1. Static config in `INTEGRATION_CONFIGS`
2. Dynamic discovery from registered tools

### Provider Abstraction

Integration types are user-facing categories (Gmail, Google Drive), while OAuth providers are the underlying auth system (Google). This separation allows:
- Multiple integrations to share one OAuth connection
- Clean UI organization by integration type
- Simplified OAuth management by provider

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/seer/services/integrations/metadata.py` | INTEGRATION_CONFIGS and metadata functions |
| `src/seer/api/integrations/metadata_models.py` | Pydantic models for API response |
| `src/seer/services/integrations/auth/oauth.py` | `get_oauth_provider()` mapping |
| `src/seer/api/integrations/router.py` | Metadata API endpoint |

## Related Skills

- `/oauth-integration` - OAuth flow and token management
- `/add-integration` - Step-by-step instructions for adding integrations

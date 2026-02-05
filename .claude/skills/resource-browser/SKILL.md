---
name: resource-browser
description: Context about the resource picker/browser system in seer. Use when understanding how resources are browsed, ResourceProvider implementations, or debugging resource picker issues.
allowed-tools: Read, Grep, Glob
---

# Resource Browser Architecture

This skill explains how the resource picker/browser system works - enabling users to browse and select resources from external services (files, repos, channels, etc.).

## Architecture Overview

```
Frontend (ResourcePicker UI)
    ↓
GET /api/integrations/resources/{provider}/{resource_type}
    ↓
ResourceBrowser.list_resources()
    ↓
ResourceProviderRegistry.get(provider)
    ↓
ResourceProvider.list_resources()
    ├── OAuth tokens → External APIs (Google, GitHub)
    ├── Bot tokens → Discord API
    └── Database → IntegrationResource table
    ↓
Paginated response with items
```

## Core Components

### ResourceContext (`resource_providers/base.py`)

Authentication context supporting multiple auth types:

```python
@dataclass(frozen=True)
class ResourceContext:
    user: Any                              # User object
    access_token: Optional[str] = None     # OAuth token
    bot_token: Optional[str] = None        # Discord bot token
    api_key: Optional[str] = None          # API key
    service_key: Optional[str] = None      # Supabase service role key
    custom_auth: Optional[Dict] = None     # Provider-specific auth
```

### ResourceProvider (`resource_providers/base.py`)

Abstract base class for provider-specific resource browsers:

```python
class ResourceProvider:
    provider: str                          # Provider name
    aliases: Set[str] = set()              # Alternative names
    resource_configs: Dict[str, Dict] = {} # Supported resource types

    def matches_provider(self, name: str) -> bool: ...
    def supports_resource_type(self, type: str) -> bool: ...
    def get_supported_resource_types(self) -> List[str]: ...

    async def list_resources(
        self, *,
        resource_type: str,
        query: Optional[str],           # Search query
        parent_id: Optional[str],       # For hierarchy
        page_token: Optional[str],      # Pagination
        page_size: int,
        filter_params: Optional[Dict],
        depends_on_values: Optional[Dict],  # Dependent values
        context: Optional[ResourceContext],
    ) -> Dict[str, Any]: ...
```

### ResourceProviderRegistry

Registry for looking up providers:

```python
class ResourceProviderRegistry:
    def register(self, provider: ResourceProvider): ...
    def get(self, provider_name: str) -> Optional[ResourceProvider]: ...
    def find_by_resource_type(self, resource_type: str) -> Optional[ResourceProvider]: ...
    def resource_configs(self) -> Dict[str, Dict]: ...
    def supports(self, provider: str, resource_type: str) -> bool: ...
```

## Provider Implementations

### Google Provider (`resource_providers/google.py`)

**Resource Types**:
- `google_drive_file` - Files and folders
- `google_spreadsheet` - Spreadsheets only
- `google_sheet_tab` - Tabs within a spreadsheet
- `google_drive_folder` - Folders only
- `gmail_label` - Gmail labels

**Features**:
- Hierarchy support (folder navigation)
- Search via Google Drive API
- Uses OAuth access token

### GitHub Provider (`resource_providers/github.py`)

**Resource Types**:
- `github_repo` - Repositories
- `github_branch` - Branches (depends on repo)

**Features**:
- Search for repos
- Branches depend on `repo` value
- Uses OAuth access token

### Discord Provider (`resource_providers/discord.py`)

**Resource Types**:
- `guild` - Discord servers (database-backed)
- `channel` - Channels within a guild (API-backed)

**Features**:
- Guilds stored in `IntegrationResource` table
- Channels fetched from Discord API with bot token
- Channels depend on `guild_id`

**Hybrid Source Pattern**:
```python
async def list_resources(self, *, resource_type, context, depends_on_values, ...):
    if resource_type == "guild":
        # Query database
        resources = await IntegrationResource.filter(
            user=context.user,
            provider="discord",
            resource_type="guild",
        ).all()
        return self._format_guilds(resources)

    elif resource_type == "channel":
        # Fetch from Discord API
        guild_id = depends_on_values.get("guild_id")
        return await self._fetch_channels(context.bot_token, guild_id)
```

### Supabase Provider (`resource_providers/supabase.py`)

**Resource Types**:
- `supabase_project` - Projects (from Management API OAuth)
- `schema` - Database schemas (from RPC)
- `table` - Tables within a schema (from RPC)

**Features**:
- Projects via OAuth access token
- Schemas/tables via service role key RPC calls
- Dependencies: table → schema → project

## API Endpoints

### List Resource Types

```
GET /api/integrations/resources/types
    → All supported resource types across all providers

GET /api/integrations/resources/{provider}/types
    → Resource types for specific provider
```

### Browse Resources

```
GET /api/integrations/resources/{provider}/{resource_type}
    Parameters:
    - q: Search query
    - parent_id: Parent folder (for hierarchy)
    - page_token: Pagination token
    - page_size: Items per page (max 100)
    - depends_on: JSON object with dependent values
```

**Example**: List channels in a Discord guild:
```
GET /api/integrations/resources/discord/channel?depends_on={"guild_id":"123456"}
```

## Resource Response Format

```json
{
    "items": [
        {
            "id": "file_123",
            "name": "My Document",
            "display_name": "My Document",
            "type": "google_drive_file",
            "metadata": {
                "mimeType": "application/pdf",
                "modifiedTime": "2024-01-15T10:30:00Z"
            }
        }
    ],
    "next_page_token": "abc123",
    "supports_search": true,
    "supports_hierarchy": true
}
```

## Tool Integration

Tools declare resource picker support via `get_resource_pickers()`:

```python
class GoogleSheetsReadTool(BaseTool):
    def get_resource_pickers(self) -> Dict[str, ResourcePickerConfig]:
        return {
            "spreadsheet_id": {
                "resource_type": "google_spreadsheet",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": False,
            }
        }
```

The schema also supports `x-resource-picker` extension:
```json
{
    "spreadsheet_id": {
        "type": "string",
        "x-resource-picker": {
            "resource_type": "google_spreadsheet",
            "display_field": "name",
            "value_field": "id"
        }
    }
}
```

## Dependent Resource Chains

Resources can depend on other resources:

```
Discord: guild → channel
GitHub: repo → branch
Supabase: project → schema → table
```

**Frontend Flow**:
1. User selects guild from picker
2. Frontend passes `depends_on={"guild_id": "selected_value"}`
3. Channel picker shows channels from that guild

**Agent Clarification Flow**:
```python
ask_clarification_questions([
    {
        "question": "Which Discord server?",
        "question_type": "resource_picker",
        "resource_type": "guild",
    },
    {
        "question": "Which channel?",
        "question_type": "resource_picker",
        "resource_type": "channel",
        "depends_on": "q_guild",  # References previous question
        "depends_on_field": "guild_id",
    }
])
```

## Pagination

Offset-based pagination via `page_token`:

```python
def paginate_items(items: List, page_token: Optional[str], page_size: int):
    offset = parse_offset(page_token) or 0
    page = items[offset:offset + page_size]
    next_token = str(offset + page_size) if offset + page_size < len(items) else None
    return {
        "items": page,
        "next_page_token": next_token,
    }
```

## Key Design Patterns

### Unified Interface

`ResourceBrowser` facade hides provider differences:
- Same API for Google Drive files and Discord guilds
- Frontend doesn't care about underlying implementation

### Multi-Auth Support

`ResourceContext` supports various auth types:
- OAuth tokens for Google, GitHub
- Bot tokens for Discord
- Service keys for Supabase

### Hybrid Sources

Some providers mix database and API sources:
- Discord guilds: stored in `IntegrationResource` table
- Discord channels: fetched from API
- Supabase projects: OAuth API + stored resources

### Lazy Loading

Resources are fetched on-demand:
- No pre-population of all resources
- Pagination for large result sets
- Search to filter server-side

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/seer/services/integrations/resource_browser.py` | ResourceBrowser facade |
| `src/seer/services/integrations/resource_providers/base.py` | Base classes, registry |
| `src/seer/services/integrations/resource_providers/__init__.py` | Provider registration |
| `src/seer/services/integrations/resource_providers/google.py` | Google Drive/Sheets/Gmail |
| `src/seer/services/integrations/resource_providers/github.py` | GitHub repos/branches |
| `src/seer/services/integrations/resource_providers/discord.py` | Discord guilds/channels |
| `src/seer/services/integrations/resource_providers/supabase.py` | Supabase projects/schemas |
| `src/seer/api/integrations/resource_router.py` | API endpoints |

## Related Skills

- `/tools-creation` - How tools integrate with resource pickers
- `/add-integration` - Step-by-step instructions for adding resource support

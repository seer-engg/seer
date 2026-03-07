---
name: resource-browser
description: Context about the resource picker/browser system in seer. Use when understanding how resources are browsed, ResourceProvider implementations, or debugging resource picker issues.
allowed-tools: Read, Grep, Glob
---

# Resource Browser Architecture

Enables users to browse external service resources (files, repos, channels) for tool parameter binding.

## Flow

`Frontend ResourcePicker` → `GET /api/integrations/resources/{provider}/{resource_type}` → `ResourceBrowser.list_resources()` → `ResourceProviderRegistry.get(provider)` → `ResourceProvider.list_resources()` → paginated response

## Provider Implementations

**Google** — types: `google_drive_file`, `google_spreadsheet`, `google_sheet_tab`, `google_drive_folder`, `gmail_label`. Hierarchy + search. File: `resource_providers/google.py`

**GitHub** — types: `github_repo`, `github_branch` (depends on repo). File: `resource_providers/github.py`

**Discord** — types: `guild` (DB-backed via `IntegrationResource`), `channel` (Discord API + bot token, depends on `guild_id`). Hybrid source. File: `resource_providers/discord.py`

**Supabase** — types: `supabase_project` (OAuth Mgmt API), `schema`, `table` (RPC + service role key). Chain: project → schema → table. File: `resource_providers/supabase.py`

## Dependent Resource Chains

`Discord: guild → channel` | `GitHub: repo → branch` | `Supabase: project → schema → table`

Frontend passes `depends_on={"guild_id": "value"}` to filter child resources. Agent clarification questions support `depends_on` referencing prior answers.

## Tool Integration

Tools declare resource picker support via `get_resource_pickers()` or `x-resource-picker` JSON Schema extension on parameter fields.

## Key Design Patterns
- **Unified interface** — `ResourceBrowser` facade hides provider differences
- **Multi-auth** — `ResourceContext` supports OAuth tokens, bot tokens, API keys, service keys
- **Hybrid sources** — Discord guilds (DB) vs channels (API)
- **Lazy loading** — on-demand with pagination

## Key Files

| File | Purpose |
|------|---------|
| `src/seer/services/integrations/resource_browser.py` | `ResourceBrowser` facade |
| `src/seer/services/integrations/resource_providers/base.py` | `ResourceProvider`, `ResourceContext`, registry |
| `src/seer/api/integrations/resource_router.py` | API endpoints |

## Related Skills
- `/tools-creation` · `/add-integration`

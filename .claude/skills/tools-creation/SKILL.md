---
name: tools-creation
description: Context about the tools system architecture in seer. Use when understanding BaseTool, tool registry, execution patterns, or debugging tool issues.
allowed-tools: Read, Grep, Glob
---

# Tools System Architecture

Extensible framework for integrating external services with workflows.

## Flow

`register_tool() at import time` → `_TOOL_REGISTRY[name] = tool` → `execute_tool()` called by workflow runtime → `CredentialResolver.resolve()` (OAuth lookup, token refresh, resource binding, secrets) → `tool.execute(access_token, arguments, credentials)`

## BaseTool Attributes (`src/seer/tools/base.py`)

| Attribute | Purpose |
|-----------|---------|
| `name` / `description` | Identifier + human label |
| `required_scopes` / `provider` | OAuth scopes + provider name |
| `integration_type` | Category (e.g. `"gmail"`) |
| `required_secrets` / `default_resource` | Secrets + auto-resolved resource |

Key methods: `execute()` (required), `get_parameters_schema()`, `get_resource_pickers()`, `get_output_schema()`, `get_metadata()`.

**Structure:** `src/seer/tools/` → `base.py · executor.py · __init__.py` · `google/` (GoogleAPIClient) · `github/` · `discord/` (DiscordAPIClient) · `supabase/`. Tools auto-register at module level via `register_tool(MyTool())`; import in `__init__.py` triggers it. Idempotent.

## Key Design Patterns
- **Inheritance** — `GoogleAPIClient` / `DiscordAPIClient` abstract provider-specific HTTP boilerplate
- **Credential abstraction** — tools don't manage OAuth; executor handles lookup, refresh, scope validation
- **Provider inference** — `tool.provider` → `tool.integration_type` → OAuth connection → resource binding
- **Modern execute signature** — `execute(access_token, arguments, credentials=None)` exposes resource + secrets

## Key Files

| File | Purpose |
|------|---------|
| `src/seer/tools/base.py` | `BaseTool`, registry functions |
| `src/seer/tools/executor.py` | `execute_tool()` |
| `src/seer/tools/__init__.py` | Tool import/registration |
| `src/seer/tools/google/base.py` | `GoogleAPIClient` |
| `src/seer/tools/discord/base.py` | `DiscordAPIClient` |

## Related Skills
- `/credential-resolution` · `/resource-browser` · `/add-integration`

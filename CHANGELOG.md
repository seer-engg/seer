All notable changes to this project will be documented in this file.

# Project Updates & Roadmap

## 🔭 Roadmap

### Planned
- support for teams and workspaces
- interruptability in workflows
- simple markdown template viewer in email body in gmail_draft
- loading table schema in supabase connector
- multiple accounts support per integration

### Researching
- use business logic via MCP

### Consideration
- generic seer mcp server

---

## 🏗️ In Progress

### Engineering
- backend comprehensive test suite
    - units tests : core, services, tools
    - integrations tests: database, tools , triggers
    - e2e tests : apis

### Stability
- seer agent for workflow builder
    - create/update existing workflows with agent

### Feature
- discord chat bot trigger
- guardrails on costs / tokens
    -  for agent
    -  llm nodes in workflows

### UX/UI
- agent first interface
- triggers
    - clean & minimal trigger config panel
    - easy code snippets for webhooks


## ✅ Released (History)

## [0.1.5] - 2026-01-20

### Added
- workflow spec v2 schema :
    - root level trigger objects
    - edges instead of nested json
    - inline ui meta in nodes/triggers/edges
    - support "Diamond Pattern" (Split-Merge)
    - remove inputs property , rely on trigger events for inputs
    - refrence resolution for triggers "${trigger_title.data.value}"
- "run" now uses sample trigger events
- Auto-connect edges in workflow canvas UI

### Deprecated
- Older workflow schema format (will no longer be supported)

### Fixed
- form triggers routing issue
- supabase auth coonnection timeouts

## [0.1.4] - 2026-01-15

### Added
- LLM usage gates to manage credit consumption
- Google Analytics 4 integration for docs.getseer.dev with custom event tracking
- Anniversary billing cycle support
- Automatic frontend opening in development mode

### Fixed
- OAuth token exchange error handling improvements
- Intermittent OAuth callback failures with multiple workers
- Stripe customer creation race condition
- Usage tracking bugs
- LLM credit exhaustion capture and exception handling
- Supabase callback URL exemption

### Changed
- Increased timeout limits for long-running workflows
- Read limits from .env for easier override
- Removed Ultra pricing tier

### Removed
- Unnecessary local configuration files

## [0.0.9] - 2026-01-09

Initial public release with core workflow automation features.

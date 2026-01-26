# Nexus Interactive Testing Scripts

This directory contains test scripts for the Nexus chat API:
- `test_nexus_simple.py` - Simple version (recommended)
- `test_nexus_local.py` - Full-featured version with advanced options

## Quick Start

**Simplest way to test Nexus:**

```bash
# 1. Start services
docker compose up

# 2. Run the simple test script
uv run python scripts/test_nexus_simple.py
```

That's it! The script will:
- Auto-create a test workflow
- Start an interactive chat session
- Connect to `http://localhost:8000`

Type messages like "Build a workflow that sends daily email reports" and interact with Nexus proposals.

## Prerequisites

1. **Start local services:**
   ```bash
   docker compose up postgres valkey -d
   uv run uvicorn seer.api.main:app --reload --port 8000
   ```

2. **Enable chat in self-hosted mode:**
   Add to `.env`:
   ```
   CHAT_MESSAGES_TOTAL_SELF_HOSTED=-1
   ```

3. **Database setup:**
   Ensure database migrations are applied. The script requires tables: `users`, `workflows`, `workflow_drafts`, `workflow_chat_sessions`, `workflow_chat_messages`, `workflow_proposals`, `usage_counters`.

## Usage

### Simple Version (Recommended)

```bash
uv run python scripts/test_nexus_simple.py
```

No configuration needed - just run and start chatting!

### Advanced Version with Options

```bash
# Basic usage
uv run python scripts/test_nexus_local.py

# Supervisor mode (multi-agent architecture)
uv run python scripts/test_nexus_local.py --supervisor-mode

# Use existing workflow
uv run python scripts/test_nexus_local.py --workflow-id wf_123

# Custom API URL
uv run python scripts/test_nexus_local.py --base-url http://localhost:3000
```

## Interactive Commands

### Simple Version Commands

- Type your message to chat with Nexus
- `quit` - Exit the chat

When a proposal is offered:
- `yes` - Accept the proposal
- `no` - Reject the proposal
- `view` - View full proposal spec JSON

### Advanced Version Commands

All simple version commands, plus:
- `/clear` - Start a new session
- `/help` - Show help message

## Example Session

```
$ uv run python scripts/test_nexus_simple.py

✓ Connected to API at http://localhost:8000
✓ Created test workflow: wf_abc123

================================================================================
Nexus Interactive Chat
================================================================================

Type your messages to chat with Nexus. Type 'quit' to exit.

Example prompts:
  • Build a workflow that sends daily email reports
  • Create a workflow to sync data from Airtable to Notion
  • Design a customer onboarding workflow

You > Build a workflow that sends daily email reports

Sending message...

[Thinking]
  • Analyzing requirements for email report workflow
  • Identifying required tools and integrations
  • Designing workflow structure

Nexus: I can help you build an email report workflow! Here's what I'm proposing...

📋 Workflow Proposal #1
Summary: Daily email report workflow with scheduler and email integration
Nodes: 3 (trigger, transform, email)

Accept proposal? (yes/no/view): yes

Accepting proposal...
✓ Proposal accepted!
Updated workflow has 3 nodes

You > quit
Goodbye!
```

## Features

- **Color-coded output** for easy reading
- **Automatic authentication** via JWT tokens
- **Session persistence** across multiple messages
- **Proposal management** with accept/reject/view options
- **Thinking steps** display for transparency
- **Error handling** with helpful messages

## Troubleshooting

**"Chat AI is not available in self-hosted mode"**
- Add `CHAT_MESSAGES_TOTAL_SELF_HOSTED=-1` to `.env` and restart API

**"relation does not exist" errors** (e.g., `usage_counters`, `llm_usage_records`)
- Apply database migrations: `uv run aerich upgrade`
- Verify tables exist: `docker compose exec -T postgres psql -U postgres -d seer -c "\dt usage_counters"`
- Required tables: `users`, `workflows`, `workflow_drafts`, `workflow_chat_sessions`, `workflow_chat_messages`, `workflow_proposals`, `usage_counters`, `llm_usage_records`

**Connection refused**
- Ensure API is running: `curl http://localhost:8000/health`
- Check Docker services: `docker compose ps`

**"Unauthorized" errors**
- Script auto-generates test JWT tokens (no external auth needed)
- Check logs for authentication middleware issues

## Architecture Notes

The script uses:
- `httpx` for async HTTP requests
- JWT tokens (unvalidated in local mode)
- Workflow state in ReactFlow format
- LangGraph thread IDs for session persistence

API endpoints used:
- `POST /api/v1/workflows` - Create workflow
- `POST /api/nexus/{workflow_id}/chat` - Chat with Nexus
- `POST /api/nexus/{workflow_id}/proposals/{id}/accept` - Accept proposal
- `POST /api/nexus/{workflow_id}/proposals/{id}/reject` - Reject proposal

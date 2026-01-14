---
sidebar_position: 1
---

# Configuration Reference

Complete environment variable reference for Seer configuration.

## Quick Start Configuration

For basic usage, you only need:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (for persistence)
DATABASE_URL=postgresql://user:password@host:port/database
REDIS_URL=redis://localhost:6379/0
```

Docker development automatically configures `DATABASE_URL` and `REDIS_URL`.

---

## Required Configuration

### LLM Services

| Variable | Description | How to Get |
|----------|-------------|------------|
| `OPENAI_API_KEY` | OpenAI API key for workflow execution and AI assistance | [OpenAI Platform](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | Alternative to OpenAI for Claude models | [Anthropic Console](https://console.anthropic.com/) |

At least one LLM API key is required.

---

## Core Services (Docker Auto-Configured)

### Database

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Auto-configured in Docker |

**Format**: `postgresql://user:password@host:port/database`

**Docker**: Automatically set to `postgresql://postgres:postgres@postgres:5432/seer`

**Railway**: Use the `DATABASE_URL` provided by Railway Postgres service

### Redis

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection string for task queue | Auto-configured in Docker |

**Format**: `redis://host:port/db`

**Docker**: Automatically set to `redis://redis:6379/0`

**Railway**: Use the `REDIS_URL` provided by Railway Redis service

---

## OAuth Integrations

### Google Workspace

| Variable | Description | How to Get |
|----------|-------------|------------|
| `GOOGLE_CLIENT_ID` | OAuth client ID for Google integrations | [Google Cloud Console](https://console.cloud.google.com/) |
| `GOOGLE_CLIENT_SECRET` | OAuth client secret | Same as above |

**Enables**: Gmail, Google Drive, Google Sheets tools

### GitHub

| Variable | Description | How to Get |
|----------|-------------|------------|
| `GITHUB_CLIENT_ID` | OAuth client ID for GitHub integration | [GitHub Developer Settings](https://github.com/settings/developers) |
| `GITHUB_CLIENT_SECRET` | OAuth client secret | Same as above |
| `GITHUB_TOKEN` | Personal access token for repo access | [GitHub Tokens](https://github.com/settings/tokens) |

**Enables**: Repository management, issues, pull requests

### Supabase

| Variable | Description | How to Get |
|----------|-------------|------------|
| `SUPABASE_CLIENT_ID` | OAuth client ID for Supabase management | [Supabase Dashboard](https://supabase.com/dashboard) |
| `SUPABASE_CLIENT_SECRET` | OAuth client secret | Same as above |
| `SUPABASE_MANAGEMENT_API_BASE` | Management API base URL | Default: `https://api.supabase.com` |

**Enables**: Supabase project management, REST API tools

---

## Web Services & APIs

### Web Search

| Variable | Description | How to Get |
|----------|-------------|------------|
| `TAVILY_API_KEY` | API key for web search capabilities | [Tavily](https://tavily.com/) |

### Tool Search (Optional)

| Variable | Description | How to Get |
|----------|-------------|------------|
| `PINECONE_API_KEY` | Pinecone API key for semantic tool search | [Pinecone](https://app.pinecone.io/) |
| `PINECONE_INDEX_NAME` | Pinecone index name | Create in Pinecone console |
| `CONTEXT7_API_KEY` | Context7 API key for MCP tools | [Context7](https://context7.com/) |

---

## Analytics & Monitoring

### PostHog

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTHOG_API_KEY` | PostHog project API key | - |
| `POSTHOG_HOST` | PostHog instance URL | - |
| `POSTHOG_ENABLED` | Enable analytics tracking | `false` |

**PostHog URLs**:
- Cloud US: `https://app.posthog.com`
- Cloud EU: `https://eu.posthog.com`
- Self-hosted: Your Railway/custom URL

---

## Authentication

### Clerk (Cloud Deployments)

| Variable | Description |
|----------|-------------|
| `CLERK_JWKS_URL` | Clerk JWKS endpoint for token verification |
| `CLERK_ISSUER` | Clerk issuer URL |

**Required for**: Cloud deployments with user authentication

---

## Advanced Configuration

### Database Management

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTO_APPLY_DATABASE_MIGRATIONS` | Run Aerich migrations on startup | `true` in Railway |

### Worker Configuration

See [Triggers Documentation](./TRIGGERS.md) for worker-specific configuration.

### Tool Indexing

| Variable | Description | Default |
|----------|-------------|---------|
| `TOOL_INDEX_AUTO_GENERATE` | Auto-generate tool search index | Enabled |

### Trigger Poller

| Variable | Description | Default |
|----------|-------------|---------|
| `TRIGGER_POLLER_ENABLED` | Enable automatic trigger polling | Enabled |

---

## Configuration Priority

Environment variables are loaded in this order (highest to lowest priority):

1. System environment variables
2. `.env` file in project root
3. Defaults in `shared/config.py`

---

## Viewing Current Configuration

Configuration is loaded from environment variables and `.env` file in this priority order:

1. System environment variables
2. `.env` file in project root
3. Defaults in `shared/config.py`

You can view your active configuration by checking:
- Your `.env` file for local overrides
- `shared/config.py` for default values and structure
- Docker container environment via `docker compose exec langgraph-server env | grep -E "OPENAI|ANTHROPIC|GOOGLE|GITHUB|TAVILY"`

---

## Related Documentation

- [Getting Started](/)
- [Railway Deployment](../deployment/RAILWAY)
- [Triggers Documentation](./TRIGGERS)

## Seer (`seeragents`)

Seer is a **workflow builder with fine-grained control** for creating and executing automated workflows with integrated tools and services. Build complex automation workflows with visual editing, AI-assisted development, and seamless integrations (Google Workspace, GitHub, and more).

> **Note:** Package name is `seeragents` on PyPI (name conflict), but CLI command is `seer`.

### Quick Start

```bash
git clone <repo> && cd seer
uv run seer dev
```

That's it! No installation needed. Starts Docker services (Postgres, MLflow, backend), installs dependencies in containers, tails logs, waits for readiness, and opens the workflow builder in your browser.

### Installation (Optional)

**Only needed if you want to use `seer` directly without `uv run`:**

**CLI only (lightweight):**
```bash
pip install "seeragents[cli]"  # or: uv pip install "seeragents[cli]"
```

**Full installation:**
```bash
pip install seeragents  # or: uv pip install seeragents
```

**Local development:**
```bash
git clone <repo> && cd seer
uv venv && source .venv/bin/activate
uv pip install -e ".[cli]"  # Install CLI only
rehash  # Refresh shell command cache (zsh) or restart terminal
seer dev  # Now you can use 'seer' directly
```

### Configuration

Create a `.env` file (automatically loaded):

```bash
# Required for workflow execution and AI assistance
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...  # Alternative to OpenAI

# Integrations
TAVILY_API_KEY=...  # For web search tools

# OAuth Configuration (for cloud deployments)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...

# Optional: Persistence and monitoring
DATABASE_URL=...  # PostgreSQL for workflow persistence
MLFLOW_TRACKING_URI=...  # MLflow for execution tracking

```

Check: `uv run seer config` or `seer config` (if installed)

### Usage

**Development:**
```bash
uv run seer dev  # Recommended: no installation needed
# or if installed:
seer dev
```

**Configuration:**
```bash
uv run seer config           # Show current configuration
uv run seer config --format json  # JSON output format
```

**Data Export:**
```bash
uv run seer export <thread-id>      # Export workflow execution results
uv run seer export <thread-id> --format markdown  # Export in markdown format
```

### Development Workflow

**What runs where:**
- **Local:** CLI tool (`seer` command) - lightweight (`click`/`rich` only)
- **Docker:** Backend API, Postgres, MLflow - all dependencies installed here

**Steps:**
1. Run: `uv run seer dev` (no installation needed!)
2. Code changes hot-reload via volume mounts (uvicorn --reload)
3. Access workflow builder at: http://localhost:5173/workflows?backend=http://localhost:8000
4. View logs: `docker compose logs -f`
5. Stop: `docker compose down`

**Services started:**
- **Backend API** (port 8000): FastAPI server with workflow execution engine
- **Postgres** (port 5432): Workflow and user data persistence
- **MLflow** (port 5001): Execution tracking and observability

### API Keys & Integrations

| Feature | Required Keys |
|---------|---------------|
| **Workflow Execution** | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` |
| **AI Chat Assistant** | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` |
| **GitHub Integration** | `GITHUB_TOKEN`, `GITHUB_CLIENT_ID/SECRET` |
| **Google Workspace** | `GOOGLE_CLIENT_ID/SECRET` |
| **Web Search** | `TAVILY_API_KEY` |
| **Persistence** | `DATABASE_URL` (PostgreSQL) |
| **Monitoring** | `MLFLOW_TRACKING_URI` |
| **Cloud Auth** | `CLERK_JWKS_URL`, `CLERK_ISSUER` |

**Supported Integrations:**
- **Google Workspace**: Gmail, Google Drive, Google Sheets
- **GitHub**: Repositories, Issues, Pull Requests
- **Web Tools**: Search, content fetching
- **Database**: PostgreSQL with read/write controls

Missing keys? Seer prompts interactively and supports OAuth flows.

### Key Features

**🛠️ Visual Workflow Builder**
- Drag-and-drop interface for creating automation workflows
- Node-based editor with custom blocks and integrations
- Real-time workflow validation and execution

**🤖 AI-Assisted Development**
- Chat interface for workflow design and debugging
- AI suggestions for workflow improvements
- Intelligent error handling and recovery

**🔗 Rich Integrations**
- **Google Workspace**: Gmail, Drive, Sheets with OAuth
- **GitHub**: Repository management, issues, PRs
- **Web Tools**: Search, content fetching, APIs
- **Databases**: PostgreSQL with approval-based write controls

**⚡ Advanced Execution Engine**
- Streaming execution with real-time updates
- Interrupt handling for human-in-the-loop workflows
- Persistent state management with PostgreSQL
- MLflow integration for observability

**🔒 Enterprise-Ready**
- Self-hosted or cloud deployment options
- OAuth-based authentication (Clerk integration)
- Role-based access control
- Audit trails and execution history

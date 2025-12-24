## Seer (`seeragents`)

Seer is a **LangGraph-based evaluation orchestrator** for testing autonomous agents end-to-end (align → generate tests → run tests), with optional telemetry (MLflow) and persistence (Postgres/Neo4j).

> **Note:** Package name is `seeragents` on PyPI (name conflict), but CLI command is `seer`.

### Quick Start

```bash
git clone <repo> && cd seer
uv run seer dev
```

That's it! No installation needed. Starts Docker services (Postgres, MLflow, backend), installs dependencies in containers, tails logs, waits for readiness, and opens your browser.

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
OPENAI_API_KEY=...
GITHUB_TOKEN=...
E2B_API_KEY=...  # For sandbox provisioning
DATABASE_URL=...  # Optional: Postgres persistence
MLFLOW_TRACKING_URI=...  # Optional: MLflow tracing
```

Check: `uv run seer config` or `seer config` (if installed)

### Usage

**Development:**
```bash
uv run seer dev  # Recommended: no installation needed
# or if installed:
seer dev
```

**Eval agent:**
```bash
uv run seer run  # Interactive loop (alignment → plan → testing → finalize)
uv run seer run --thread-id <uuid>  # Resume session
```

**Supervisor (database ops):**
```bash
uv run seer new-supervisor
```

**Other commands:**
```bash
uv run seer config           # Show configuration
uv run seer export <id>      # Export results
uv run seer -v run          # Verbose mode
```

### Development Workflow

**What runs where:**
- **Local:** CLI tool (`seer` command) - lightweight (`click`/`rich` only)
- **Docker:** Backend API, Postgres, MLflow - all dependencies installed here

**Steps:**
1. Run: `uv run seer dev` (no installation needed!)
2. Code changes hot-reload via volume mounts
3. Logs: `docker compose logs -f`
4. Stop: `docker compose down`

### API Keys

| Stage | Required |
|-------|----------|
| **alignment/plan** | `OPENAI_API_KEY` |
| **testing** | `OPENAI_API_KEY`, `GITHUB_TOKEN` |
| **sandbox** | `E2B_API_KEY` |
| **OAuth** | `GOOGLE_CLIENT_ID/SECRET`, `GITHUB_CLIENT_ID/SECRET` |
| **Optional** | `DATABASE_URL`, `MLFLOW_TRACKING_URI`, `NEO4J_URI` |

Missing keys? Seer prompts interactively.

### Python API

```python
from agents.eval_agent.graph import build_graph
from langgraph.checkpoint.memory import MemorySaver

graph = build_graph()
agent = graph.compile(checkpointer=MemorySaver())
result = await agent.ainvoke({"step": "alignment", ...})
```

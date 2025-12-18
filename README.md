## Seer Agents (`seeragents`)

Seer Agents is a **LangGraph-based evaluation orchestrator** for testing autonomous agents end-to-end (align → generate tests → run tests), with optional telemetry (Langfuse) and persistence (Neo4j/Postgres).

### Install (recommended: no git clone)

#### Install with `pip`

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U seeragents
```

#### Install with `uv`

```bash
uv venv
source .venv/bin/activate

uv pip install -U seeragents
```

### Configuration (env vars or `.env`)

Both CLIs (`seer`, `seer-eval`) automatically load a local `.env` file (via `python-dotenv`). You can also export environment variables directly.

Create a `.env` in your working directory:

```bash
OPENAI_API_KEY=...
GITHUB_TOKEN=...
COMPOSIO_API_KEY=...
E2B_API_KEY=...
```

You can inspect what Seer sees with:

```bash
seer-eval config
```

### API keys by stage (what’s required when)

Seer validates many keys up-front, and **`seer-eval` will prompt you interactively** if something is missing.

| Stage | Required keys | Notes |
| --- | --- | --- |
| **alignment** | `OPENAI_API_KEY` | Used to turn a human request into a concrete spec (functional requirements + required integrations). |
| **plan** | `OPENAI_API_KEY` | Generates dataset-style test cases. |
| **testing** | `OPENAI_API_KEY`, `GITHUB_TOKEN` | Always required for executing tests and provisioning target repos. |
| **testing (MCP)** | `COMPOSIO_API_KEY` | Required **only** when the aligned spec includes MCP services (e.g. GitHub/Asana tools). |
| **sandbox provisioning** | `E2B_API_KEY` | Required when Seer provisions an E2B sandbox to clone/build/run the target agent. |
| **Asana scenarios** | `ASANA_WORKSPACE_ID` or `ASANA_TEAM_GID` (or `ASANA_PROJECT_ID`) | Required when running tests that create/update Asana entities. |
| **optional tracing** | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | Enables Langfuse traces (recommended for debugging). |
| **optional persistence** | `DATABASE_URI` | Enables Postgres checkpointer for pause/resume across runs and richer history. |
| **optional memory** | `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | Enables reflection/tool indexing in Neo4j. |

### Usage: CLI (fastest path)

#### Run the full eval pipeline

```bash
seer-eval run "Evaluate my agent that syncs GitHub PR merges to Asana tasks" \
  --repo owner/repo \
  --user-id you@example.com
```

#### Run step-by-step (alignment → plan → test)

```bash
seer-eval align "Evaluate my agent..." --repo owner/repo
# copy the printed thread id

seer-eval plan --thread-id <thread-id>
seer-eval test --thread-id <thread-id>
```

#### About interactive prompts

- If a required key is missing, **`seer-eval` will ask you** for the value and continue.
- To avoid prompts (CI / automation), set keys via `.env` or exported env vars ahead of time.

### Usage: start the LangGraph dev server

This starts the eval agent as a LangGraph dev server (useful with LangGraph Studio / HTTP invocation):

```bash
seer
```

By default it serves the Eval Agent on `http://127.0.0.1:8002` and writes logs under `seer-logs/`.

### Usage: import and run from Python (notebook-style)

This mirrors the flow in `examples/github_asana_bot.ipynb`.

```python
import uuid
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()  # loads .env in the current directory (optional but recommended)

from agents.eval_agent.graph import build_graph

graph = build_graph()
memory = MemorySaver()
eval_agent = graph.compile(checkpointer=memory)

thread_id = str(uuid.uuid4())

# 1) alignment
aligned = await eval_agent.ainvoke(
    {
        "messages": [{"type": "human", "content": "Evaluate my agent..."}],
        "step": "alignment",
        "input_context": {
            "integrations": {"github": {"name": "owner/repo"}},
            "user_id": "you@example.com",
        },
    },
    config=RunnableConfig(configurable={"thread_id": thread_id}),
)

# 2) plan
planned = await eval_agent.ainvoke(
    {"step": "plan"},
    config=RunnableConfig(configurable={"thread_id": thread_id}),
)

# 3) testing
tested = await eval_agent.ainvoke(
    {"step": "testing"},
    config=RunnableConfig(configurable={"thread_id": thread_id}),
)
```

### Developing locally (optional: git clone)

If you’re contributing to Seer itself:

```bash
git clone <your-fork>
cd seer

uv venv
source .venv/bin/activate
uv pip install -e .
```
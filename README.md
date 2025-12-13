# 🔮 Seer – Multi-Agent Evaluation Orchestrator

Seer is a LangGraph-based multi-agent system that performance-tests autonomous software agents, repairs them when possible, and records the full lifecycle of each evaluation. Two graphs collaborate:

- **Eval Agent** plans and executes black-box test suites, uploads traces (Langfuse + Neo4j), and decides when to stop.
- **Codex Agent** (optional) accepts handoffs from the eval loop to triage failures, modify the target repo inside an E2B sandbox, and raise deployment artifacts when fixes land.

## Why Seer?

- **Graph-native workflows** – Both agents compile LangGraph state machines, making control flow explicit and observable.
- **Sandbox-first execution** – Target repos run inside E2B code sandboxes with reusable helpers in `sandbox/`.
- **Persistent memory** – Evaluation reflections and MCP tool embeddings are indexed in Neo4j (`graph_db/`), enabling cross-run learning.
- **Configurable automation** – Feature flags (e.g., `CODEX_HANDOFF_ENABLED`) live in `shared/config.py` so you can experiment without code changes.

## Architecture Overview

| Component | Responsibility | Key Files |
| --- | --- | --- |
| Eval Agent | Plans, provisions, executes, reflects, and finalizes eval rounds. | `agents/eval_agent/graph.py`, `agents/eval_agent/nodes/*` |
| Codex Agent | Plans → codes → tests → reflects inside a loop; can deploy or raise PRs. | `agents/codex/graph.py`, `agents/codex/nodes/*` |
| Shared services | Configuration, logging, MCP tooling, schema definitions, and the test runner. | `shared/` |
| Sandbox utilities | Connect to and run commands inside E2B sandboxes. | `sandbox/` |
| Graph DB client | Creates Neo4j vector indexes for reflections + tools. | `graph_db/client.py` |
| Indexer | Embedding + retrieval services for docs/tooling metadata. | `indexer/` |
| Launcher | `run.py` orchestrates LangGraph dev servers per agent and manages logs. | `run.py`, `logs/` (created on demand) |

## Prerequisites

- Python 3.12+
- [LangGraph CLI](https://python.langchain.com/docs/langgraph) (installed into your virtualenv)
- Neo4j instance (local or remote) if you want persistence for reflections/tools
- Valid API keys: OpenAI (required), plus Tavily, Langfuse, E2B, Pinecone, etc., depending on enabled features

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e .

# Copy and edit environment variables
cp .env.example .env
```

The package exposes a CLI entry point so you can also install globally and run `seer`.

## Configuration

All settings flow through `shared/config.py` (Pydantic). Frequently used variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required for every LangChain/LangGraph call. |
| `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `LANGFUSE_PROJECT_NAME` | Trace eval + codex runs (Langfuse self-hosted). |
| `CODEX_HANDOFF_ENABLED` | `true` enables the codex graph + port 8003 launch. |
| `E2B_API_KEY` | Required to spin up sandboxes for repo manipulation. |
| `NEO4J_*` | Connect Seer to a Neo4j instance for reflections + tool indexes. |
| `TAVILY_API_KEY`, `PINECONE_API_KEY` | Optional retrieval + tool features. |

See `.env.example` for the authoritative list. Any values omitted there still have defaults defined in `shared/config.py`.

## Running the Agents

```bash
# From the repo root with your virtualenv activated
seer           # or: python run.py
```

What happens:

1. `run.py` creates `logs/`, locates `venv/bin/python` + `langgraph`, and ensures `OPENAI_API_KEY` exists.
2. The Eval Agent graph is served via `langgraph dev --port 8002`.
3. If `CODEX_HANDOFF_ENABLED=true`, the Codex graph is served on port `8003`.
4. The launcher tails each process, strips ANSI codes, and restarts cleanly on Ctrl+C.

Logs per service live under `logs/*.log`.

## Evaluation Flow

1. **Planning** – `agents/eval_agent/nodes/plan` configures test cases, tool access, and provisioning steps.
2. **Execution** – `agents/eval_agent/nodes/execute` provisions environments, invokes target agents, and uploads telemetry to Langfuse + Neo4j.
3. **Reflection / Finalization** – Failures trigger reflection tooling to decide whether to stop or request a new target-agent version. Successful runs finalize and clean up resources before either ending or starting another round.
4. **Codex handoff (optional)** – If the eval loop needs fixes, it passes context + repo metadata to the Codex graph. Codex runs an initialize → plan → code → test → reflect loop (`agents/codex/graph.py`) and can raise PRs/deploy artifacts on success.

## Sample Eval Agent Request

### Sample Eval Agent Input
Evaluate my agent buggy_coder
1. The agent should sync Asana ticket updates when a GitHub PR is merged on it's own. Whenever i merge a PR it should search for realted asana tickets and update/close them.
2. I wanna turn Buggy Coder into an agent that can keep Asana tickets and project status in sync. For example, whenever an Asana ticket is opened or closed, the project description must be appropriately updated.
3. I want to turn buggycoder into a GitHub PR review bot. For example, whenever a new pull request is created, it should check if the pull request has a summary and a title. And if it doesn't, then it should try to understand the contents of the pull request and update the summary and title accordingly. This should also include edge cases where developer just enters One or two words such as bug fix or hotfix or feature, etc.
4. Evolve my agent at https://github.com/seer-engg/langgraph-skeleton. It should be a PR review bot that should watch for any new or updated raised pr in repo https://github.com/seer-engg/buggy-coder and provide a detailed review .

These prompts are typically provided through whatever orchestration layer is invoking the eval graph (e.g., LangGraph Studio or a bespoke controller).

## Troubleshooting

- **LangGraph CLI not found** – Install it inside your virtualenv (`pip install langgraph-cli[inmem]`) so `run.py` can resolve `venv/bin/langgraph`.
- **Sandbox failures** – Confirm `E2B_API_KEY` is valid and the template alias in `shared/config.py` exists.
- **Neo4j errors** – Ensure the database is reachable and your user can create vector indexes. The client lazily creates indexes on import.
- **Handoff disabled** – If you expect Codex to run, double-check `CODEX_HANDOFF_ENABLED=true`.

## Next Steps

- Customize tool loading by editing `shared/tools/registry.py`.
- Add new eval tactics by extending `agents/eval_agent/nodes/plan/*`.
- Instrument experiment runs under `experiments/` to keep investigations reproducible (see workspace rules).

Seer is intentionally modular—feel free to swap agents, add new subgraphs, or integrate additional toolchains as your evaluation needs evolve.
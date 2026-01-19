# Repository Guidelines

## Project Structure & Module Organization
- `src/seer/api/`: FastAPI HTTP layer (routers, middleware, API models).
- `src/seer/services/`: business logic used by API/worker (workflows, integrations, triggers).
- `src/seer/core/`: workflow compiler, runtime, schema models, and global compiler singleton.
- `src/seer/tools/`: tool registry, executor, credential resolution, and provider tool implementations.
- `src/seer/tool_hub/`: tool index/search helpers for discovery.
- `src/seer/agents/`: agent-specific orchestration (LangGraph-based workflow agent).
- `src/seer/worker/`: Taskiq background worker and polling tasks.
- `src/seer/database/`: Tortoise ORM models/configuration; migrations live in `/migrations`.
- `src/seer/analytics/`, `src/seer/observability/`, `src/seer/utilities/`: shared instrumentation and helpers.
- `documentation/`: docs site assets (Node-based).
- `scripts/`: maintenance and debugging scripts.

## Core Systems (Compiler, Tools, Triggers)
- Workflow compiler: `src/seer/core/` validates workflow specs, builds LangGraph graphs, and hosts runtime node executors; API/services/worker call into it via the global compiler singleton.
- Tool registry: `src/seer/tools/` provides `BaseTool`, registry helpers, executor, and credential resolution; discovery is powered by `src/seer/tool_hub/`.
- Trigger polling: trigger catalog + subscription management lives under `src/seer/api/workflows/services` and `src/seer/services/workflows/triggers.py`; Taskiq worker (`src/seer/worker/`) polls and dispatches runs.

## Build, Test, and Development Commands
- `docker compose up`: start Postgres, Redis, API, worker, and frontend via Docker.
- `uv run uvicorn seer.api.main:app --reload --port 8000`: run the API locally without Docker.
- `uv run taskiq worker seer.worker.broker:broker`: run the Taskiq background worker locally.
- `uv run pytest` (or `pytest src/seer/core/tests`): run the Python test suite.

## Coding Style & Naming Conventions
- Python 3.12, 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes.
- Line length limit is 150 characters (see `pyproject.toml`).
- Linting uses `pylint` and `pre-commit`; any `# pylint: disable=...` must include a reason comment.

## Testing Guidelines
- `pytest` + `pytest-asyncio` (asyncio mode is `auto`).
- Tests are named `test_*.py`; prefer `src/seer/core/tests/` or module-level test folders.
- Add regression tests for bug fixes and workflow schema/validation changes.

## Adding Tools, Triggers, and Workflow Nodes
- New tool: add a `BaseTool` in `src/seer/tools/<provider>/`, register via `register_tool(...)`, and ensure the module is imported by the loader. Provide a JSON schema in `get_parameters_schema()`.
- New trigger: add trigger definitions/subscriptions in `src/seer/api/workflows/services/triggers.py` (API surface) and polling logic in `src/seer/services/workflows/triggers.py`/`src/seer/worker/tasks/triggers.py`. Register adapters and polling intervals where appropriate.
- New workflow node type: add executor logic in `src/seer/core/runtime/nodes.py` and validate/transform rules in `src/seer/core/compiler.py` or schema models as needed.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits (e.g., `fix:`, `feat:`, `chore:`) based on recent history.
- PRs should include: a concise summary, linked issue (if any), testing notes, and screenshots for UI/doc changes.

## Security & Configuration Tips
- Keep secrets in `.env` (e.g., `OPENAI_API_KEY`, `GOOGLE_CLIENT_SECRET`) and never commit them.
- Docker compose injects `DATABASE_URL` and `REDIS_URL`; keep local overrides in `.env`.

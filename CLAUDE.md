# Repository Guidelines

## Project Structure & Module Organization
- `api/`: FastAPI HTTP layer (routers, services, middleware, API models).
- `worker/`: Taskiq background worker and polling tasks.
- `shared/`: shared config, analytics, database models, and utilities.
- `agents/`: agent-specific logic and orchestration.
- `workflow_compiler/`: workflow validation/compilation plus runtime helpers; tests live here.
- `migrations/`: Aerich database migrations.
- `documentation/`: docs site assets (Node-based).
- `scripts/`: maintenance and debugging scripts.

## Core Systems (Compiler, Tools, Triggers)
- Workflow compiler: `workflow_compiler/` turns workflow specs into LangGraph graphs and runs node executors; API calls into it.
- Tool registry: `shared/tools/` provides `BaseTool`, registry helpers, and execution with credential resolution.
- Trigger polling: `api/triggers/` defines triggers, subscriptions, poll adapters, and dedupe; worker executes polls and dispatches runs.

## Build, Test, and Development Commands
- `docker compose up`: start Postgres, Redis, API, worker, and frontend via Docker.
- `uvicorn api.main:app --reload --port 8000`: run the API locally without Docker.
- `uv run taskiq worker worker.broker:broker`: run the background worker locally.
- `uv run pytest` (or `pytest workflow_compiler/tests`): run the Python test suite.

## Coding Style & Naming Conventions
- Python 3.12, 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes.
- Line length limit is 150 characters (see `pyproject.toml`).
- Linting uses `pylint` and `pre-commit`; any `# pylint: disable=...` must include a reason comment.

## Testing Guidelines
- `pytest` + `pytest-asyncio` (asyncio mode is `auto`).
- Tests are named `test_*.py`; prefer `workflow_compiler/tests/` or module-level test folders.
- Add regression tests for bug fixes and workflow schema/validation changes.

## Adding Tools, Triggers, and Workflow Nodes
- New tool: add a `BaseTool` in `shared/tools/<provider>/`, register via `register_tool(...)`, and ensure the module is imported by the loader. Provide a JSON schema in `get_parameters_schema()`.
- New trigger: implement a poll adapter in `api/triggers/polling/adapters/`, register it in the adapter registry, add a trigger definition in `api/triggers/services.py`, and set a polling interval in `api/triggers/polling/scheduler.py`.
- New workflow node type: add executor logic in `workflow_compiler/runtime/nodes.py` and validate/transform rules in `workflow_compiler/compiler.py` or schema models as needed.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits (e.g., `fix:`, `feat:`, `chore:`) based on recent history.
- PRs should include: a concise summary, linked issue (if any), testing notes, and screenshots for UI/doc changes.

## Security & Configuration Tips
- Keep secrets in `.env` (e.g., `OPENAI_API_KEY`, `GOOGLE_CLIENT_SECRET`) and never commit them.
- Docker compose injects `DATABASE_URL` and `REDIS_URL`; keep local overrides in `.env`.

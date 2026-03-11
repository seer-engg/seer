# Repository Guidelines

## Project Structure & Module Organization
- `src/seer/api/`: FastAPI HTTP layer (routers, middleware, API models).
- `src/seer/services/`: business logic used by API/worker (workflows, integrations, triggers).
- `src/seer/core/`: workflow compiler, runtime, schema models, and global compiler singleton.
- `src/seer/tools/`: tool registry, executor, credential resolution, and provider tool implementations.
- `src/seer/agents/`: agent-specific orchestration (LangGraph-based workflow agent).
- `src/seer/worker/`: Taskiq background worker and polling tasks.
- `src/seer/database/`: Tortoise ORM models/configuration; migrations live in `/migrations`.
- `src/seer/analytics/`, `src/seer/observability/`, `src/seer/utilities/`: shared instrumentation and helpers.
- `documentation/`: docs site assets (Node-based).
- `scripts/`: maintenance and debugging scripts.

## Core Systems (Compiler, Tools, Triggers)
- Workflow compiler: `src/seer/core/` validates workflow specs, builds LangGraph graphs, and hosts runtime node executors; API/services/worker call into it via the global compiler singleton.
- Tool registry: `src/seer/tools/` provides `BaseTool`, registry helpers, executor, and credential resolution.
- Trigger polling: trigger catalog + subscription management lives under `src/seer/api/workflows/services` and `src/seer/services/workflows/triggers.py`; Taskiq worker (`src/seer/worker/`) polls and dispatches runs.



## Coding Style & Naming Conventions
- Python 3.12, 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes.
- Line length limit is 150 characters (see `pyproject.toml`).
- Linting uses `pylint` and `pre-commit`; any `# pylint: disable=...` must include a reason comment.


## Build & Run Commands
- **Run Scripts:** Always use `uv run <script_name>` (e.g., `uv run main.py`)
- **Install Packages:** Always use `uv add <package>` (NEVER use `pip install`)
- **Run Tests:** `uv run pytest`
- **Lockfile:** Rely on `uv.lock`. Do not create requirements.txt unless explicitly asked.

## Environment
- This project uses `uv` for dependency management.
- Do not attempt to create virtual environments manually (venv). Let `uv` handle it.

## Testing Guidelines
- `pytest` + `pytest-asyncio` (asyncio mode is `auto`).
- Tests are named `test_*.py`;
- Add regression tests for bug fixes and workflow schema/validation changes.
- all tests are in /tests
- for every change related to `/src/seer/core` make sure to add concerned unit tests and full json spec tests and validate that the changes passess all the tests ( regression testing )

## Git Workflow
- Branch naming: `<name>/<MMDD>-<slug>` (e.g., `akshay/0311-fix-templates`)
- PRs always target `dev` branch
- Linting must pass before committing (pre-commit hooks enforced)
- After PR merges to `dev` and CI passes, publish to `main`

# Repository Guidelines

## Architectural Philosophy

Seer is an open-source AI workflow automation platform (AGPL-3.0). Monorepo: Python backend + React frontend in `seer-frontend/`.

**Dependency direction is strict and one-way:**
```
api → services → core → tools
         ↓
       worker
```
Never reach "upward." If you need to, extract a shared abstraction into `core/` or `services/`.

## Project Structure
- `src/seer/api/`: FastAPI HTTP layer — thin routers, validates input, calls services, returns output. No business logic.
- `src/seer/services/`: Business logic for API and worker — workflow orchestration, integrations, triggers.
- `src/seer/core/`: Workflow compiler, runtime, schema models, global compiler singleton. The engine. Must not import from api/services/worker.
- `src/seer/tools/`: Tool registry, executor, credential resolution, per-provider implementations. See each module's docstring for how to add new tools.
- `src/seer/agents/`: LangGraph-based workflow agent orchestration.
- `src/seer/worker/`: Taskiq background worker and polling tasks. Nothing imports from worker.
- `src/seer/database/`: Tortoise ORM models/config. Migrations in `/migrations`.
- `src/seer/observability/`: Structured logging and Sentry. Use this — never `print()` or bare `logging.getLogger()`.
- `documentation/`: Docs site (Node-based). `scripts/`: Maintenance scripts.
- `seer-frontend/src/`: React + TypeScript + Vite frontend.
  - `components/ui/`: Base UI components (shadcn/ui)
  - `components/workflows/`: Workflow builder (React Flow)
  - `stores/`: Zustand state management
  - `hooks/`: Custom React hooks (including auth abstraction)
  - `lib/`: Utilities and API client
  - `pages/`: Route pages

## Credentials & Secrets
- **Never hardcode secrets.** Not in code, comments, or test fixtures.
- All credentials flow through `src/seer/tools/` credential resolution. Tools receive resolved credentials at execution time.
- App config and env vars: `src/seer/core/config.py`.
- In tests, mock the credential resolver — never use real tokens.

## Coding Style
- Python 3.12, 4-space indent, `snake_case` functions/modules, `PascalCase` classes, 150-char line limit.
- Type hints on all public function signatures. Use `from __future__ import annotations` for forward refs.
- Any `# pylint: disable=...` must include a reason comment.
- Use domain exceptions from `core/` (e.g., `WorkflowCompilationError`, `ToolExecutionError`) — not bare `Exception`/`ValueError`.
- API layer catches domain exceptions and maps to HTTP status. Services never return HTTP responses.
- In async code, let exceptions propagate — the caller decides recovery.

## Build & Run — Critical Rules
- **Always** use `uv run <script>` and `uv add <package>`. **NEVER** `pip install`. **NEVER** manually create venvs.
- **Migrations:** Only via `uv run aerich migrate --name <n>`. Manual migration files lack `MODELS_STATE` and break CI.
- **Pre-commit:** Run `uv run pytest --tb=short -q` and **wait for it to pass** before committing. No background commits.
- Rely on `uv.lock`. No `requirements.txt` unless explicitly asked.

## Testing
- `pytest` + `pytest-asyncio` (mode: `auto`). All tests in `/tests`, named `test_*.py`.
- **Bug fixes:** Every fix must include a test that fails without the fix and passes with it.
- **Core changes (`src/seer/core/`):** Require unit tests + full JSON spec tests. Run the entire suite, not just new tests.
- Mock external dependencies (HTTP, DB, credentials). Do not mock internal modules in the same layer — test through public interfaces.
- Tests must not depend on execution order or shared mutable state.

## Git Workflow
- Branch: `<name>/<MMDD>-<slug>` (e.g., `akshay/0311-fix-templates`)
- Commits: conventional format — `feat(core): add conditional validation`, `fix(tools): handle OAuth refresh failure`
- PRs target `dev`. Description must include: what changed, why, how to test.
- After merge to `dev` + CI green → publish to `main`.

## Isolated Development (ISO)

**Setup (first time only):** `git clone https://github.com/seer-engg/iso ~/iso && ~/iso/setup.sh`

When starting any feature, bugfix, or dev work in this repo:
1. Use the `iso_init_thread` MCP tool to create an isolated thread
2. Report the thread ID, ports, and worktree path to the user
3. All work happens in the worktree — never modify the main repo working directory
4. On completion, remind user to `iso cleanup <id>`

## Trigger Handlers Must Be Idempotent
The same event delivered twice must not produce duplicate workflow runs. This is a hard requirement for all trigger implementations in `src/seer/services/workflows/triggers.py` and `src/seer/worker/`.

## Frontend

### Tech Stack
- React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui (Radix), React Query, React Router v6, React Flow (@xyflow/react), Zustand
- Authentication: Clerk (cloud) or local mode (self-hosted) — abstracted via `src/hooks/useAuthProvider.ts`

### Frontend Coding Style
- TypeScript strict mode. Prefer named exports over default exports.
- Use `cn()` from `@/lib/utils` for class merging.
- All components must support dark mode (`dark:` Tailwind prefix).
- Follow CVA (Class Variance Authority) pattern for component variants.

### Frontend Build & Run
- `cd seer-frontend && npm install && npm run dev` — or just `docker compose up`
- `npm run lint -- --fix` for linting
- `npm run build` for production build

### Type Placement
- Used in 3+ components across subdirs → root-level `types.ts` or `buildtypes.ts`
- Used in 2+ files within same subdir → subdirectory-level `types.ts`
- Otherwise → keep in component file

### Auth Abstraction
Auth is controlled by a single `AUTH_PROVIDER` env var on the backend. Frontend fetches mode at runtime from `/api/auth/config`.
- Never import `useAuth`/`useUser`/`useClerk` directly — use `useAuthStatus`/`useCurrentUser`/`useSignOut` from `src/hooks/useAuthProvider.ts`
- Local mode: no login, single default user, no Clerk key needed
- Clerk mode: full Clerk auth with login

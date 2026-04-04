# Seer Frontend

## Core Architecture Principle

**If workflows and agents are fundamentally different at the UI layer, they should be different at the API layer.**

This principle guides our component design: workflows (deterministic, node-based execution with DAG visualization) and agents (dynamic, message-based conversations with reasoning) have distinct mental models, data structures, and user needs. Rather than forcing unification through shared components or transformation layers, we maintain separate UI components and routes that align with their fundamental differences. This reduces complexity, improves maintainability, and ensures each system can evolve independently.

## Getting Started

```sh
# Clone the repository
git clone <YOUR_GIT_URL>

# Navigate to the project directory
cd <YOUR_PROJECT_NAME>

# Install dependencies (requires Bun - https://bun.sh/docs/installation)
bun install

# Start the development server
bun run dev
```

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## Backend Configuration

### Connecting to Self-Hosted Backend

**Automatic Setup (Recommended):**
1. Start backend: `cd backend && docker compose up`
2. Visit `http://localhost:8000` in your browser
3. Automatically redirects to frontend with backend configured

**Manual Setup:**
Visit `http://localhost:5173?backend=localhost:8000`

### Backend URL Priority

1. Query Parameter: `?backend=localhost:8000` (highest)
2. LocalStorage: Custom URL saved in Settings
3. Environment: `VITE_BACKEND_API_URL` in `.env`
4. Default: `http://localhost:8000`

### Testing with Cloud Frontend

Set backend environment variable:
```bash
FRONTEND_URL=https://app.getseer.dev
```

Then visit `http://localhost:8000` to redirect to cloud frontend.

The backend must implement these API endpoints:
- `/health` - Health check endpoint (returns server info including status, server name, and version)
- `/info` - Server info endpoint (returns same data as `/health` for connectivity checks)

## LangGraph Agent Chat

The dashboard now embeds the reusable Agent Chat UI from `agent-chat-ui` for both the supervisor and eval agents. The shared `AgentChatContainer` component lives in `src/features/agent-chat/AgentChatContainer.tsx` and simply needs an API URL plus an assistant (graph) ID.

You can override the built-in defaults via environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `VITE_BACKEND_API_URL` | Backend API base URL | `http://localhost:8000` (fallback) |
| `VITE_AGENT_CHAT_API_URL` | Optional global LangGraph deployment URL fallback | `http://localhost:8000` |
| `VITE_AGENT_CHAT_ASSISTANT_ID` | Optional global assistant ID fallback | _none_ |
| `VITE_EVAL_AGENT_ID` | Eval graph ID | `eval_agent` |
| `VITE_SUPERVISOR_AGENT_ID` | Supervisor graph ID | `supervisor` |

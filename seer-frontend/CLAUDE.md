# Seer Frontend - Project Memory

## Project Overview

Seer is a workflow automation and AI agent platform with a React-based frontend for building, testing, and monitoring AI workflows.

## Tech Stack

- **Framework:** React 18 with TypeScript
- **Build Tool:** Vite
- **Styling:** Tailwind CSS with custom design system
- **Component Library:** shadcn/ui (Radix UI primitives)
- **State Management:** React Query (TanStack Query)
- **Routing:** React Router v6
- **UI Patterns:** Class Variance Authority (CVA) for component variants
- **Authentication:** Clerk
- **Backend:** Supabase
- **Workflow Engine:** React Flow (@xyflow/react)

## Project Structure

```
src/
├── components/
│   ├── ui/              # Base UI components (shadcn/ui)
│   ├── workflows/       # Workflow builder components
│   └── ...              # Feature-specific components
├── pages/               # Route pages
├── lib/                 # Utilities and helpers
├── hooks/               # Custom React hooks
└── index.css           # Design system and global styles
```

## Common Commands

```bash
# Development
npm run dev              # Start dev server on http://localhost:5173

# Build
npm run build           # Production build
npm run build:dev       # Development build

# Linting
npm run lint            # Run ESLint
npm run lint -- --fix   # Auto-fix linting issues

# Preview
npm run preview         # Preview production build
```

## Dark Mode Support

All components MUST support both light and dark modes:
- Use CSS variables from `:root` and `.dark` (defined in `src/index.css`)
- Test all UI changes in both themes before finalizing
- Apply dark mode variants using Tailwind's `dark:` prefix

## Workflow Components

The workflow builder uses React Flow for visual workflow creation:

- Block nodes: LLM, If/Else, For Loop, Trigger
- Tool integrations: GitHub, Gmail, webhooks, custom tools
- State management: Zustand stores (canvas, workflow, tools, triggers)
- Reference: `src/components/workflows/`

## Code Style

- Use TypeScript strict mode (already enabled)
- Prefer named exports over default exports
- Use `cn()` utility from `@/lib/utils` for class merging
- Follow existing component patterns in `src/components/ui/`

## Decision Tree: Where Should a Type Live?
Is the type used in 3+ components across different subdirectories?
  ├─ YES → Place in root-level types.ts or buildtypes.ts
  └─ NO → Is it used in 2+ files within same subdirectory?
      ├─ YES → Place in subdirectory-level types.ts
      └─ NO → Keep in component file where it's used

## Git Workflow
- Branch naming: `<name>/<MMDD>-<slug>` (e.g., `akshay/0311-fix-templates`)
- PRs always target `dev` branch
- Linting must pass before committing (pre-commit hooks enforced)
- After PR merges to `dev` and CI passes, publish to `main`

## Isolated Development (ISO)

**Setup (first time only):** `git clone https://github.com/seer-engg/iso ~/iso && ~/iso/setup.sh`

When starting any feature, bugfix, or dev work in this repo:
1. Use the `iso_init_thread` MCP tool to create an isolated thread
2. Report the thread ID, ports, and worktree path to the user
3. All work happens in the worktree — never modify the main repo working directory
4. On completion, remind user to `iso cleanup <id>`

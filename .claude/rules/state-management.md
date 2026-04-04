---
paths: src/**/*.{tsx,ts}
---

# State Management Guidelines

Follow these patterns to maintain clean, efficient state management using Zustand stores.

## Zustand Store vs Local State

### Use Zustand Stores When

- State is **shared across multiple components**
- State **persists across route changes** or component unmounts
- State is part of **core domain logic** (workflows, integrations, canvas, triggers)
- State has **complex update logic** that should be centralized
- State needs to be **accessed by deeply nested components** (avoid prop drilling)

### Use Local useState When

- State is **purely UI** (modal open/closed, dropdown expanded, form input during typing)
- State **never leaves the component** boundary
- State has **no business logic** or side effects
- State is **temporary** and resets on every component mount

## Existing Zustand Stores

Always check existing stores before creating new hooks or useState:

| Store | Path | Purpose |
|-------|------|---------|
| `canvasStore` | `src/stores/canvasStore.ts` | Workflow canvas nodes, edges, selection |
| `workflowStore` | `src/stores/workflowStore.ts` | Workflow CRUD, versions, execution |
| `toolsStore` | `src/stores/toolsStore.ts` | Tool integrations and credentials |
| `triggersStore` | `src/stores/triggersStore.ts` | Trigger configurations |
| `chatStore` | `src/stores/chatStore.ts` | Chat history and AI responses |
| `uiStore` | `src/stores/uiStore.ts` | UI state (modals, panels, etc.) |

## Anti-Pattern: Unnecessary Custom Hooks

❌ **Before - Unnecessary Hook:**

```tsx
// ❌ Don't: Creating a custom hook for global state
const useToolState = () => {
  const [tools, setTools] = useState<Tool[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Fetch tools and update state
    fetchTools().then(setTools);
  }, []);

  return { tools, loading, setTools };
};

// Used in multiple components
function ToolList() {
  const { tools } = useToolState(); // ❌ Separate instance per component!
  return <div>{tools.map(...)}</div>;
}
```

**Problem:** Each component gets its own instance of state. No sharing happens.

✅ **After - Zustand Store:**

```tsx
// ✅ Do: Use Zustand for shared state
// src/stores/toolStore.ts
import { create } from 'zustand';

interface ToolStore {
  tools: Tool[];
  loading: boolean;
  fetchTools: () => Promise<void>;
  setTools: (tools: Tool[]) => void;
}

export const useToolStore = create<ToolStore>((set) => ({
  tools: [],
  loading: false,
  fetchTools: async () => {
    set({ loading: true });
    const tools = await fetchTools();
    set({ tools, loading: false });
  },
  setTools: (tools) => set({ tools }),
}));

// Used in multiple components - same state!
function ToolList() {
  const { tools } = useToolStore(); // ✅ Shared state
  return <div>{tools.map(...)}</div>;
}

function ToolCount() {
  const tools = useToolStore((state) => state.tools); // ✅ Same state, optimized selector
  return <span>{tools.length}</span>;
}
```

## Pattern: Move Component State to Store

### Identifying Candidates for Migration

Look for these patterns that indicate state should be in a store:

1. **Prop drilling** - passing state through 3+ component levels
2. **Duplicate API calls** - multiple components fetching the same data
3. **Global context** - using React Context for non-theme data
4. **Custom hooks with useState** - hooks that manage shared state

### Migration Steps

1. **Identify the store** - Check if existing store should handle this (prefer extending existing stores)
2. **Move state** - Add state and actions to store
3. **Update components** - Replace useState/context with store hooks
4. **Delete unused code** - Remove custom hooks, context providers, prop drilling

### Example: Refactoring Integration State

❌ **Before - Props and Local State:**

```tsx
// Parent fetches and passes down
function IntegrationPage() {
  const [integrations, setIntegrations] = useState<Integration[]>([]);

  useEffect(() => {
    fetchIntegrations().then(setIntegrations);
  }, []);

  return <IntegrationList integrations={integrations} />;
}

function IntegrationList({ integrations }: { integrations: Integration[] }) {
  return <div>{integrations.map(...)}</div>;
}
```

✅ **After - Zustand Store (Actual Pattern from Codebase):**

```tsx
// Already exists: src/stores/toolsStore.ts
export const useToolsStore = create<IntegrationStore>((set) => ({
  integrations: [],
  fetchIntegrations: async () => {
    const data = await fetchIntegrations();
    set({ integrations: data });
  },
}));

// Parent doesn't manage state
function IntegrationPage() {
  const fetchIntegrations = useToolsStore((state) => state.fetchIntegrations);

  useEffect(() => {
    fetchIntegrations();
  }, [fetchIntegrations]);

  return <IntegrationList />;
}

// Child accesses store directly
function IntegrationList() {
  const integrations = useToolsStore((state) => state.integrations);
  return <div>{integrations.map(...)}</div>;
}
```

## Zustand Best Practices

### 1. Selective Subscriptions

Use selectors to prevent unnecessary re-renders:

```tsx
// ❌ Bad: Re-renders on ANY store change
const { tools, loading, error } = useToolStore();

// ✅ Good: Only re-renders when tools change
const tools = useToolStore((state) => state.tools);
```

### 2. Store Organization

Keep related state together in the same store:

```tsx
// ✅ Good: Related workflow state in one store
interface WorkflowStore {
  workflows: Workflow[];
  currentWorkflow: Workflow | null;
  isExecuting: boolean;
  fetchWorkflows: () => Promise<void>;
  executeWorkflow: (id: string) => Promise<void>;
}

// ❌ Bad: Splitting tightly coupled state
interface WorkflowListStore {
  workflows: Workflow[];
}
interface WorkflowExecutionStore {
  isExecuting: boolean;
}
```

### 3. Action Naming

Use clear, action-oriented names:

```tsx
// ✅ Good names
setNodes(nodes: Node[])
addNode(node: Node)
updateNode(id: string, updates: Partial<Node>)
removeNode(id: string)

// ❌ Bad names
nodes(nodes: Node[])          // Not clear it's a setter
change(id: string, data: any) // Too vague
```

### 4. Async Actions

Handle loading and error states in async actions:

```tsx
interface Store {
  data: Data[];
  loading: boolean;
  error: string | null;
  fetchData: () => Promise<void>;
}

const useStore = create<Store>((set) => ({
  data: [],
  loading: false,
  error: null,
  fetchData: async () => {
    set({ loading: true, error: null });
    try {
      const data = await api.fetchData();
      set({ data, loading: false });
    } catch (error) {
      set({ error: error.message, loading: false });
    }
  },
}));
```

## When NOT to Refactor

Keep local state for these cases:

```tsx
// ✅ Good: Temporary UI state
function SearchBar() {
  const [query, setQuery] = useState(''); // Local - just for input
  const [isOpen, setIsOpen] = useState(false); // Local - dropdown state

  const handleSearch = () => {
    // When search is submitted, use store
    useSearchStore.getState().search(query);
  };

  return <input value={query} onChange={(e) => setQuery(e.target.value)} />;
}

// ✅ Good: Component-specific derived state
function WorkflowCard({ workflow }: { workflow: Workflow }) {
  const [isHovered, setIsHovered] = useState(false); // Local - just for this card
  return <div onMouseEnter={() => setIsHovered(true)}>...</div>;
}
```

## Recent Refactoring Pattern

Recent commits show this pattern being applied:

```
706e376 chore: 2nd pass refactor use global stores
994e872 chore: refactor use global stores
d354356 decouple integration and trigger stores
```

This means:
- **Tool/integration data** moved to `toolsStore`
- **Trigger data** moved to `triggersStore`
- **Canvas state** moved to `canvasStore`
- Unnecessary custom hooks and local state were deleted

When you see opportunities to continue this pattern, suggest moving state to appropriate stores.

## Checklist

Before adding new state:

- [ ] Does this state need to be shared? → Use Zustand
- [ ] Is there an existing store for this domain? → Extend it
- [ ] Is this temporary UI state? → Use local useState
- [ ] Am I duplicating API calls across components? → Move to store
- [ ] Am I prop drilling more than 2 levels? → Move to store

## Summary

**Default to Zustand stores for domain state, local useState for UI state.** When in doubt, ask yourself: "Will another component need this?" If yes, use a store.

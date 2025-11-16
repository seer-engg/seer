# 🔄 Phase 2 Migration Guide

Quick reference for developers working with the new Phase 2 architecture.

---

## 🎯 Quick Reference: What Changed

| Old Pattern | New Pattern | Status |
|------------|-------------|--------|
| `from shared.tool_catalog import ...` | `from shared.tools import ...` | ✅ Required |
| `mcp_client, _ = await get_mcp_client_and_configs(...)` | `tool_service = get_tool_service(); await tool_service.initialize(...)` | ✅ Recommended |
| `EVAL_PASS_THRESHOLD = 0.8` in multiple files | `from shared.config import config; config.eval_pass_threshold` | ✅ Recommended |
| State fields duplicated across agents | `context: AgentContext` in state | ✅ Required for new code |

---

## 1. Tool Imports

### ❌ Old:
```python
from shared.tool_catalog import (
    load_tool_entries,
    select_relevant_tools,
    canonicalize_tool_name,
    ToolEntry,
)
```

### ✅ New:
```python
from shared.tools import (
    load_tool_entries,
    select_relevant_tools,
    canonicalize_tool_name,
    ToolEntry,
)
```

**Action**: Update all imports. The API is identical, just the import path changed.

---

## 2. Tool Service Usage

### ❌ Old Pattern:
```python
# Scattered across multiple files
mcp_client, _ = await get_mcp_client_and_configs(state.mcp_services)
mcp_tools = await mcp_client.get_tools()
tools_dict = {canonicalize_tool_name(t.name): t for t in mcp_tools}
```

### ✅ New Pattern:
```python
from shared.tool_service import get_tool_service

tool_service = get_tool_service()
await tool_service.initialize(state.context.mcp_services)
tools_dict = tool_service.get_tools()  # Already canonicalized!
```

**Benefits**:
- Automatic caching
- Consistent pattern everywhere
- Easier to test

---

## 3. Configuration

### ❌ Old:
```python
import os

EVAL_PASS_THRESHOLD = float(os.getenv("EVAL_PASS_THRESHOLD", "0.8"))
N_ROUNDS = int(os.getenv("N_ROUNDS", "2"))
```

### ✅ New:
```python
from shared.config import config

# Access via config object
if score >= config.eval_pass_threshold:
    ...

# Or use backward-compatible exports
from shared.config import EVAL_PASS_THRESHOLD, N_ROUNDS
```

**Benefits**:
- Type safety (Pydantic validation)
- Single source of truth
- Easy to mock in tests

---

## 4. Agent State

### ❌ Old Pattern (Duplication):
```python
class EvalAgentState(BaseModel):
    user_context: Optional[UserContext]
    github_context: Optional[GithubContext]
    sandbox_context: Optional[SandboxContext]
    target_agent_version: int
    mcp_services: List[str]
    mcp_resources: Dict[str, Any]
    # ... plus eval-specific fields

class CodexInput(BaseModel):
    user_context: UserContext  # DUPLICATE
    github_context: GithubContext  # DUPLICATE
    sandbox_context: Optional[SandboxContext]  # DUPLICATE
    target_agent_version: int  # DUPLICATE
    mcp_services: List[str]  # DUPLICATE
    # ... plus codex-specific fields
```

### ✅ New Pattern (DRY):
```python
from shared.agent_context import AgentContext

class EvalAgentState(BaseModel):
    context: AgentContext  # All shared fields
    # ... only eval-specific fields
    
    # Backward compatibility
    @property
    def user_context(self):
        return self.context.user_context

class CodexInput(BaseModel):
    context: AgentContext  # All shared fields
    # ... only codex-specific fields
```

**Benefits**:
- No duplication
- Easier handoffs
- Single source of truth

---

## 5. Creating Agent State

### ✅ New Code:
```python
from shared.agent_context import AgentContext

# Create context
context = AgentContext(
    user_context=user_ctx,
    github_context=github_ctx,
    sandbox_context=sandbox_ctx,
    target_agent_version=1,
    mcp_services=["asana", "github"],
    mcp_resources={},
)

# Create state with context
state = EvalAgentState(
    context=context,
    messages=[],
    attempts=0,
    # ... eval-specific fields
)
```

### ✅ Backward Compatible (Still Works):
```python
# Old code still works due to properties!
print(state.user_context)  # → state.context.user_context
print(state.mcp_services)  # → state.context.mcp_services
```

---

## 6. Agent Handoffs

### ❌ Old Pattern:
```python
# Manual field copying (easy to miss fields!)
codex_input = CodexInput(
    user_context=state.user_context,
    github_context=state.github_context,
    sandbox_context=state.sandbox_context,
    target_agent_version=state.target_agent_version,
    mcp_services=state.mcp_services,
    dataset_context=state.dataset_context,
    experiment_context=state.active_experiment,
    dataset_examples=state.dataset_examples,
)
```

### ✅ New Pattern:
```python
# Just pass the context!
codex_input = CodexInput(
    context=state.context,  # All shared fields in one go
    dataset_context=state.dataset_context,
    experiment_context=state.active_experiment,
    dataset_examples=state.dataset_examples,
)
```

**Benefits**:
- Can't forget fields
- Cleaner code
- Easier to maintain

---

## 7. Accessing State Fields

### All of these work (backward compatible):

```python
# Direct access (new preferred way)
state.context.sandbox_context
state.context.mcp_services
state.context.target_agent_version

# Property access (backward compatible)
state.sandbox_context  # → state.context.sandbox_context
state.mcp_services     # → state.context.mcp_services
state.target_agent_version  # → state.context.target_agent_version
```

**Recommendation**: Use `state.context.*` in new code, but old code continues to work.

---

## 8. Tool Selection

### ❌ Old:
```python
from shared.tool_catalog import load_tool_entries, select_relevant_tools

entries = await load_tool_entries(services)
selected = await select_relevant_tools(
    entries,
    context="create a task",
    max_total=20
)
```

### ✅ New (Option 1 - Direct):
```python
from shared.tools import load_tool_entries, select_relevant_tools

entries = await load_tool_entries(services)
selected = await select_relevant_tools(
    entries,
    context="create a task",
    max_total=20
)
```

### ✅ New (Option 2 - Via Service):
```python
from shared.tool_service import get_tool_service

tool_service = get_tool_service()
await tool_service.initialize(services)
selected_tools = await tool_service.select_relevant_tools(
    context="create a task",
    max_total=20
)
```

**Recommendation**: Use ToolService for consistency.

---

## 🧪 Testing Changes

### Testing with new config:
```python
from shared.config import config

def test_something():
    # Override config for test
    original = config.eval_pass_threshold
    config.eval_pass_threshold = 0.5
    
    try:
        # Your test
        ...
    finally:
        # Restore
        config.eval_pass_threshold = original
```

### Testing with ToolService:
```python
from shared.tool_service import get_tool_service, reset_tool_service

def test_something():
    # Get fresh service
    reset_tool_service()
    tool_service = get_tool_service()
    
    # Your test
    ...
```

---

## ⚠️ Common Migration Pitfalls

### 1. Forgetting to update imports
```python
# ❌ Will fail
from shared.tool_catalog import canonicalize_tool_name

# ✅ Correct
from shared.tools import canonicalize_tool_name
```

### 2. Not initializing ToolService
```python
# ❌ Will raise RuntimeError
tool_service = get_tool_service()
tools = tool_service.get_tools()  # ERROR: not initialized!

# ✅ Correct
tool_service = get_tool_service()
await tool_service.initialize(["asana"])
tools = tool_service.get_tools()
```

### 3. Creating context incorrectly
```python
# ❌ Missing required field
context = AgentContext()  # Will use defaults

# ✅ Be explicit about required fields
context = AgentContext(
    user_context=None,  # Can be None
    github_context=None,
    sandbox_context=None,
    target_agent_version=0,
    mcp_services=[],
    mcp_resources={},
)
```

---

## 📚 File Structure Reference

```
shared/
├── agent_context.py      # NEW: Unified context
├── config.py             # UPDATED: Pydantic-based config
├── tool_service.py       # NEW: Tool management service
├── schema.py             # UPDATED: Uses AgentContext
└── tools/                # NEW: Focused tool modules
    ├── __init__.py
    ├── loader.py
    ├── normalizer.py
    ├── registry.py
    ├── selector.py
    └── vector_store.py

agents/
├── eval_agent/
│   └── models.py         # UPDATED: Uses AgentContext
└── codex/
    └── state.py          # UPDATED: Uses CodexInput/Output
```

---

## 🎯 Migration Checklist

For each file you modify:

- [ ] Update imports from `tool_catalog` to `tools`
- [ ] Consider using `ToolService` instead of direct MCP client
- [ ] Use `config` object instead of `os.getenv()`
- [ ] If creating new state, use `AgentContext`
- [ ] For handoffs, pass `context` directly
- [ ] Test that old property access still works

---

## 💡 Pro Tips

1. **Use IDE refactoring**: Most IDEs can update imports automatically
2. **Test incrementally**: Don't change everything at once
3. **Properties are your friend**: Backward compatibility means no rush
4. **Embrace the new patterns**: Cleaner code is worth the migration

---

## 🆘 Need Help?

If you encounter issues:

1. Check this guide first
2. Look at `PHASE2_SUMMARY.md` for the big picture
3. Review example usages in `agents/eval_agent/graph.py`
4. The backward compatibility properties should handle most cases

---

**Happy Coding! 🚀**


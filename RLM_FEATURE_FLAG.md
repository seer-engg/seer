# RLM Tool Feature Flag Guide

## 🎯 Quick Reference

The Recursive Language Model (RLM) tool is **disabled by default**. Enable it with:

```bash
# In .env file
ENABLE_RLM_TOOL=true
```

## 🚀 Deployment Options

### Option 1: Local Development
```bash
# .env
ENABLE_RLM_TOOL=true

# Restart server
uvicorn api.main:app --reload
```

### Option 2: Docker Compose
```yaml
# docker-compose.yml
services:
  api:
    environment:
      - ENABLE_RLM_TOOL=true
```

### Option 3: Environment-Specific
```bash
# staging.env
ENABLE_RLM_TOOL=true

# production.env
ENABLE_RLM_TOOL=false  # Keep disabled until tested

# Run with specific env file
docker-compose --env-file staging.env up
```

### Option 4: Runtime Environment Variable
```bash
# Single run
ENABLE_RLM_TOOL=true uvicorn api.main:app

# Export for session
export ENABLE_RLM_TOOL=true
uvicorn api.main:app
```

## ✅ Verification

### Check if RLM is enabled:

```python
# Python check
from shared.config import config
from shared.tools.base import get_tool

print(f"RLM enabled: {config.enable_rlm_tool}")
print(f"RLM available: {get_tool('recursive_language_model') is not None}")
```

```bash
# CLI check
uv run python -c "from shared.config import config; print(f'RLM: {config.enable_rlm_tool}')"
```

### Check workflow agent tool count:

```bash
# Without RLM: 45 tools
# With RLM: 46 tools

uv run python -c "from shared.tools.base import list_tools; print(f'Tools: {len(list_tools())}')"
```

## 📊 Comparison: Old vs New Agent

### Before RLM (Standard Agent)
```
User: "Analyze 10,000 customer support tickets and find recurring issues"

Agent Response:
❌ "This dataset is too large for me to process effectively. Can you:
   - Provide a smaller sample (first 100 tickets)?
   - Pre-filter the tickets by date range?
   - Summarize the tickets first?"

Result: Shallow analysis, requires user to chunk data manually
```

### With RLM Enabled
```
User: "Analyze 10,000 customer support tickets and find recurring issues"

Agent Response:
✅ "I'll analyze all 10,000 tickets using recursive decomposition..."

[Agent uses recursive_language_model tool:]
- Chunks tickets into 200 batches of 50
- Processes each batch to identify patterns
- Recursively synthesizes findings across batches
- Returns comprehensive analysis

Result: Deep analysis of all data, automatic chunking
```

## 🎮 Example Workflow

With `ENABLE_RLM_TOOL=true`, the workflow agent can now:

```json
{
  "nodes": [
    {
      "id": "analyze_tickets",
      "type": "tool",
      "tool": "recursive_language_model",
      "in": {
        "code": "examine(context['tickets']); chunks = chunk(context['tickets'], 100); result = f'{len(chunks)} batches'",
        "context": {"tickets": "/* 10,000 tickets */"}
      },
      "out": "analysis"
    }
  ]
}
```

## 🧪 Testing Feature Flag

### Test 1: Verify flag is off by default
```bash
uv run python -c "
from shared.tools.base import get_tool
assert get_tool('recursive_language_model') is None
print('✅ RLM correctly disabled by default')
"
```

### Test 2: Verify flag enables tool
```bash
ENABLE_RLM_TOOL=true uv run python -c "
from shared.tools.base import get_tool
assert get_tool('recursive_language_model') is not None
print('✅ RLM correctly enabled with flag')
"
```

### Test 3: Run full test suite
```bash
# With RLM enabled
ENABLE_RLM_TOOL=true uv run pytest workflow_compiler/tests/test_rlm_tool.py -v
```

## 🔄 Rollout Strategy

### Phase 1: Internal Testing (Week 1)
```bash
# staging.env only
ENABLE_RLM_TOOL=true
```
- Test with internal team
- Verify no regressions
- Collect performance metrics

### Phase 2: Beta Users (Week 2-3)
```bash
# production.env - still disabled
ENABLE_RLM_TOOL=false

# Per-user override (future enhancement)
# Enable for specific beta users via database flag
```

### Phase 3: Gradual Rollout (Week 4+)
```bash
# production.env - enable for everyone
ENABLE_RLM_TOOL=true
```
- Monitor error rates
- Track usage analytics
- Collect user feedback

## 🐛 Troubleshooting

### Tool not available after enabling flag

**Check 1: Verify env var is set**
```bash
uv run python -c "import os; print(os.getenv('ENABLE_RLM_TOOL'))"
# Should print: true
```

**Check 2: Verify config loaded**
```bash
uv run python -c "from shared.config import config; print(config.enable_rlm_tool)"
# Should print: True
```

**Check 3: Restart server**
```bash
# Config is loaded at startup
# Must restart after changing .env
pkill -f uvicorn
uvicorn api.main:app --reload
```

### Tool available but not working

**Check 1: RestrictedPython installed**
```bash
uv pip list | grep RestrictedPython
# Should show: restrictedpython==8.1
```

**Check 2: Run tool tests**
```bash
ENABLE_RLM_TOOL=true uv run pytest workflow_compiler/tests/test_rlm_tool.py -v
# Should show: 16 passed
```

## 📈 Monitoring

Track RLM usage in logs:

```python
# Look for these log messages:
# "Executing RLM tool: model=..., max_depth=..., context_size=..."
# "RLM execution completed: X LLM calls, max depth Y, Zms total"
```

Monitor key metrics:
- **Execution count**: How many times RLM is used
- **Average depth**: How deep recursion goes
- **Average time**: How long executions take
- **Error rate**: % of failed executions
- **Context size**: Average input size

## 🎓 Education & Support

### For Users
- Document when to use RLM vs standard prompts
- Provide example workflows
- Show performance characteristics

### For Developers
- Review sandbox restrictions (`shared/tools/reasoning/README.md`)
- Understand helper functions
- Know resource limits

## 📝 Summary

| Environment | Flag Value | Result |
|-------------|-----------|--------|
| **Default** | `false` | RLM disabled, 45 tools available |
| **Local dev** | `true` | RLM enabled, 46 tools available |
| **Staging** | `true` | RLM enabled for testing |
| **Production** | `false` → `true` | Gradual rollout |

**Next Steps:**
1. Set `ENABLE_RLM_TOOL=true` in your `.env`
2. Restart the server
3. Verify with: `get_tool('recursive_language_model') is not None`
4. Test with sample workflow
5. Monitor and iterate!

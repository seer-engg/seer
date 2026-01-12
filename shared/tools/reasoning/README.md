# Recursive Language Model (RLM) Tool

Enables multi-step complex reasoning for Seer's workflow agent by providing a sandboxed Python environment for decomposing and recursively processing large contexts.

Based on: [Recursive Language Models (arxiv.org/html/2512.24601v1)](https://arxiv.org/html/2512.24601v1)

## Feature Flag Control

The RLM tool is **feature-flagged** and disabled by default. Enable it using environment variables:

### Quick Start

```bash
# In your .env file
ENABLE_RLM_TOOL=true

# Or as environment variable
export ENABLE_RLM_TOOL=true
```

### Verification

Check if RLM is enabled:

```python
from shared.config import config
from shared.tools.base import get_tool

print(f"RLM enabled: {config.enable_rlm_tool}")

tool = get_tool("recursive_language_model")
print(f"RLM available: {tool is not None}")
```

## Usage in Workflow Agent

Once enabled, the workflow agent automatically has access to the RLM tool:

```json
{
  "type": "tool",
  "tool": "recursive_language_model",
  "in": {
    "code": "...",
    "context": {...},
    "model": "gpt-4o-mini",
    "max_depth": 3
  },
  "out": "analysis_result"
}
```

## Example: Analyzing Large Datasets

```python
{
  "code": """
# Examine data structure
info = examine(context['tickets'])

# Chunk into batches
chunks = chunk(context['tickets'], size=50)

# Process each chunk
results = []
for chunk in chunks:
    # Count issues by type
    login_issues = search(chunk['text'], pattern='login')
    payment_issues = search(chunk['text'], pattern='payment')

    results.append({
        'batch': chunk['index'],
        'login_count': len(login_issues),
        'payment_count': len(payment_issues)
    })

# Aggregate results
result = {
    'total_batches': len(chunks),
    'total_tickets': info['size'],
    'issue_breakdown': results
}
""",
  "context": {
    "tickets": [/* large array of ticket objects */]
  }
}
```

## Helper Functions

Available in the sandboxed environment:

### `examine(data, max_depth=3, max_items=5)`
Inspect data structure without loading all content.

```python
info = examine(context['large_dataset'])
# Returns: {'type': 'list', 'size': 10000, 'sample': [...]}
```

### `search(data, pattern=None, key=None, value=None)`
Filter/search data by criteria.

```python
# Search by regex pattern
errors = search(logs, pattern='error|failure')

# Search by key substring
user_fields = search(data, key='user_')

# Search by exact value
high_priority = search(items, value='high')
```

### `chunk(data, size=1000, overlap=0)`
Split data into manageable pieces.

```python
# Chunk list into batches of 100
batches = chunk(tickets, size=100)

# Chunk string with 50-char overlap
text_chunks = chunk(document, size=1000, overlap=50)

# Chunk dict by keys
dict_chunks = chunk(large_object, size=50)
```

### `sub_llm(prompt, context_chunk=None, system=None)`
Make recursive LLM call (async, tracks depth).

```python
# Simple call
answer = sub_llm("Summarize this data", context_chunk=chunk)

# With system prompt
analysis = sub_llm(
    prompt="Extract key findings",
    context_chunk=chunk['text'],
    system="You are a data analyst"
)
```

## Safety & Limits

### Sandboxing (RestrictedPython)
- ✅ **Allowed**: Data manipulation, iteration, math, string ops
- ❌ **Blocked**: `import`, `open()`, `exec()`, `eval()`, network, file I/O

### Resource Limits
- Max context size: **10MB**
- Max chunk size: **1MB**
- Max recursion depth: **10** (default: 3)
- Timeout: **10-300 seconds** (default: 60s)

### Error Handling
If code execution fails, partial results are still returned:

```python
{
  "result": {...},         # Partial results
  "error": "...",          # Error message
  "execution_log": [...],  # What executed successfully
  "stats": {...}           # Execution stats
}
```

## Deployment Strategies

### 1. Global Enable (Production)
```bash
# .env
ENABLE_RLM_TOOL=true
```

### 2. Testing/Staging Only
```bash
# staging.env
ENABLE_RLM_TOOL=true

# production.env
ENABLE_RLM_TOOL=false
```

### 3. Per-Environment Docker Compose
```yaml
services:
  api:
    environment:
      - ENABLE_RLM_TOOL=${ENABLE_RLM_TOOL:-false}
```

### 4. Runtime Toggle (Advanced)
For user-level or session-level flags, extend `agents/workflow_agent/utils.py`:

```python
def get_workflow_tools(workflow_state=None, enable_rlm=False):
    base_tools = [
        analyze_workflow,
        submit_workflow_spec,
        search_tools,
    ]

    if enable_rlm or config.enable_rlm_tool:
        rlm_tool = get_tool("recursive_language_model")
        if rlm_tool:
            base_tools.append(
                base_tool_to_langchain_tool(rlm_tool, user)
            )

    return base_tools
```

## Monitoring & Observability

Track RLM usage via execution logs:

```python
result = await tool.execute(None, {...})

print(f"LLM calls made: {result['stats']['total_llm_calls']}")
print(f"Max depth reached: {result['stats']['max_depth_reached']}")
print(f"Execution time: {result['stats']['execution_time_ms']}ms")

for log in result['execution_log']:
    print(f"  Depth {log['depth']}: {log['operation']} ({log['duration_ms']}ms)")
```

## Testing

Run RLM tool tests:

```bash
pytest workflow_compiler/tests/test_rlm_tool.py -v
```

## Rollout Checklist

- [ ] Set `ENABLE_RLM_TOOL=true` in `.env`
- [ ] Restart API server
- [ ] Verify tool is registered: `get_tool("recursive_language_model")` returns non-None
- [ ] Test with sample workflow
- [ ] Monitor execution logs for errors
- [ ] Gradually enable for more users
- [ ] Collect feedback and iterate

## Troubleshooting

**Tool not available:**
```python
from shared.config import config
print(config.enable_rlm_tool)  # Should be True
```

**Import errors:**
```bash
uv pip install RestrictedPython
```

**Execution errors:**
Check `shared/tools/reasoning/sandbox.py` for allowed operations.

## Future Enhancements

- [ ] Parallel chunk processing
- [ ] Caching of repeated sub_llm calls
- [ ] Streaming execution progress
- [ ] Token usage tracking
- [ ] Visual execution trace UI

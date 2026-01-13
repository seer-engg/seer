# Workflow Schema V2

This directory contains the V2 workflow schema definition, examples, and validator.

## Files

### `v2_schema.json`
Formal JSON Schema (Draft-07) defining the complete V2 workflow structure with:
- Top-level: `name`, `description`, `tags`, `spec`, `metadata`
- Spec structure: `triggers`, `nodes`, `edges`, `output`, `ui`
- Node types: `task`, `tool`, `llm`, `if`, `for_each`
- Validation rules for all fields

### `validate_v2.py`
Python validator that checks:
- ✅ JSON Schema compliance (structure validation)
- ✅ DAG properties (no cycles, all nodes connected, no orphans)
- ✅ Node references (edges point to existing nodes)
- ✅ Template expressions (balanced `${}` braces)

Usage:
```python
from workflow_compiler.schema.validate_v2 import validate_workflow_v2
import json

with open('workflow.json') as f:
    workflow = json.load(f)
with open('workflow_compiler/schema/v2_schema.json') as f:
    schema = json.load(f)

is_valid, errors = validate_workflow_v2(workflow, schema)
if not is_valid:
    for error in errors:
        print(error)
```

## Examples (`examples/v2/`)

Seven diverse example workflows demonstrating V2 schema features:

1. **`01_simple_email_alert.json`** - Sequential workflow
   - Form submission → Format email → Send via Gmail
   - Demonstrates: Basic flow, template expressions, `${trigger.config.*}` refs

2. **`02_parallel_data_fetch.json`** - Parallel execution
   - Fetch user & products simultaneously → Generate AI recommendation
   - Demonstrates: Parallel edges (2 nodes from `_start`), LLM node with JSON output

3. **`03_conditional_branching.json`** - Conditional logic
   - Check email attachments → Upload to Drive OR log message
   - Demonstrates: `if` node with `then`/`else` branches

4. **`04_loop_processing.json`** - Iteration
   - Fetch new signups → Send welcome email to each → Update database
   - Demonstrates: `for_each` node, loop variable access (`${signup.email}`)

5. **`05_complex_multi_step.json`** - Combined patterns
   - Parallel fetch (sales + inventory) → LLM analysis → Conditional alert → Sheets → Email batch
   - Demonstrates: All patterns combined, complex DAG

6. **`06_database_crud.json`** - Database operations
   - Webhook trigger → Extract changes → Audit log → Conditional email verification
   - Demonstrates: Supabase CRUD, webhook triggers, audit patterns

7. **`07_file_processing_pipeline.json`** - File processing
   - Drive file upload → Download → LLM extraction → Save to DB → Notify
   - Demonstrates: Sequential pipeline, file operations, AI processing

## Key Schema Features

### Explicit Execution Edges
Workflows define explicit edges for DAG execution:
```json
"edges": [
  {"from": "_start", "to": "node1"},
  {"from": "_start", "to": "node2"},  // Parallel
  {"from": "node1", "to": "node3"},
  {"from": "node2", "to": "node3"},   // Join
  {"from": "node3", "to": "_end"}
]
```

### Template References
Use `${...}` to reference:
- Trigger data: `${trigger.data.email}`
- **Trigger config**: `${trigger.config.integration_resource_id}`
- Node outputs: `${user_data}`
- Loop variables: `${signup.name}`

**Note**: Inputs and outputs are defined at the **node level only** through `in` and `out` fields. There is no workflow-level `inputs` field.

### Node Types

**Task Node** - Set values
```json
{
  "id": "format_text",
  "type": "task",
  "kind": "set",
  "value": "Hello ${user.name}",
  "out": "greeting"
}
```

**Tool Node** - Execute tools
```json
{
  "id": "send_email",
  "type": "tool",
  "tool": "gmail_send_email",
  "in": {
    "integration_resource_id": "${trigger.config.gmail_id}",
    "to": ["${user.email}"],
    "body_text": "${greeting}"
  },
  "out": "result"
}
```

**LLM Node** - AI processing
```json
{
  "id": "analyze",
  "type": "llm",
  "model": "gpt-4o-mini",
  "prompt": "Analyze: ${data}",
  "out": "insights",
  "output": {
    "mode": "json",
    "schema": {
      "schema": {
        "type": "object",
        "properties": {
          "summary": {"type": "string"}
        }
      }
    }
  }
}
```

**If Node** - Conditional branching
```json
{
  "id": "check_status",
  "type": "if",
  "condition": "${status == 'active'}",
  "then": [/* nodes */],
  "else": [/* nodes */],
  "out": "result"
}
```

**For Each Node** - Iteration
```json
{
  "id": "process_items",
  "type": "for_each",
  "items": "${users}",
  "item_var": "user",
  "body": [/* nodes */],
  "out": "results"
}
```

## Testing

Run all validator tests:
```bash
uv run pytest workflow_compiler/tests/test_validate_v2.py -v
```

Validate all examples:
```bash
for file in workflow_compiler/schema/examples/v2/*.json; do
  uv run python -c "
from workflow_compiler.schema.validate_v2 import validate_workflow_v2
import json

with open('$file') as f:
    workflow = json.load(f)
with open('workflow_compiler/schema/v2_schema.json') as f:
    schema = json.load(f)

is_valid, errors = validate_workflow_v2(workflow, schema)
print('✓ Valid' if is_valid else f'✗ Invalid: {errors}')
"
done
```

## Next Steps

After schema stabilization:
1. Update Pydantic models (`models.py`) to match V2 schema
2. Migrate existing workflows in database
3. Update compiler to read `spec.edges` for execution
4. Update workflow agent to generate V2 schemas

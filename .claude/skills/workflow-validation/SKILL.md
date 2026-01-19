---
name: workflow-validation
description: Validates workflow schemas, expressions, and block configurations in seer. Use when reviewing workflow changes, adding new blocks, debugging workflow compilation errors, or validating workflow JSON structures.
allowed-tools: Read, Grep, Glob, Bash(pytest:*)
---

# Workflow Validation Skill

Validates workflow specifications through a 5-stage compilation pipeline. Use when working on workflow-related code. See `workflow_compiler/README.md` for architecture overview.

## Compilation Pipeline

Validation happens across 5 stages:

1. **Parse** (`compiler/parse.py`): JSON → Pydantic `WorkflowSpec` (structural validation)
2. **Type Environment** (`compiler/type_env.py`): Build type environment from schemas → Raises `TypeEnvironmentError`
3. **Validate References** (`compiler/validate_refs.py`): Check `${...}` references exist → Raises `ValidationPhaseError`
4. **Lower** (`compiler/lower_control_flow.py`): Transform to execution plan → Raises `LoweringError`
5. **Emit** (`compiler/emit_langgraph.py`): Generate LangGraph StateGraph

**Runtime**: Expression evaluation and output validation

## Error Hierarchy

```
WorkflowCompilerError (base)
├── ValidationPhaseError (Stage 1-3: structural/reference validation)
├── TypeEnvironmentError (Stage 2: type env construction)
├── LoweringError (Stage 4: lowering failures)
└── ExecutionError (runtime: tool execution, schema validation)
    └── EvaluationError (runtime: expression evaluation)
```

## Key Validation Components

### 1. **Input Validation** (`workflow_compiler/runtime/input_validation.py`)

Validates and coerces runtime inputs against the workflow spec.

**Key patterns:**
- Uses `InputDef` from `workflow_compiler/schema/models.py` for type definitions
- Supports types: `string`, `integer`, `number`, `boolean`, `object`, `array`
- Applies default values declared on each InputDef
- Raises `WorkflowCompilerError` if required input is missing or cannot be coerced

**Type coercion rules:**
- `string`: Accepts str, int, float, bool (converts to string)
- `integer`: Accepts int, or parses from string
- `number`: Accepts int/float, or parses from string
- `boolean`: Accepts bool, or parses from string literals (`true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`)
- `object`: Accepts dict, or parses JSON from string
- `array`: Accepts list, tuple, or parses JSON from string

**Example validation:**
```python
from seer.core.runtime.input_validation import coerce_inputs
from seer.core.schema.models import WorkflowSpec

# Validate inputs against spec
coerced = coerce_inputs(spec, provided_inputs)
```

### 2. **Reference Validation** (`workflow_compiler/compiler/validate_refs.py`)

Validates all `${...}` references against the computed type environment.

**Key patterns:**
- Uses `TypeEnvironment` and `Scope` from `workflow_compiler/expr.typecheck`
- Validates references in node inputs, values, prompts, and conditions
- Handles nested scopes for `for_each` loops (with `item_var` and `index_var` locals)
- Raises `ValidationPhaseError` with detailed error messages

**Reference resolution order:**
1. Local variables (loop variables, etc.)
2. State variables (node outputs)
3. Special `inputs` object (workflow inputs)
4. Config variables (if provided)

**Example validation:**
```python
from seer.core.compiler.validate_refs import validate_references
from seer.core.expr.typecheck import TypeEnvironment

# Validate all references in workflow
type_env = TypeEnvironment()
validate_references(spec, type_env)
```

### 3. **Expression Evaluation** (`workflow_compiler/expr/evaluator.py`)

Runtime evaluation of `${...}` expressions.

**Key patterns:**
- Uses `EvaluationContext` with state, inputs, locals, and config
- Supports property access (`${node.field}`) and index access (`${array[0]}`)
- Raises `EvaluationError` for unknown references or invalid access

**Context structure:**
```python
from seer.core.expr.evaluator import EvaluationContext, resolve_reference

ctx = EvaluationContext(
    state={"node1": {"output": "value"}},
    inputs={"user_input": "test"},
    locals={"item": "current_item", "index": 0},
    config={"api_key": "***"}
)
```

### 4. **Schema Models** (`workflow_compiler/schema/models.py`)

Pydantic models for workflow specification.

**Key patterns:**
- All models extend `StrictModel` (extra="forbid", validate_assignment=True)
- Uses `@model_validator` for cross-field validation
- Supports `SchemaRef` (reference to known schema) and `InlineSchema` (JSON Schema)

**Important types:**
- `WorkflowSpec`: Top-level workflow definition
- `Node`: Base class for all block types (if_else, for_each, llm, tool, etc.)
- `InputDef`: Input parameter definition with type and default value
- `OutputContract`: Declares what a node writes (text or JSON with schema)

### 5. **JSON Schema Validation** (`workflow_compiler/schema/jsonschema_adapter.py`)

Core JSON Schema validation utilities using `jsonschema` library (Draft 2020-12).

**Key functions:**
- `get_validator(schema)`: Returns Draft202012Validator
- `validate_instance(instance, schema)`: Validates data against schema
- `check_schema(schema)`: Validates schema structure
- `dereference_schema(schema)`: Resolves `$ref` references
- `format_validation_error(error)`: Human-friendly error messages

**Example:**
```python
from seer.core.schema.jsonschema_adapter import validate_instance

validate_instance({"name": "test"}, {"type": "object", "properties": {"name": {"type": "string"}}})
```

### 6. **Runtime Output Validation** (`workflow_compiler/runtime/validate_output.py`)

Runtime JSON schema validation wrapper.

**Key patterns:**
- Uses `validate_against_schema(value, schema)` to check node outputs
- Raises `ExecutionError` for schema mismatches
- Called after tool/LLM nodes execute when `OutputContract` specifies JSON mode

### 7. **Parsing** (`workflow_compiler/compiler/parse.py`)

Stage 1 compilation: Parse JSON dict → `WorkflowSpec`.

**Key function:**
- `parse_workflow_spec(spec_dict)`: Validates structure with Pydantic
- Raises `ValidationPhaseError` for missing required fields, invalid node types

## Common Validation Scenarios

### When Adding a New Block Type

1. **Define the block model** in `workflow_compiler/schema/models.py`
   - Extend `Node` base class
   - Use Pydantic field validators for constraints
   - Add to the `Node` union type

2. **Add validation logic** if the block has special requirements
   - Reference validation in `validate_refs.py`
   - Input coercion in `input_validation.py`
   - Runtime evaluation in `evaluator.py`

3. **Write tests** in `workflow_compiler/tests/`
   - Test valid configurations
   - Test invalid inputs (should raise appropriate errors)
   - Test edge cases (empty values, null handling, etc.)

### When Reviewing Workflow Changes

Check for:
- **Required fields**: All required fields are present (use Pydantic validation)
- **Type safety**: References resolve to correct types
- **Expression syntax**: `${...}` expressions are valid and references exist
- **Schema compliance**: JSON output matches declared schemas
- **Error handling**: Validation errors provide clear, actionable messages

### When Debugging Compilation Errors

Common error types:
1. **ValidationPhaseError**: Structural/reference validation failed (Stages 1-3)
   - Check `compiler/parse.py:parse_workflow_spec()` for parsing errors
   - Check `compiler/validate_refs.py:validate_references()` for `${...}` reference errors
   - Verify required fields, node types, and references exist

2. **TypeEnvironmentError**: Type environment construction failed (Stage 2)
   - Check `compiler/type_env.py`
   - Verify node output schemas are valid and schema refs resolve

3. **LoweringError**: Lowering to execution plan failed (Stage 4)
   - Check `compiler/lower_control_flow.py`
   - Verify control flow structures (if_else, for_each) are valid

4. **ExecutionError**: Runtime execution failed
   - Check `runtime/validate_output.py` for schema validation errors
   - Check tool execution errors in `runtime/nodes.py`

5. **EvaluationError**: Expression evaluation failed (runtime)
   - Check `expr/evaluator.py:resolve_reference()`
   - Verify state contains expected values for `${...}` expressions

## Testing Workflow Validation

Run validation tests:
```bash
# Run all workflow compiler tests
pytest workflow_compiler/tests/

# Run specific validation tests
pytest workflow_compiler/tests/test_input_validation.py
pytest workflow_compiler/tests/test_jsonschema_adapter.py

# Run with verbose output
pytest -v workflow_compiler/tests/
```

## Best Practices

1. **Fail fast with clear errors**: Validation should catch errors early with actionable messages
2. **Type coercion is lenient**: Allow reasonable conversions (string to number, etc.)
3. **Validation is layered**: 5-stage compilation pipeline catches errors progressively
4. **Preserve extra inputs**: Don't discard inputs not declared in spec (forwards compatibility)
5. **Use nested scopes**: Loop variables (`item_var`, `index_var`) are local to loop body

## Key Files Reference

| File | Purpose | Stage/Phase |
|------|---------|-------------|
| `compiler/parse.py` | Parse JSON → WorkflowSpec | Stage 1 |
| `compiler/type_env.py` | Build type environment | Stage 2 |
| `compiler/validate_refs.py` | Validate `${...}` references | Stage 3 |
| `compiler/lower_control_flow.py` | Lower to execution plan | Stage 4 |
| `compiler/emit_langgraph.py` | Emit LangGraph StateGraph | Stage 5 |
| `schema/models.py` | Pydantic models for workflow spec | All stages |
| `schema/jsonschema_adapter.py` | JSON Schema validation utilities | Stages 2-3, Runtime |
| `runtime/input_validation.py` | Input coercion and validation | Runtime |
| `runtime/validate_output.py` | Runtime output schema validation | Runtime |
| `expr/evaluator.py` | Runtime expression evaluation | Runtime |
| `expr/typecheck.py` | Type checking for references | Stage 2-3 |
| `errors.py` | Error hierarchy | All |
| `README.md` | Architecture overview | Reference |

## Quick Checklist

When validating workflow changes:
- [ ] All required fields are present
- [ ] Input types match InputDef declarations
- [ ] All `${...}` references resolve to existing state/inputs
- [ ] Output contracts specify schemas when mode=json
- [ ] Nested blocks (if_else, for_each) have valid children
- [ ] Loop variables are scoped correctly
- [ ] Tests cover success and error cases
- [ ] Error messages are clear and actionable

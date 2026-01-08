---
name: workflow-validation
description: Validates workflow schemas, expressions, and block configurations in seer. Use when reviewing workflow changes, adding new blocks, debugging workflow compilation errors, or validating workflow JSON structures.
allowed-tools: Read, Grep, Glob, Bash(pytest:*)
---

# Workflow Validation Skill

This Skill helps validate workflow specifications, expressions, and block configurations in the seer backend. Use this when working on workflow-related code to ensure all validation patterns are followed correctly.

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
from workflow_compiler.runtime.input_validation import coerce_inputs
from workflow_compiler.schema.models import WorkflowSpec

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
from workflow_compiler.compiler.validate_refs import validate_references
from workflow_compiler.expr.typecheck import TypeEnvironment

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
from workflow_compiler.expr.evaluator import EvaluationContext, resolve_reference

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

**Validation example:**
```python
from workflow_compiler.schema.models import OutputContract, OutputMode

# Validate output contract
contract = OutputContract(mode=OutputMode.json, schema=schema_spec)
# Raises ValueError if schema is missing when mode=json
```

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
1. **WorkflowCompilerError**: Input validation or coercion failed
   - Check `input_validation.py:coerce_inputs()`
   - Verify InputDef types match provided values

2. **ValidationPhaseError**: Reference validation failed
   - Check `validate_refs.py:validate_references()`
   - Verify `${...}` references exist in scope

3. **TypeCheckError**: Type checking failed for references
   - Check `expr/typecheck.py`
   - Verify node output schemas match expected types

4. **EvaluationError**: Runtime expression evaluation failed
   - Check `expr/evaluator.py:resolve_reference()`
   - Verify state contains expected values

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
3. **Validation is layered**:
   - Stage 1: Pydantic model validation (structure)
   - Stage 2: Type environment building
   - Stage 3: Reference validation
   - Runtime: Expression evaluation
4. **Preserve extra inputs**: Don't discard inputs not declared in spec (for forwards compatibility)
5. **Use nested scopes**: For loop variables are local to loop body

## Key Files Reference

| File | Purpose | When to Check |
|------|---------|---------------|
| `workflow_compiler/schema/models.py` | Pydantic models for workflow spec | Adding new block types |
| `workflow_compiler/runtime/input_validation.py` | Input coercion and validation | Debugging input errors |
| `workflow_compiler/compiler/validate_refs.py` | Reference validation | Debugging `${...}` errors |
| `workflow_compiler/expr/evaluator.py` | Runtime expression evaluation | Debugging runtime errors |
| `workflow_compiler/expr/typecheck.py` | Type checking for references | Understanding type inference |
| `workflow_compiler/tests/` | Test suite for validation | Writing new tests |

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

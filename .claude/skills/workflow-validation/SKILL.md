---
name: workflow-validation
description: Validates workflow schemas, expressions, and block configurations in seer. Use when reviewing workflow changes, adding new blocks, debugging workflow compilation errors, or validating workflow JSON structures.
allowed-tools: Read, Grep, Glob, Bash(pytest:*)
---

# Workflow Validation

Layered across a 5-stage compilation pipeline. See `/workflow-compiler` skill for full architecture.

## Compilation Pipeline

1. **Parse** (`compiler/parse.py`) — JSON → `WorkflowSpec`; raises `ValidationPhaseError`
2. **Type Environment** (`compiler/type_env.py`) — builds type env; raises `TypeEnvironmentError`
3. **Validate References** (`compiler/validate_refs.py`) — checks `${...}` expressions; raises `ValidationPhaseError`
4. **Lower** (`compiler/lower_control_flow.py`) — execution plan; raises `LoweringError`
5. **Emit** (`compiler/emit_langgraph.py`) — LangGraph `StateGraph`

**Runtime:** `input_validation.py` coerces inputs · `validate_output.py` checks node outputs against `OutputContract`

## Error Hierarchy
`WorkflowCompilerError` → `ValidationPhaseError` · `TypeEnvironmentError` · `LoweringError` · `ExecutionError` → `EvaluationError`

## Key Validation Components

- **Input coercion** (`input_validation.py`) — lenient (string↔number↔bool, JSON strings → object/array). Applies `InputDef` defaults. Preserves undeclared inputs.
- **Reference validation** (`validate_refs.py`) — resolves `${...}` order: locals → state → `inputs` → config. Loop vars (`item_var`, `index_var`) scoped to body.
- **Expression evaluation** (`expr/evaluator.py`) — runtime `${node.field}`, `${array[0]}`, nested. Context: state, inputs, locals, config.
- **Schema models** (`schema/models.py`) — `StrictModel` (extra="forbid"). Key types: `WorkflowSpec`, `Node`, `InputDef`, `OutputContract`, `SchemaRef`, `InlineSchema`.
- **JSON Schema utils** (`schema/jsonschema_adapter.py`) — Draft 2020-12. `get_validator()`, `validate_instance()`, `dereference_schema()`, `format_validation_error()`.
- **Output validation** (`validate_output.py`) — post tool/LLM when `OutputContract` mode=json. Raises `ExecutionError`.

## When Adding a New Block Type
1. Extend `Node` in `schema/models.py`, add to `Node` union
2. Add `${...}` reference validation in `validate_refs.py`
3. Add input coercion in `input_validation.py` if needed
4. Write tests: valid config, invalid inputs, edge cases

## Quick Checklist
- [ ] Required fields present; input types match `InputDef`
- [ ] All `${...}` references resolve; output contracts have schemas when mode=json
- [ ] Nested blocks valid; loop vars scoped; tests cover success + error cases

---
name: workflow-compiler
description: Guide for understanding and working with the workflow compiler and runtime in src/seer/core/. Use when implementing node executors, debugging compilation/runtime issues, adding workflow features, or understanding the workflow execution architecture.
allowed-tools: Read, Grep, Glob, Bash(pytest:*)
---

# Workflow Compiler & Runtime

**Key principle:** `src/seer/core/` is pure compilation + runtime — NO database or HTTP. API layer orchestrates compiler + persistence.

## Flow

`WorkflowSpec (JSON)` → `WorkflowCompiler.compile()` (validate → transform → LangGraph) → `CompiledGraph` → Runtime (node executors + state + checkpointing)

## Directory Structure

`src/seer/core/` → `compiler/` (parse · type_env · validate_refs · lower_control_flow · emit_langgraph) · `runtime/` (global_compiler · nodes · execution · state · input_validation · validate_output) · `schema/` (models · jsonschema_adapter) · `expr/` (evaluator · typecheck) · `errors.py · registry/ · triggers/`

## Node Types (`runtime/nodes.py`)
`tool` · `llm` · `code` · `conditional` · `transform` · `trigger` · `input` · `output` · `extract` · `if_else` · `for_each`

## State & Expressions
State keys: `user`, `node_outputs`, `workflow_inputs`, `messages`, `interrupts`. Input resolution: `static` | `input` (node output ref) | `expression` (`${node.field}`) | `workflow_input`. Checkpointing: `MemorySaver` (dev) · `DatabaseCheckpointer` (prod).

## Error Hierarchy
`WorkflowCompilerError` → `ValidationPhaseError` (1–3) · `TypeEnvironmentError` (2) · `LoweringError` (4) · `ExecutionError` → `EvaluationError` (runtime)

## Adding a New Node Type
1. Define Pydantic model in `schema/models.py`, add to `Node` union
2. Implement executor in `runtime/nodes.py`
3. Register in `compiler/compiler.py`
4. Add reference validation in `compiler/validate_refs.py` if needed
5. Write unit tests + full JSON spec tests in `tests/core/`

**Best practices:** No DB/HTTP in compiler · use `StrictModel` · nodes return partial state updates · always add unit + integration spec tests for `src/seer/core/` changes

## Key Files

| File | Purpose |
|------|---------|
| `src/seer/core/runtime/global_compiler.py` | `WorkflowCompilerSingleton` |
| `src/seer/core/runtime/nodes.py` | All node executors |
| `src/seer/core/schema/models.py` | `WorkflowSpec`, `Node`, `InputDef`, `OutputContract` |
| `src/seer/core/compiler/validate_refs.py` | Reference validation |
| `src/seer/core/errors.py` | Error hierarchy |
| `src/seer/core/README.md` | Architecture overview |

## Related Skills
- `/workflow-validation`

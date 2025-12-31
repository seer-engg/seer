"""
Stage 2 — Build the type environment that tracks the schema for each state key.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from workflow_compiler.errors import TypeEnvironmentError
from workflow_compiler.expr.typecheck import (
    TypeEnvironment,
    register_inputs,
    schema_from_output_contract,
)
from workflow_compiler.registry.tool_registry import ToolRegistry
from workflow_compiler.schema.models import (
    ForEachNode,
    LLMNode,
    Node,
    OutputContract,
    TaskKind,
    TaskNode,
    ToolNode,
    WorkflowSpec,
    JSONValue,
)
from workflow_compiler.schema.schema_registry import SchemaRegistry


def build_type_environment(
    spec: WorkflowSpec, *, schema_registry: SchemaRegistry, tool_registry: ToolRegistry
) -> TypeEnvironment:
    env = TypeEnvironment()
    register_inputs(env, spec.inputs)
    for node in spec.nodes:
        _process_node(node, env, schema_registry, tool_registry)
    return env


def _process_node(
    node: Node,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
) -> None:
    if isinstance(node, TaskNode):
        schema = _schema_for_task(node, schema_registry)
        _register_symbol(env, node.out, schema)
        _process_children(node, env, schema_registry, tool_registry)
        return

    if isinstance(node, ToolNode):
        tool_def = tool_registry.get(node.tool)
        schema = tool_def.output_schema
        if node.expect_output is not None:
            expected = schema_from_output_contract(node.expect_output, schema_registry)
            _ensure_schema_match(schema, expected, symbol=node.out or node.id)
        _register_symbol(env, node.out, schema)
        return

    if isinstance(node, LLMNode):
        schema = schema_from_output_contract(node.output, schema_registry)
        _register_symbol(env, node.out, schema)
        return

    if isinstance(node, ForEachNode):
        if node.out:
            if node.output:
                loop_schema = schema_from_output_contract(node.output, schema_registry)
            else:
                loop_schema = {"type": "array"}
            _register_symbol(env, node.out, loop_schema)
        for child in node.body:
            _process_node(child, env, schema_registry, tool_registry)
        return

    # If node is an IfNode (or any other future composite) we still need to
    # process its children.
    _process_children(node, env, schema_registry, tool_registry)


def _process_children(
    node: Node,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
) -> None:
    child_lists: List[List[Node]] = []
    if hasattr(node, "then"):
        child_lists.append(getattr(node, "then"))
    if hasattr(node, "else_"):
        child_lists.append(getattr(node, "else_"))
    if hasattr(node, "body"):
        child_lists.append(getattr(node, "body"))

    for group in child_lists:
        for child in group:
            _process_node(child, env, schema_registry, tool_registry)


def _schema_for_task(node: TaskNode, registry: SchemaRegistry) -> Optional[Dict]:
    if node.output:
        return schema_from_output_contract(node.output, registry)
    if node.kind == TaskKind.set and node.value is not None:
        return _infer_schema_from_value(node.value)
    return None


def _infer_schema_from_value(value: JSONValue) -> Dict:
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if value is None:
        return {"type": "null"}
    if isinstance(value, list):
        item_schema = None
        if value:
            item_schema = _infer_schema_from_value(value[0])
        schema: Dict = {"type": "array"}
        if item_schema:
            schema["items"] = item_schema
        return schema
    if isinstance(value, dict):
        properties = {k: _infer_schema_from_value(v) for k, v in value.items()}
        return {"type": "object", "properties": properties, "additionalProperties": True}
    raise TypeEnvironmentError(f"Unsupported literal type {type(value).__name__}")


def _register_symbol(env: TypeEnvironment, symbol: str | None, schema: Dict | None) -> None:
    if not symbol or schema is None:
        return
    env.register(symbol, schema)


def _ensure_schema_match(actual: Dict, expected: Dict, *, symbol: str) -> None:
    if actual == expected:
        return
    raise TypeEnvironmentError(
        f"Schema mismatch for '{symbol}': registry returned {actual} but node expects {expected}"
    )



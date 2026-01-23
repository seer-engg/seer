"""
Stage 2 — Build the type environment that tracks the schema for each state key.

V2: With explicit edges, nodes no longer have nested children. Loop variables
(item_var, index_var) are written to state and registered as symbols.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set

from seer.core.errors import TypeEnvironmentError
from seer.core.expr.typecheck import (
    TypeEnvironment,
    schema_from_output_contract,
)
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.schema.models import (
    EdgeType,
    ForEachNode,
    JSONValue,
    LLMNode,
    Node,
    TaskKind,
    TaskNode,
    ToolNode,
    TriggerSpec,
    WorkflowSpec,
)
from seer.core.schema.schema_registry import SchemaRegistry

VALID_IDENTIFIER = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def build_type_environment(
    spec: WorkflowSpec, *, schema_registry: SchemaRegistry, tool_registry: ToolRegistry
) -> TypeEnvironment:
    env = TypeEnvironment()

    # Register each trigger by its ID
    if spec.triggers:
        _register_triggers(spec.triggers, env)

    # Process all nodes
    for node in spec.nodes:
        _process_node(node, env, schema_registry, tool_registry)

    # Register loop variable symbols for nodes inside loop bodies
    _register_loop_variables(spec, env)

    return env


def _register_triggers(triggers: List[TriggerSpec], env: TypeEnvironment) -> None:
    """
    Register triggers in type environment.

    Single-trigger workflows: registers both 'trigger' and trigger.id
    Multi-trigger workflows: only registers by trigger.id

    For single-trigger workflows, allows intuitive ${trigger.data.X} syntax.
    For multi-trigger workflows, requires explicit ${trigger_id.data.X} syntax.
    """
    # Always register by trigger ID (explicit, works for all cases)
    for trigger in triggers:
        # Get event schema
        event_schema = trigger.event_schema if trigger.event_schema else {
            "type": "object",
            "additionalProperties": True
        }

        # Register ID as symbol
        env.register(trigger.id, event_schema)

        # Also register sub-properties by ID
        properties = event_schema.get("properties", {})
        for name, schema in properties.items():
            env.register(f"{trigger.id}.{name}", schema)

    # Single-trigger convenience: also register as "trigger"
    if len(triggers) == 1:
        trigger = triggers[0]
        event_schema = trigger.event_schema if trigger.event_schema else {
            "type": "object",
            "additionalProperties": True
        }

        # Register "trigger" as root symbol
        env.register("trigger", event_schema)

        # Register "trigger.property" for all sub-properties
        properties = event_schema.get("properties", {})
        for name, schema in properties.items():
            env.register(f"trigger.{name}", schema)


def _register_loop_variables(spec: WorkflowSpec, env: TypeEnvironment) -> None:
    """
    Register loop variable symbols (item_var, index_var) for ForEachNodes.

    With edge-based control flow, loop variables are written to state and need
    to be registered as symbols for body nodes to access via ${item}, ${index}.
    """
    # Build a map of ForEachNode by id
    for_each_nodes = {n.id: n for n in spec.nodes if isinstance(n, ForEachNode)}

    if not for_each_nodes:
        return

    # For each ForEachNode, register its loop variables
    for node in for_each_nodes.values():
        # Register item_var with a permissive schema (actual type depends on items)
        # The schema could be inferred from the items expression, but for now we use "any"
        env.register(node.item_var, {"type": "object", "additionalProperties": True})
        env.register(node.index_var, {"type": "integer"})


def _process_node(
    node: Node,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
) -> None:
    if isinstance(node, TaskNode):
        schema = _schema_for_task(node, schema_registry)
        _register_symbol(env, node.id, schema)
        return

    if isinstance(node, ToolNode):
        tool_def = tool_registry.get(node.tool)
        schema = tool_def.output_schema
        if node.expect_outputs is not None:
            expected = schema_from_output_contract(node.expect_outputs, schema_registry)
            _ensure_schema_match(schema, expected, symbol=node.id)
        _register_symbol(env, node.id, schema)
        return

    if isinstance(node, LLMNode):
        schema = schema_from_output_contract(node.outputs, schema_registry)
        _register_symbol(env, node.id, schema)
        return

    if isinstance(node, ForEachNode):
        # Register loop output schema using node ID
        if node.outputs:
            loop_schema = schema_from_output_contract(node.outputs, schema_registry)
        else:
            loop_schema = {"type": "array"}
        _register_symbol(env, node.id, loop_schema)
        return

    # IfNode doesn't produce output directly (branches do)
    # No special handling needed


def _schema_for_task(node: TaskNode, registry: SchemaRegistry) -> Optional[Dict]:
    if node.outputs:
        return schema_from_output_contract(node.outputs, registry)
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

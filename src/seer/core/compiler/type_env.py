"""
Stage 2 — Build the type environment that tracks the schema for each state key.

V2: With explicit edges, nodes no longer have nested children. Loop variables
(item_var, index_var) are written to state and registered as symbols.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from seer.core.errors import TypeEnvironmentError
from seer.core.expr.typecheck import (
    TypeEnvironment,
    schema_from_output_contract,
)
from seer.core.registry.mcp_client_registry import MCPClientRegistry, MCPServerConfig
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.schema.models import (
    ForEachNode,
    JSONValue,
    LLMNode,
    MCPNode,
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
    spec: WorkflowSpec,
    *,
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
    _mcp_client_registry: Optional[MCPClientRegistry] = None,
) -> TypeEnvironment:
    """
    Build the type environment synchronously.

    For workflows with MCP nodes, use build_type_environment_async() instead
    to enable compile-time MCP server validation.
    """
    env = TypeEnvironment()

    # Register each trigger by its ID
    if spec.triggers:
        _register_triggers(spec.triggers, env)

    # Process all nodes (sync path — skips MCP validation)
    for node in spec.nodes:
        _process_node_sync(node, env, schema_registry, tool_registry)

    # Register loop variable symbols for nodes inside loop bodies
    _register_loop_variables(spec, env)

    return env


async def build_type_environment_async(
    spec: WorkflowSpec,
    *,
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
    mcp_client_registry: MCPClientRegistry,
) -> TypeEnvironment:
    """
    Build the type environment with MCP compile-time validation.

    Use this in the async compile pipeline when MCP nodes may be present.
    """
    env = TypeEnvironment()

    # Register each trigger by its ID
    if spec.triggers:
        _register_triggers(spec.triggers, env)

    # Process all nodes (async path — includes MCP validation)
    for node in spec.nodes:
        await _process_node_async(node, env, schema_registry, tool_registry, mcp_client_registry)

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


def _process_node_sync(
    node: Node,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
) -> None:
    """Process a node synchronously. MCP nodes are registered with a generic schema."""
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

    if isinstance(node, MCPNode):
        # Sync path: register with user-declared schema or generic fallback
        if node.expect_outputs:
            schema = schema_from_output_contract(node.expect_outputs, schema_registry)
        else:
            schema = {"type": "object", "additionalProperties": True}
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


async def _process_node_async(
    node: Node,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
    mcp_client_registry: MCPClientRegistry,
) -> None:
    """Process a node with async MCP validation."""
    if isinstance(node, MCPNode):
        # Build server config (auth not resolved yet - use placeholder)
        server_config = MCPServerConfig(
            server=node.server,
            server_type=node.server_type,
            auth=None,  # Auth resolved at runtime
        )

        # Validate tool existence and fetch schema at compile time
        try:
            await mcp_client_registry.validate_tool(server_config, node.tool)
            # MCP tools don't typically have output schemas, so we use a generic object schema
            output_schema = {"type": "object", "additionalProperties": True}
        except ConnectionError as exc:
            raise TypeEnvironmentError(
                f"MCP connection failed for server '{node.server}': {exc}"
            ) from exc
        except ValueError as exc:
            raise TypeEnvironmentError(
                f"MCP tool '{node.tool}' not found on server '{node.server}': {exc}"
            ) from exc
        except Exception as exc:
            raise TypeEnvironmentError(
                f"Failed to validate MCP tool '{node.tool}' on server '{node.server}': {exc}"
            ) from exc

        # If expect_outputs specified, use client-declared schema
        if node.expect_outputs:
            schema = schema_from_output_contract(node.expect_outputs, schema_registry)
        else:
            schema = output_schema

        _register_symbol(env, node.id, schema)
        return

    # Non-MCP nodes use the sync path
    _process_node_sync(node, env, schema_registry, tool_registry)


def _schema_for_task(node: TaskNode, registry: SchemaRegistry) -> Optional[Dict]:
    if node.outputs:
        return schema_from_output_contract(node.outputs, registry)
    if node.kind == TaskKind.set and node.value is not None:
        return _infer_schema_from_value(node.value)
    return None


def _infer_schema_from_value(value: JSONValue) -> Dict:  # pylint: disable=too-many-return-statements  # Type checking requires multiple returns
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

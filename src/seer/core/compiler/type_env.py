"""
Stage 2 — Build the type environment that tracks the schema for each state key.

V2: With explicit edges, nodes no longer have nested children. Loop variables
(item_var, index_var) are written to state and registered as symbols.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from seer.core.errors import TypeEnvironmentError
from seer.core.expr.parser import parse_reference_string, REFERENCE_PATTERN, PathSegment, PropertySegment, IndexSegment
from seer.core.expr.typecheck import TypeEnvironment
from seer.core.nodes.base import TypeRegistrationContext
from seer.core.nodes.registry import node_type_registry
from seer.core.registry.mcp_client_registry import MCPClientRegistry
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.schema.models import (
    ForEachNode,
    Node,
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
    Register triggers in type environment by their explicit IDs.

    All workflows use explicit ${trigger_id.X} syntax for consistency.
    """
    # Always register by trigger ID (explicit, works for all cases)
    for trigger in triggers:
        # Get event schema
        # Note: Using {} as additionalProperties (not True) because the typecheck code
        # at _resolve_property() requires additionalProperties to be a dict for property access
        event_schema = trigger.event_schema if trigger.event_schema else {
            "type": "object",
            "additionalProperties": {}
        }

        # Register ID as symbol
        env.register(trigger.id, event_schema)

        # Also register sub-properties by ID
        properties = event_schema.get("properties", {})
        for name, schema in properties.items():
            env.register(f"{trigger.id}.{name}", schema)


def _navigate_schema_by_segment(current: Dict, segment: PathSegment) -> Dict | None:
    """
    Navigate a JSON schema by a single path segment.

    Returns the resulting schema after navigating, or None if navigation fails.
    """
    if isinstance(segment, PropertySegment):
        return _navigate_property_segment(current, segment.key)

    if isinstance(segment, IndexSegment):
        return _navigate_index_segment(current, segment.index)

    return None


def _navigate_property_segment(schema: Dict, key: str) -> Dict | None:
    """Navigate into an object schema by property name."""
    properties = schema.get("properties", {})
    if key in properties:
        return properties[key]
    additional = schema.get("additionalProperties")
    return additional if isinstance(additional, dict) else None


def _navigate_index_segment(schema: Dict, index: int | str) -> Dict | None:
    """Navigate into a schema by index (numeric for arrays, string for objects)."""
    if isinstance(index, int):
        items = schema.get("items")
        return items if isinstance(items, dict) else None
    # String index on object
    properties = schema.get("properties", {})
    if index in properties:
        return properties[index]
    additional = schema.get("additionalProperties")
    return additional if isinstance(additional, dict) else None


def _infer_item_schema_from_items_expression(items_expr: str, env: TypeEnvironment) -> Dict | None:
    """
    Attempt to infer the item schema from a for_each items expression.

    Given an expression like "${prepare_data}" where prepare_data is an array with
    a defined items schema, extract and return that items schema.

    Returns None if inference fails (e.g., expression is not a simple reference,
    source schema is not an array, or array has no items schema).
    """
    # Check if this is a bare ${...} reference
    match = REFERENCE_PATTERN.fullmatch(items_expr.strip())
    if not match:
        return None

    try:
        ref = parse_reference_string(match.group(1))
    except ValueError:
        return None

    # Resolve the root symbol's schema
    source_schema = env.get(ref.root)
    if source_schema is None:
        return None

    # Apply path segments to navigate the schema
    current = source_schema
    for segment in ref.segments:
        current = _navigate_schema_by_segment(current, segment)
        if current is None:
            return None

    # The resolved schema should be an array type with items
    if current.get("type") != "array":
        return None

    items_schema = current.get("items")
    return items_schema if isinstance(items_schema, dict) else None


def _register_loop_variables(spec: WorkflowSpec, env: TypeEnvironment) -> None:
    """
    Register loop variable symbols (item_var, index_var) for ForEachNodes.

    With edge-based control flow, loop variables are written to state and need
    to be registered as symbols for body nodes to access via ${item}, ${index}.

    Type inference: When the items expression references an array with a defined
    items schema, we propagate that schema to the loop variable. This allows
    property access like ${item.name} to be validated correctly.
    """
    # Build a map of ForEachNode by id
    for_each_nodes = {n.id: n for n in spec.nodes if isinstance(n, ForEachNode)}

    if not for_each_nodes:
        return

    # Permissive fallback schema that allows any property access
    # Note: Using {} as additionalProperties (not True) because the typecheck code
    # at _resolve_property() requires additionalProperties to be a dict, not a boolean
    permissive_schema: Dict = {"type": "object", "additionalProperties": {}}

    # For each ForEachNode, register its loop variables
    for node in for_each_nodes.values():
        # Try to infer item schema from the items expression
        inferred_schema = _infer_item_schema_from_items_expression(node.items, env)

        if inferred_schema is not None:
            # Use the inferred schema for type-safe property access
            env.register(node.item_var, inferred_schema)
        else:
            # Fallback to permissive schema that allows any property access
            env.register(node.item_var, permissive_schema)

        env.register(node.index_var, {"type": "integer"})


def _process_node_sync(
    node: Node,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
) -> None:
    """
    Process a node synchronously using the registry.

    All node types must be registered in the node_type_registry.
    There is no fallback to isinstance checks - if a node type is not
    registered, it's an error.
    """
    node_impl = node_type_registry.get(node.type)
    if node_impl is None:
        raise TypeEnvironmentError(f"Unknown node type: '{node.type}' - not registered in node_type_registry")

    ctx = TypeRegistrationContext(
        schema_registry=schema_registry,
        tool_registry=tool_registry,
    )
    node_impl.register_type_sync(node, env, ctx)


async def _process_node_async(
    node: Node,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
    mcp_client_registry: MCPClientRegistry,
) -> None:
    """
    Process a node with async MCP validation using the registry.

    All node types must be registered in the node_type_registry.
    There is no fallback to isinstance checks - if a node type is not
    registered, it's an error.
    """
    node_impl = node_type_registry.get(node.type)
    if node_impl is None:
        raise TypeEnvironmentError(f"Unknown node type: '{node.type}' - not registered in node_type_registry")

    ctx = TypeRegistrationContext(
        schema_registry=schema_registry,
        tool_registry=tool_registry,
        mcp_client_registry=mcp_client_registry,
    )
    await node_impl.register_type_async(node, env, ctx)

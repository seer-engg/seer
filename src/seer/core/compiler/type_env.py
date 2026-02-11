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
from seer.core.expr.typecheck import (
    TypeEnvironment,
    schema_from_output_contract,
)
from seer.core.registry.mcp_client_registry import MCPClientRegistry, MCPServerConfig
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.schema.models import (
    BrowserNode,
    ForEachNode,
    HITLInputType,
    HITLNode,
    LLMNode,
    MCPNode,
    Node,
    OutputMode,
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


def _build_hitl_output_schema(node: HITLNode) -> Dict:
    """
    Build output schema dynamically from HITL input field definitions.

    Each input field becomes a property in the output schema with the appropriate type.
    """
    properties: Dict = {}
    required: List[str] = []

    for input_field in node.inputs:
        field_schema: Dict = {}

        if input_field.input_type == HITLInputType.text:
            field_schema = {"type": "string"}
        elif input_field.input_type == HITLInputType.number:
            field_schema = {"type": "number"}
        elif input_field.input_type == HITLInputType.boolean:
            field_schema = {"type": "boolean"}
        elif input_field.input_type == HITLInputType.single_choice:
            # Single choice returns the selected value as a string
            if input_field.options:
                field_schema = {
                    "type": "string",
                    "enum": [opt.value for opt in input_field.options],
                }
            else:
                field_schema = {"type": "string"}
        elif input_field.input_type == HITLInputType.multi_choice:
            # Multi choice returns an array of selected values
            if input_field.options:
                field_schema = {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [opt.value for opt in input_field.options],
                    },
                }
            else:
                field_schema = {"type": "array", "items": {"type": "string"}}

        properties[input_field.id] = field_schema
        if input_field.required:
            required.append(input_field.id)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _process_tool_node(
    node: ToolNode,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
) -> None:
    """Process a ToolNode and register its output schema."""
    tool_def = tool_registry.get(node.tool)
    schema = tool_def.output_schema
    if node.expect_outputs is not None:
        expected = schema_from_output_contract(node.expect_outputs, schema_registry)
        _ensure_schema_match(schema, expected, symbol=node.id)
    _register_symbol(env, node.id, schema)


def _process_mcp_node_sync(
    node: MCPNode,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
) -> None:
    """Process an MCPNode synchronously with user-declared or generic schema."""
    if node.expect_outputs:
        schema = schema_from_output_contract(node.expect_outputs, schema_registry)
    else:
        # Note: Using {} as additionalProperties (not True) because the typecheck code
        # at _resolve_property() requires additionalProperties to be a dict for property access
        schema = {"type": "object", "additionalProperties": {}}
    _register_symbol(env, node.id, schema)


def _process_llm_node(
    node: LLMNode,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
) -> None:
    """Process an LLMNode and validate structured output constraints."""
    schema = schema_from_output_contract(node.outputs, schema_registry)

    # Validate: OpenAI structured outputs require root type to be "object"
    if node.outputs.mode == OutputMode.json:
        root_type = schema.get("type")
        if root_type == "array":
            raise TypeEnvironmentError(
                f"LLM node '{node.id}': JSON output schema must have root type 'object', "
                f"not 'array'. OpenAI structured outputs do not support array root types. "
                f"Wrap your array in an object property, e.g.: "
                f'{{"type": "object", "properties": {{"items": <your-array-schema>}}}}'
            )

    _register_symbol(env, node.id, schema)


def _process_for_each_node(
    node: ForEachNode,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
) -> None:
    """Process a ForEachNode and register its loop output schema."""
    if node.outputs:
        loop_schema = schema_from_output_contract(node.outputs, schema_registry)
    else:
        loop_schema = {"type": "array"}
    _register_symbol(env, node.id, loop_schema)


def _process_hitl_node(node: HITLNode, env: TypeEnvironment) -> None:
    """Process an HITLNode and register its dynamically-built output schema."""
    schema = _build_hitl_output_schema(node)
    _register_symbol(env, node.id, schema)


def _process_browser_node(
    node: BrowserNode,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
) -> None:
    """
    Process a BrowserNode and register its output schema.

    Browser nodes ALWAYS output {success, result, extracted_data, final_url, screenshots}.
    If expect_outputs is specified, its schema applies to extracted_data, not the root.
    This ensures ${browser_id.extracted_data.field} references validate correctly.
    """
    # Determine the extracted_data schema
    if node.expect_outputs:
        extracted_data_schema = schema_from_output_contract(node.expect_outputs, schema_registry)
    else:
        # Default: permissive object that allows any property access
        # Note: Using {} as additionalProperties (not True) because the typecheck code
        # at _resolve_property() requires additionalProperties to be a dict for property access
        extracted_data_schema = {"type": "object", "additionalProperties": {}}

    # Browser always produces the same envelope structure
    # Note: final_url can be null on error/timeout, so we use ["string", "null"]
    schema = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "result": {"type": "string"},
            "extracted_data": extracted_data_schema,
            "final_url": {"type": ["string", "null"]},
            "screenshots": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": {},
    }
    _register_symbol(env, node.id, schema)


def _process_node_sync(
    node: Node,
    env: TypeEnvironment,
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
) -> None:
    """Process a node synchronously. MCP nodes are registered with a generic schema."""
    if isinstance(node, ToolNode):
        _process_tool_node(node, env, schema_registry, tool_registry)
    elif isinstance(node, MCPNode):
        _process_mcp_node_sync(node, env, schema_registry)
    elif isinstance(node, LLMNode):
        _process_llm_node(node, env, schema_registry)
    elif isinstance(node, ForEachNode):
        _process_for_each_node(node, env, schema_registry)
    elif isinstance(node, HITLNode):
        _process_hitl_node(node, env)
    elif isinstance(node, BrowserNode):
        _process_browser_node(node, env, schema_registry)
    # IfNode doesn't produce output directly (branches do)


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
            # Note: Using {} as additionalProperties (not True) because the typecheck code
            # at _resolve_property() requires additionalProperties to be a dict for property access
            output_schema = {"type": "object", "additionalProperties": {}}
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

    # Browser nodes use sync path (no async validation needed)
    if isinstance(node, BrowserNode):
        _process_browser_node(node, env, schema_registry)
        return

    # Non-MCP nodes use the sync path
    _process_node_sync(node, env, schema_registry, tool_registry)


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

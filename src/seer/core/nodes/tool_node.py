"""
ToolNode - Execute registered tools from the tool registry.

This is the simplest node type and serves as the reference implementation
for the "one file per node" pattern.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict

from seer.core.errors import ExecutionError
from seer.core.expr.typecheck import schema_from_output_contract
from seer.core.nodes.base import BaseNodeType, NodeExecutionContext, TypeRegistrationContext, get_trace_key
from seer.core.nodes.registry import register_node_type
# Import model from schema/models.py (canonical location)
from seer.core.schema.models import ToolNode
from seer.tools.coercion import coerce_arguments

if TYPE_CHECKING:
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.runtime.nodes import RuntimeServices
    from seer.core.schema.models import NodeBase

logger = logging.getLogger(__name__)


# =============================================================================
# Node Type Implementation
# =============================================================================

class ToolNodeType(BaseNodeType):
    """Implementation of the tool node type."""

    @property
    def type_literal(self) -> str:
        return "tool"

    @property
    def model_class(self) -> type["NodeBase"]:
        return ToolNode

    async def execute_async(  # pylint: disable=too-many-locals  # Reason: Node execution requires many context variables
        self,
        node: ToolNode,  # type: ignore[override]
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
    ) -> Dict[str, Any]:
        """
        Execute a tool node asynchronously.

        Steps:
        1. Evaluate input expressions against current state
        2. Apply schema-driven coercion (fixes LLM quoting issues)
        3. Execute the tool's async handler
        4. Validate output against schema if specified
        5. Return state updates with trace data
        """
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.core.expr.evaluator import EvaluationContext, evaluate_value
        from seer.core.runtime.state import INTERNAL_STATE_PREFIX
        from seer.core.runtime.validate_output import validate_against_schema

        # Build evaluation context
        visible_state = {k: v for k, v in ctx.state.items() if not k.startswith(INTERNAL_STATE_PREFIX)}
        eval_ctx = EvaluationContext(
            state=visible_state,
            locals=ctx.locals_ctx or {},
            config=ctx.config,
            trigger=ctx.trigger,
        )

        # Evaluate input expressions
        inputs: Dict[str, Any] = {}
        for key, expr in node.inputs.items():
            try:
                inputs[key] = evaluate_value(eval_ctx, expr)
            except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Capture eval error in trace instead of failing
                inputs[key] = {"__error__": str(e), "__expression__": expr}

        # Execute tool
        try:
            tool_def = services.tool_registry.get(node.tool)

            # Apply schema-driven coercion (see BUG FIX comment in runtime/nodes.py)
            inputs = coerce_arguments(inputs, tool_def.input_schema)

            if tool_def.async_handler is None:
                raise ExecutionError(f"Tool '{node.tool}' has no async handler registered")

            result = await tool_def.async_handler(inputs, dict(ctx.config), ctx.runtime_context)

        except Exception as exc:
            # Build error trace for failed execution (loop-aware key)
            trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
            error_trace = {
                trace_key: {
                    "node_id": node.id,
                    "node_type": "tool",
                    "inputs": inputs,
                    "error": {"type": exc.__class__.__name__, "message": str(exc)},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                }
            }
            ctx.state.update(error_trace)  # type: ignore[arg-type]
            raise ExecutionError(f"Tool '{node.tool}' failed: {exc}", trace_data=error_trace) from exc

        # Validate output against schema if specified
        schema = services.type_env.get(node.id)
        if schema is not None:
            validate_against_schema(schema, result, schema_id=node.id)

        # Prepare output
        if node.id.startswith(INTERNAL_STATE_PREFIX):
            raise ExecutionError(f"Node IDs starting with '{INTERNAL_STATE_PREFIX}' are reserved")

        output = {node.id: result}

        # Add trace data (loop-aware key for nested loop support)
        trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
        output[trace_key] = {
            "node_id": node.id,
            "node_type": "tool",
            "inputs": inputs,
            "output": result,
            "output_key": node.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "succeeded",
        }

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Tool node '%s' output keys: %s",
                node.id,
                list(output.keys()),
                extra={"node_id": node.id, "output_keys": list(output.keys())},
            )

        return output

    def register_type_sync(
        self,
        node: ToolNode,  # type: ignore[override]
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """
        Register tool node's output schema in the type environment.

        The output schema is derived from the ToolRegistry. If expect_outputs
        is specified, it's validated against the registry schema.
        """
        from seer.core.errors import TypeEnvironmentError  # pylint: disable=import-outside-toplevel  # Reason: Rare error path

        tool_def = ctx.tool_registry.get(node.tool)
        schema = tool_def.output_schema

        if node.expect_outputs is not None:
            expected = schema_from_output_contract(node.expect_outputs, ctx.schema_registry)
            if schema != expected:
                raise TypeEnvironmentError(
                    f"Schema mismatch for '{node.id}': registry returned {schema} but node expects {expected}"
                )

        if node.id and schema:
            env.register(node.id, schema)


# Auto-register on module import
register_node_type(ToolNodeType())

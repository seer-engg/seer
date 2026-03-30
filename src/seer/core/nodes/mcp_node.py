"""
MCPNode - Execute tools from external MCP (Model Context Protocol) servers.

Supports both HTTP and stdio MCP servers with optional authentication.
Auth expressions like ${secrets.api_key} are resolved at runtime.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.core.errors import ExecutionError
from seer.core.expr.typecheck import schema_from_output_contract
from seer.logger import get_logger
from seer.core.nodes.base import BaseNodeType, NodeExecutionContext, TypeRegistrationContext, get_trace_key
from seer.core.nodes.registry import register_node_type
# Import model from schema/models.py (canonical location)
from seer.core.schema.models import MCPNode

if TYPE_CHECKING:
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.runtime.nodes import RuntimeServices
    from seer.core.schema.models import NodeBase

logger = get_logger(__name__)


# =============================================================================
# Node Type Implementation
# =============================================================================

class MCPNodeType(BaseNodeType):
    """Implementation of the MCP node type."""

    @property
    def type_literal(self) -> str:
        return "mcp"

    @property
    def model_class(self) -> type["NodeBase"]:
        return MCPNode

    async def execute_async(  # pylint: disable=too-many-locals  # Reason: MCP execution requires imports, eval ctx, server config, result, trace, and output vars
        self,
        node: MCPNode,  # type: ignore[override]
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
    ) -> Dict[str, Any]:
        """Execute MCP node with runtime auth resolution."""
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.core.expr.evaluator import EvaluationContext
        from seer.core.runtime.state import INTERNAL_STATE_PREFIX
        from seer.core.runtime.validate_output import validate_against_schema

        if services.mcp_client_registry is None:
            raise ExecutionError(
                "MCPClientRegistry is required to execute MCP nodes. "
                "Ensure the compiler is initialized with MCP support."
            )

        visible_state = {k: v for k, v in ctx.state.items() if not k.startswith(INTERNAL_STATE_PREFIX)}
        eval_ctx = EvaluationContext(
            state=visible_state,
            locals=ctx.locals_ctx or {},
            config=ctx.config,
            trigger=ctx.trigger,
            vars=ctx.vars,
        )

        inputs = self._evaluate_inputs(node, eval_ctx)
        server_url, server_type, saved_auth = await self._resolve_server_config(node, ctx)
        server_config, resolved_auth = self._build_server_config(server_url, server_type, saved_auth, node, eval_ctx)

        try:
            result = await self._invoke_tool(services, server_config, node, inputs)
        except Exception as exc:
            trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
            error_trace = {
                trace_key: {
                    "node_id": node.id,
                    "node_type": "mcp",
                    "inputs": inputs,
                    "error": {"type": exc.__class__.__name__, "message": str(exc)},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                }
            }
            ctx.state.update(error_trace)  # type: ignore[arg-type]
            raise ExecutionError(f"MCP tool '{node.tool}' failed: {exc}", trace_data=error_trace) from exc

        if node.expect_outputs:
            type_schemas = services.type_env.as_dict()
            schema = type_schemas.get(node.id)
            if schema:
                validate_against_schema(schema, result, schema_id=node.id)

        output = {node.id: result}
        self._attach_trace(output, node, ctx, inputs, result, resolved_auth)
        return output

    async def _resolve_server_config(
        self,
        node: MCPNode,
        ctx: NodeExecutionContext,
    ) -> tuple[str, str, Optional[Dict[str, Any]]]:
        """Resolve server URL, type, and saved auth from node config or saved MCP server config."""
        server_url: str = node.server
        server_type: str = node.server_type
        saved_auth: Optional[Dict[str, Any]] = None

        if node.mcp_server_config_id and not node.server:
            if ctx.runtime_context is None or ctx.runtime_context.mcp_config_resolver is None:
                raise ExecutionError(
                    "MCP server config resolver is required when using mcp_server_config_id. "
                    "Ensure the runtime context is initialized with MCP config support."
                )
            saved_config = await ctx.runtime_context.mcp_config_resolver.resolve(node.mcp_server_config_id)
            server_url = saved_config["server"]
            server_type = saved_config.get("server_type", "http")
            saved_auth = saved_config.get("auth")

        if not server_url:
            raise ExecutionError("MCP node has no server URL configured (neither inline nor via saved config)")

        return server_url, server_type, saved_auth

    def _build_server_config(  # pylint: disable=too-many-positional-arguments  # Reason: server config requires URL, type, saved auth, node, and eval context — no natural grouping
        self,
        server_url: str,
        server_type: str,
        saved_auth: Optional[Dict[str, Any]],
        node: MCPNode,
        eval_ctx: Any,
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        """Build MCPServerConfig with resolved and merged auth. Returns (config, resolved_auth)."""
        from seer.core.registry.mcp_client_registry import MCPServerConfig  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

        resolved_auth = self._resolve_auth(node, eval_ctx)
        if saved_auth and not resolved_auth:
            resolved_auth = saved_auth
        elif saved_auth and resolved_auth:
            for section in ("headers", "env"):
                if section in saved_auth and section not in resolved_auth:
                    resolved_auth[section] = saved_auth[section]

        valid_server_type = server_type if server_type in ("http", "stdio") else "http"
        server_config = MCPServerConfig(
            server=server_url,
            server_type=valid_server_type,  # type: ignore[arg-type]
            auth=resolved_auth,
        )
        return server_config, resolved_auth

    def _evaluate_inputs(self, node: MCPNode, ctx: Any) -> Dict[str, Any]:
        """Evaluate input expressions."""
        from seer.core.expr.evaluator import evaluate_value  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

        inputs = {}
        for key, expr in node.inputs.items():
            try:
                inputs[key] = evaluate_value(ctx, expr)
            except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Capture eval error in trace
                inputs[key] = {"__error__": str(e), "__expression__": expr}
        return inputs

    def _resolve_auth(self, node: MCPNode, ctx: Any) -> Optional[Dict[str, Any]]:
        """Resolve runtime auth expressions."""
        from seer.core.expr.evaluator import evaluate_value  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

        if not node.auth:
            return None

        resolved: Dict[str, Any] = {}
        for section in ("headers", "env"):
            if section in node.auth:
                resolved[section] = {
                    k: evaluate_value(ctx, v) if isinstance(v, str) and "${" in v else v
                    for k, v in node.auth[section].items()
                }
        return resolved

    async def _invoke_tool(
        self,
        services: "RuntimeServices",
        server_config: Any,
        node: MCPNode,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Invoke the MCP tool and normalize result to dict."""
        try:
            result = await services.mcp_client_registry.invoke_tool(
                server_config, node.tool, inputs
            )
        except ConnectionError as exc:
            raise ExecutionError(f"MCP connection failed for server '{node.server}': {exc}") from exc
        except Exception as exc:
            raise ExecutionError(
                f"MCP tool '{node.tool}' failed on server '{node.server}': {exc}"
            ) from exc

        # MCP tools return strings or content lists; normalize to dict
        if not isinstance(result, dict):
            result = {"result": result}
        return result

    def _attach_trace(  # pylint: disable=too-many-positional-arguments  # Reason: Trace data requires multiple fields
        self,
        output: Dict[str, Any],
        node: MCPNode,
        ctx: NodeExecutionContext,
        inputs: Dict[str, Any],
        result: Any,
        resolved_auth: Optional[Dict[str, Any]],
    ) -> None:
        """Attach trace data with redacted auth (loop-aware key)."""
        trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})

        safe_auth = None
        if resolved_auth:
            safe_auth = {
                "headers": {k: "***REDACTED***" for k in resolved_auth.get("headers", {})},
                "env": {k: "***REDACTED***" for k in resolved_auth.get("env", {})},
            }

        output[trace_key] = {
            "node_id": node.id,
            "node_type": "mcp",
            "server": node.server,
            "server_type": node.server_type,
            "tool": node.tool,
            "auth": safe_auth,
            "inputs": inputs,
            "output": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "succeeded",
        }

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "MCP node '%s' (server='%s', tool='%s') output keys: %s",
                node.id, node.server, node.tool, list(output.keys()),
            )

    def register_type_sync(
        self,
        node: MCPNode,  # type: ignore[override]
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """
        Register MCP node's output schema (sync path - uses generic schema).

        For actual MCP validation, use register_type_async.
        """
        if node.expect_outputs:
            schema = schema_from_output_contract(node.expect_outputs, ctx.schema_registry)
        else:
            # Generic object schema for property access
            schema = {"type": "object", "additionalProperties": {}}

        if node.id:
            env.register(node.id, schema)

    async def register_type_async(
        self,
        node: MCPNode,  # type: ignore[override]
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """
        Register MCP node's output schema with async server validation.

        Validates that the tool exists on the MCP server at compile time.
        """
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.core.errors import TypeEnvironmentError
        from seer.core.registry.mcp_client_registry import MCPServerConfig

        if ctx.mcp_client_registry is None:
            # Fall back to sync path if no MCP registry available
            self.register_type_sync(node, env, ctx)
            return

        # When using a saved config, server URL isn't known at compile time
        # Skip server validation and fall back to generic schema
        if node.mcp_server_config_id and not node.server:
            self.register_type_sync(node, env, ctx)
            return

        # Build server config (auth not resolved yet)
        server_config = MCPServerConfig(
            server=node.server,
            server_type=node.server_type,
            auth=None,
        )

        # Validate tool existence at compile time
        try:
            await ctx.mcp_client_registry.validate_tool(server_config, node.tool)
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

        # Use client-declared schema if specified
        if node.expect_outputs:
            schema = schema_from_output_contract(node.expect_outputs, ctx.schema_registry)
        else:
            schema = output_schema

        if node.id:
            env.register(node.id, schema)


# Auto-register on module import
register_node_type(MCPNodeType())

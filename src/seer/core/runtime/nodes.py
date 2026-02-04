# pylint: disable=too-many-lines,unused-argument,import-outside-toplevel,broad-exception-caught,too-many-return-statements,no-else-return
# Reason: Node runtime contains executors for all workflow node types; splitting would reduce cohesion; context params required by interface
"""
Runtime node executors – each workflow node type is compiled into a callable
that LangGraph can schedule. Control flow nodes (if / for_each) execute their
children inline using the same dispatch logic, ensuring consistent semantics
between top-level and nested blocks.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence

from langgraph._internal._runnable import RunnableCallable

from seer.core.errors import ExecutionError
from seer.core.expr.evaluator import (
    EvaluationContext,
    evaluate_condition,
    evaluate_value,
    render_template,
)
from seer.core.expr.typecheck import TypeEnvironment
from seer.core.registry.mcp_client_registry import MCPClientRegistry
from seer.core.registry.model_registry import ModelRegistry
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.core.runtime.state import INTERNAL_STATE_PREFIX, WorkflowState
from seer.core.runtime.validate_output import validate_against_schema
from seer.core.schema.models import (
    ForEachNode,
    IfNode,
    LLMNode,
    MCPNode,
    Node,
    OutputMode,
    ToolNode,
)
from seer.core.schema.schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeServices:
    schema_registry: SchemaRegistry
    tool_registry: ToolRegistry
    model_registry: ModelRegistry
    type_env: TypeEnvironment
    mcp_client_registry: Optional[MCPClientRegistry] = None


class NodeRuntime:
    def __init__(self, services: RuntimeServices) -> None:
        self.services = services
        self._type_schemas = services.type_env.as_dict()
        self._current_trigger: Mapping[str, Any] | None = None
        self._current_context: WorkflowRuntimeContext | None = None
        self._loop_body_map: Dict[str, str] = {}  # node_id -> parent_loop_id

    def build_runner(self, node: Node) -> RunnableCallable:
        def runner(
            state: WorkflowState,
            config: Mapping[str, Any] | None = None,
            context: WorkflowRuntimeContext | None = None,
        ) -> Dict[str, Any]:
            return self._run_node(node, state, config or {}, locals_ctx=None, context=context)

        async def runner_async(
            state: WorkflowState,
            config: Mapping[str, Any] | None = None,
            context: WorkflowRuntimeContext | None = None,
        ) -> Dict[str, Any]:
            return await self._run_node_async(node, state, config or {}, locals_ctx=None, context=context)

        return RunnableCallable(func=runner, afunc=runner_async, name=f"node:{node.id}")

    def bind_trigger(self, trigger: Mapping[str, Any] | None) -> None:
        """Bind trigger event envelope for ${trigger.*} resolution."""
        self._current_trigger = dict(trigger) if trigger else None

    def bind_context(self, context: WorkflowRuntimeContext | None) -> None:
        self._current_context = context

    def set_loop_body_map(self, loop_body_map: Dict[str, str]) -> None:
        """Set mapping from node_id to parent loop_id for nodes inside loops."""
        self._loop_body_map = loop_body_map

    def _get_trace_key(self, node_id: str, state: WorkflowState) -> str:
        """
        Generate trace key for a node, including loop iteration if inside a loop.

        Returns:
            - `_trace_{node_id}` if not in a loop
            - `_trace_{node_id}_iter_{N}` if in a loop (where N is the current iteration)
        """
        # Check if this node is inside a loop
        parent_loop_id = self._loop_body_map.get(node_id)
        if not parent_loop_id:
            return f"_trace_{node_id}"

        # Get the current loop iteration
        loop_state_key = f"_loop_{parent_loop_id}"
        loop_state = state.get(loop_state_key)
        if not loop_state or not isinstance(loop_state, dict):
            return f"_trace_{node_id}"

        current_index = loop_state.get("current_index", 0)
        return f"_trace_{node_id}_iter_{current_index}"

    def _check_llm_credit_limit_sync(self) -> None:
        """
        Run the credit gate in synchronous contexts before an LLM call.
        """
        if not self._current_context or not self._current_context.user:
            return

        from seer.observability.credit_gate import check_credit_limit

        try:
            asyncio.run(check_credit_limit(self._current_context.user))
        except Exception as exc:  # noqa: BLE001 - propagate credit failures, log others
            if exc.__class__.__name__ == "CreditLimitExceeded":
                raise
            logger.error("Credit limit check failed: %s", exc)

    async def _check_llm_credit_limit_async(self) -> None:
        """
        Run the credit gate in async contexts before an LLM call.
        """
        if not self._current_context or not self._current_context.user:
            return

        from seer.observability.credit_gate import check_credit_limit

        try:
            await check_credit_limit(self._current_context.user)
        except Exception as exc:  # noqa: BLE001 - propagate credit failures, log others
            if exc.__class__.__name__ == "CreditLimitExceeded":
                raise
            logger.error("Credit limit check failed: %s", exc)

    def _track_llm_usage_async(self, usage_metadata: Dict[str, Any]) -> None:
        """
        Track LLM usage asynchronously (fire and forget).

        Args:
            usage_metadata: Dict with input_tokens, output_tokens, reasoning_tokens, model
        """
        if not self._current_context or not self._current_context.user:
            logger.warning("Cannot track LLM usage: no user context")
            return

        from seer.observability.cost_tracking import CostTracker
        from seer.observability.exceptions import RunCostCapExceeded

        async def do_track():
            try:
                await CostTracker.track_and_enforce_cap(
                    usage_metadata=usage_metadata,
                    context=self._current_context,
                    operation="workflow_execution",
                )
            except RunCostCapExceeded:
                # Re-raise cost cap exception to stop execution
                raise
            except Exception as e:
                # Log error but don't fail workflow for other tracking errors
                logger.error(
                    "Failed to track LLM usage: %s",
                    str(e),
                    exc_info=True,
                    extra={
                        "user_id": self._current_context.user.user_id,
                        "model": usage_metadata.get("model"),
                        "error": str(e),
                    },
                )

        # Fire and forget (don't wait for tracking to complete)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(do_track())
            else:
                loop.run_until_complete(do_track())
        except Exception as e:
            logger.error("Failed to schedule LLM usage tracking: %s", e)

    # ------------------------------------------------------------------
    # Node handlers
    # ------------------------------------------------------------------
    def _run_node(
        self,
        node: Node,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        if isinstance(node, ToolNode):
            return self._run_tool(node, state, config, locals_ctx=locals_ctx, context=context)
        if isinstance(node, MCPNode):
            return self._run_mcp(node, state, config, locals_ctx=locals_ctx, context=context)
        if isinstance(node, LLMNode):
            self._check_llm_credit_limit_sync()
            return self._run_llm(node, state, config, locals_ctx=locals_ctx)
        if isinstance(node, IfNode):
            return self._run_if(node, state, config, locals_ctx=locals_ctx, context=context)
        if isinstance(node, ForEachNode):
            return self._run_for_each(node, state, config, locals_ctx=locals_ctx, context=context)
        raise ExecutionError(f"Unsupported node type '{node.type}'")

    async def _run_node_async(
        self,
        node: Node,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        if isinstance(node, ToolNode):
            return await self._run_tool_async(node, state, config, locals_ctx=locals_ctx, context=context)
        if isinstance(node, MCPNode):
            return await self._run_mcp_async(node, state, config, locals_ctx=locals_ctx, context=context)
        if isinstance(node, LLMNode):
            await self._check_llm_credit_limit_async()
            return self._run_llm(node, state, config, locals_ctx=locals_ctx)
        if isinstance(node, IfNode):
            return await self._run_if_async(node, state, config, locals_ctx=locals_ctx, context=context)
        if isinstance(node, ForEachNode):
            return await self._run_for_each_async(node, state, config, locals_ctx=locals_ctx, context=context)
        raise ExecutionError(f"Unsupported node type '{node.type}'")

    def _run_tool(
        self,
        node: ToolNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        # STEP 1: Capture inputs (AFTER evaluation, BEFORE execution)
        inputs = self._capture_node_inputs(node, state, config, locals_ctx)

        # STEP 2: Execute tool (existing logic)
        try:
            tool_def = self.services.tool_registry.get(node.tool)
            runtime_context = context or self._current_context
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Running tool node '%s' (tool='%s') with config_keys=%s user_in_context=%s config_type=%s configurable_keys=%s",
                    node.id,
                    node.tool,
                    sorted(config.keys()),
                    bool(getattr(runtime_context, "user", None)),
                    type(config).__name__,
                    sorted((config.get("configurable") or {}).keys()),
                )
            result = tool_def.handler(inputs, dict(config), runtime_context)
        except Exception as exc:
            error_trace = self._write_error_trace(node, state, inputs, exc=exc, node_type='tool')
            # CRITICAL: Update state with error trace BEFORE raising
            # This ensures the trace is persisted to checkpoints even when node fails
            state.update(error_trace)  # type: ignore[arg-type]  # WorkflowState is TypedDict with total=False, allows any keys
            raise ExecutionError(f"Tool '{node.tool}' failed: {exc}", trace_data=error_trace) from exc

        # STEP 3: Prepare output (existing logic)
        output = self._prepare_output(node.id, result)

        # STEP 4: Store trace data
        # Use single underscore prefix to avoid LangGraph filtering double-underscore keys
        trace_key = self._get_trace_key(node.id, state)
        output[trace_key] = {
            'node_id': node.id,
            'node_type': 'tool',
            'inputs': inputs,  # Actual runtime inputs
            'output': result,  # Raw tool result (before prepare_output)
            'output_key': node.id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'succeeded',
        }

        # Diagnostic logging: Verify trace key is in output
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Tool node '%s' output keys: %s, trace_key present: %s", node.id, list(output.keys()), trace_key in output,
                extra={"node_id": node.id, "output_keys": list(output.keys()), "trace_key": trace_key}
            )

        return output

    async def _run_tool_async(
        self,
        node: ToolNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        # STEP 1: Capture inputs (AFTER evaluation, BEFORE execution)
        inputs = self._capture_node_inputs(node, state, config, locals_ctx)

        # STEP 2: Execute tool (existing logic)
        try:
            tool_def = self.services.tool_registry.get(node.tool)
            runtime_context = context or self._current_context
            handler = getattr(tool_def, "async_handler", None)
            if handler is None:
                result = await asyncio.to_thread(tool_def.handler, inputs, dict(config), runtime_context)
            else:
                result = await handler(inputs, dict(config), runtime_context)
        except Exception as exc:
            error_trace = self._write_error_trace(node, state, inputs, exc=exc, node_type='tool')
            # CRITICAL: Update state with error trace BEFORE raising
            # This ensures the trace is persisted to checkpoints even when node fails
            state.update(error_trace)  # type: ignore[arg-type]  # WorkflowState is TypedDict with total=False, allows any keys
            raise ExecutionError(f"Tool '{node.tool}' failed: {exc}", trace_data=error_trace) from exc

        # STEP 3: Prepare output (existing logic)
        output = self._prepare_output(node.id, result)

        # STEP 4: Store trace data
        # Use single underscore prefix to avoid LangGraph filtering double-underscore keys
        trace_key = self._get_trace_key(node.id, state)
        output[trace_key] = {
            'node_id': node.id,
            'node_type': 'tool',
            'inputs': inputs,  # Actual runtime inputs
            'output': result,  # Raw tool result (before prepare_output)
            'output_key': node.id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'succeeded',
        }

        # Diagnostic logging: Verify trace key is in output
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Tool node '%s' (async) output keys: %s, trace_key present: %s", node.id, list(output.keys()), trace_key in output,
                extra={"node_id": node.id, "output_keys": list(output.keys()), "trace_key": trace_key}
            )

        return output

    def _run_mcp(
        self,
        node: MCPNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        """Execute MCP node synchronously (delegates to async implementation)."""
        return asyncio.run(
            self._run_mcp_async(node, state, config, locals_ctx=locals_ctx, context=context)
        )

    async def _run_mcp_async(
        self,
        node: MCPNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        """Execute MCP node with runtime auth resolution."""
        from seer.core.registry.mcp_client_registry import MCPServerConfig

        if self.services.mcp_client_registry is None:
            raise ExecutionError(
                "MCPClientRegistry is required to execute MCP nodes. "
                "Ensure the compiler is initialized with MCP support."
            )

        inputs = self._capture_node_inputs(node, state, config, locals_ctx)
        ctx = self._build_eval_context(state, config, locals_ctx)
        resolved_auth = self._resolve_mcp_auth(node, ctx)

        server_config = MCPServerConfig(
            server=node.server,
            server_type=node.server_type,
            auth=resolved_auth,
        )

        try:
            result = await self._invoke_mcp_tool(server_config, node, inputs)
        except Exception as exc:
            error_trace = self._write_error_trace(node, state, inputs, exc=exc, node_type='mcp')
            # CRITICAL: Update state with error trace BEFORE raising
            # This ensures the trace is persisted to checkpoints even when node fails
            state.update(error_trace)  # type: ignore[arg-type]  # WorkflowState is TypedDict with total=False, allows any keys
            raise ExecutionError(f"MCP tool '{node.tool}' failed: {exc}", trace_data=error_trace) from exc

        if node.expect_outputs:
            schema = self._type_schemas.get(node.id)
            if schema:
                validate_against_schema(schema, result, schema_id=node.id)

        output = self._prepare_output(node.id, result)
        self._attach_mcp_trace(output, node, state, inputs=inputs, result=result, resolved_auth=resolved_auth)
        return output

    def _resolve_mcp_auth(self, node: MCPNode, ctx: EvaluationContext) -> Optional[Dict[str, Any]]:
        """Resolve runtime auth expressions (headers / env) for an MCP node."""
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

    async def _invoke_mcp_tool(
        self,
        server_config: Any,
        node: MCPNode,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Invoke the MCP tool and normalise the result to a dict."""
        try:
            result = await self.services.mcp_client_registry.invoke_tool(
                server_config, node.tool, inputs
            )
        except ConnectionError as exc:
            raise ExecutionError(f"MCP connection failed for server '{node.server}': {exc}") from exc
        except Exception as exc:
            raise ExecutionError(
                f"MCP tool '{node.tool}' failed on server '{node.server}': {exc}"
            ) from exc

        # MCP tools return strings or content lists; downstream expects objects.
        if not isinstance(result, dict):
            result = {"result": result}
        return result

    def _attach_mcp_trace(
        self,
        output: Dict[str, Any],
        node: MCPNode,
        state: WorkflowState,
        *,
        inputs: Dict[str, Any],
        result: Any,
        resolved_auth: Optional[Dict[str, Any]],
    ) -> None:
        """Attach trace data to MCP output, redacting sensitive auth values."""
        trace_key = self._get_trace_key(node.id, state)

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
                "MCP node '%s' (server='%s', tool='%s') output keys: %s, trace_key present: %s",
                node.id,
                node.server,
                node.tool,
                list(output.keys()),
                trace_key in output,
                extra={"node_id": node.id, "output_keys": list(output.keys()), "trace_key": trace_key},
            )

    # pylint: disable=too-complex,too-many-locals
    # Reason: LLM node execution inherently complex with prompt construction, model invocation, and usage tracking
    def _run_llm(
        self,
        node: LLMNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        # STEP 1: Capture inputs
        inputs = self._capture_node_inputs(node, state, config, locals_ctx)

        # STEP 2: Execute LLM (existing logic)
        ctx = self._build_eval_context(state, config, locals_ctx)

        # Extract LLM configuration from inputs dict
        model = node.inputs.get("model")
        if not isinstance(model, str):
            raise ExecutionError(f"LLMNode {node.id}: 'model' must be a string in inputs")

        prompt_template = node.inputs.get("prompt")
        if not isinstance(prompt_template, str):
            raise ExecutionError(f"LLMNode {node.id}: 'prompt' must be a string in inputs")

        temperature = node.inputs.get("temperature")
        max_tokens = node.inputs.get("max_tokens")

        # All other keys are auxiliary data inputs
        reserved_keys = {"model", "prompt", "temperature", "max_tokens"}
        auxiliary = {
            key: evaluate_value(ctx, value)
            for key, value in node.inputs.items()
            if key not in reserved_keys
        }

        # Render prompt and lookup model
        prompt = render_template(ctx, prompt_template)
        model_def = self.services.model_registry.get(model)

        invocation = {
            "prompt": prompt,
            "inputs": auxiliary,
            "config": dict(config),
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "ui": node.ui,
        }

        usage_metadata = {}
        try:
            if node.outputs.mode == OutputMode.text:
                if model_def.text_handler is None:
                    raise ExecutionError(f"Model '{model}' does not support text responses")
                # Handler now returns tuple
                result, usage_metadata = model_def.text_handler(invocation)
                if not isinstance(result, str):
                    raise ExecutionError(f"LLM node '{node.id}' expected text response")
            elif node.outputs.mode == OutputMode.json:
                schema = self._type_schemas.get(node.id)
                if schema is None:
                    raise ExecutionError(f"No schema recorded for '{node.id}'")
                if model_def.json_handler is None:
                    raise ExecutionError(f"Model '{model}' does not support structured responses")
                # Handler now returns tuple
                result, usage_metadata = model_def.json_handler(invocation, schema)
                if not isinstance(result, dict):
                    raise ExecutionError(f"LLM node '{node.id}' expected JSON response")
            else:
                raise ExecutionError(f"Unsupported output mode '{node.outputs.mode}' for node '{node.id}'")
        except ExecutionError:
            # Re-raise ExecutionError without wrapping (includes validation errors)
            raise
        except Exception as exc:
            error_trace = self._write_error_trace(node, state, inputs, exc=exc, node_type='llm')
            # CRITICAL: Update state with error trace BEFORE raising
            # This ensures the trace is persisted to checkpoints even when node fails
            state.update(error_trace)  # type: ignore[arg-type]  # WorkflowState is TypedDict with total=False, allows any keys
            raise ExecutionError(f"LLM node '{node.id}' failed: {exc}", trace_data=error_trace) from exc

        # STEP 2.5: Track usage asynchronously (fire and forget)
        if usage_metadata:
            self._track_llm_usage_async(usage_metadata)

        # STEP 3: Prepare output
        output = self._prepare_output(node.id, result)

        # STEP 4: Store trace data
        # Use single underscore prefix to avoid LangGraph filtering double-underscore keys
        trace_key = self._get_trace_key(node.id, state)
        output[trace_key] = {
            'node_id': node.id,
            'node_type': 'llm',
            'inputs': inputs,  # Prompt template + evaluated input_refs
            'output': result,  # Raw LLM response
            'output_key': node.id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'succeeded',
            # Add usage metadata to trace
            'usage': {
                'model': usage_metadata.get('model', model),
                'input_tokens': usage_metadata.get('input_tokens', 0),
                'output_tokens': usage_metadata.get('output_tokens', 0),
                'reasoning_tokens': usage_metadata.get('reasoning_tokens', 0),
                'total_tokens': (
                    usage_metadata.get('input_tokens', 0) +
                    usage_metadata.get('output_tokens', 0) +
                    usage_metadata.get('reasoning_tokens', 0)
                ),
            },
        }

        # Diagnostic logging: Verify trace key is in output
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "LLM node '%s' output keys: %s, trace_key present: %s", node.id, list(output.keys()), trace_key in output,
                extra={"node_id": node.id, "output_keys": list(output.keys()), "trace_key": trace_key}
            )

        return output

    def _run_if(
        self,
        node: IfNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        """
        Evaluate the condition and store the result in state.

        Branch selection is handled by LangGraph conditional edges.
        The router reads _if_result_{node_id} to determine which branch to take.
        """
        ctx = self._build_eval_context(state, config, locals_ctx)
        condition_result = evaluate_condition(ctx, node.condition)

        # Store condition result for the router
        return {f"_if_result_{node.id}": condition_result}

    async def _run_if_async(
        self,
        node: IfNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        """
        Evaluate the condition and store the result in state (async version).

        Branch selection is handled by LangGraph conditional edges.
        """
        ctx = self._build_eval_context(state, config, locals_ctx)
        condition_result = evaluate_condition(ctx, node.condition)

        # Store condition result for the router
        return {f"_if_result_{node.id}": condition_result}

    def _run_for_each(
        self,
        node: ForEachNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        """
        Initialize or advance loop iteration state.

        On first call: Evaluate items and initialize loop state.
        On subsequent calls: Advance the index.

        Loop body execution is handled by LangGraph graph traversal.
        The router reads _loop_{node_id} to determine body vs exit.
        """
        loop_key = f"_loop_{node.id}"
        existing_loop_state = state.get(loop_key)

        if existing_loop_state is None:
            # First invocation - initialize loop state
            ctx = self._build_eval_context(state, config, locals_ctx)
            items_value = evaluate_value(ctx, node.items)
            if not isinstance(items_value, list):
                raise ExecutionError(f"for_each node '{node.id}' items expression must produce a list")

            loop_state = {
                "items": items_value,
                "current_index": 0,
                "has_more_iterations": len(items_value) > 0,
                "results": [],
            }
        else:
            # Subsequent invocation - advance to next iteration
            loop_state = dict(existing_loop_state)
            loop_state["current_index"] += 1
            loop_state["has_more_iterations"] = loop_state["current_index"] < len(loop_state["items"])

        # Build updates
        updates: Dict[str, Any] = {loop_key: loop_state}

        # Set current item and index in state for body nodes to access
        if loop_state["has_more_iterations"]:
            idx = loop_state["current_index"]
            updates[node.item_var] = loop_state["items"][idx]
            updates[node.index_var] = idx

        return updates

    async def _run_for_each_async(
        self,
        node: ForEachNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        """
        Initialize or advance loop iteration state (async version).

        On first call: Evaluate items and initialize loop state.
        On subsequent calls: Advance the index.

        Loop body execution is handled by LangGraph graph traversal.
        """
        loop_key = f"_loop_{node.id}"
        existing_loop_state = state.get(loop_key)

        if existing_loop_state is None:
            # First invocation - initialize loop state
            ctx = self._build_eval_context(state, config, locals_ctx)
            items_value = evaluate_value(ctx, node.items)
            if not isinstance(items_value, list):
                raise ExecutionError(f"for_each node '{node.id}' items expression must produce a list")

            loop_state = {
                "items": items_value,
                "current_index": 0,
                "has_more_iterations": len(items_value) > 0,
                "results": [],
            }
        else:
            # Subsequent invocation - advance to next iteration
            loop_state = dict(existing_loop_state)
            loop_state["current_index"] += 1
            loop_state["has_more_iterations"] = loop_state["current_index"] < len(loop_state["items"])

        # Build updates
        updates: Dict[str, Any] = {loop_key: loop_state}

        # Set current item and index in state for body nodes to access
        if loop_state["has_more_iterations"]:
            idx = loop_state["current_index"]
            updates[node.item_var] = loop_state["items"][idx]
            updates[node.index_var] = idx

        return updates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _execute_sequence(
        self,
        nodes: Sequence[Node],
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        sequence_state: WorkflowState = dict(state)
        accumulator: Dict[str, Any] = {}
        for child in nodes:
            updates = self._run_node(child, sequence_state, config, locals_ctx=locals_ctx, context=context)
            if updates:
                sequence_state.update(updates)
                accumulator.update(updates)
        return accumulator

    async def _execute_sequence_async(
        self,
        nodes: Sequence[Node],
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        sequence_state: WorkflowState = dict(state)
        accumulator: Dict[str, Any] = {}
        for child in nodes:
            updates = await self._run_node_async(
                child, sequence_state, config, locals_ctx=locals_ctx, context=context
            )
            if updates:
                sequence_state.update(updates)
                accumulator.update(updates)
        return accumulator

    def _build_eval_context(
        self,
        state: WorkflowState,
        config: Mapping[str, Any],
        locals_ctx: Mapping[str, Any] | None,
    ) -> EvaluationContext:
        visible_state = {k: v for k, v in state.items() if not k.startswith(INTERNAL_STATE_PREFIX)}
        locals_mapping = locals_ctx or {}
        return EvaluationContext(
            state=visible_state,
            locals=locals_mapping,
            config=config,
            trigger=self._current_trigger,
        )

    def _prepare_output(self, node_id: str, value: Any) -> Dict[str, Any]:
        """
        Prepare node output for state storage using node ID as the key.

        Args:
            node_id: The node's unique ID (used as state key)
            value: The output value to store

        Returns:
            Dictionary with node_id as key
        """
        if node_id.startswith(INTERNAL_STATE_PREFIX):
            raise ExecutionError(f"Node IDs starting with '{INTERNAL_STATE_PREFIX}' are reserved")
        schema = self._type_schemas.get(node_id)
        if schema is not None:
            validate_against_schema(schema, value, schema_id=node_id)
        return {node_id: value}

    # ------------------------------------------------------------------
    # Trace capture methods
    # ------------------------------------------------------------------
    def _evaluate_input_expressions(
        self, ctx: EvaluationContext, in_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate input expressions, capturing errors."""
        inputs = {}
        for key, expr in in_dict.items():
            try:
                inputs[key] = evaluate_value(ctx, expr)
            except Exception as e:
                inputs[key] = {"__error__": str(e), "__expression__": expr}
        return inputs

    def _capture_llm_node_inputs(
        self, node: LLMNode, ctx: EvaluationContext
    ) -> Dict[str, Any]:
        """Capture LLM node specific inputs."""
        inputs = {
            'prompt_template': node.inputs.get('prompt'),
            'model': node.inputs.get('model')
        }

        reserved_keys = {"model", "prompt", "temperature", "max_tokens"}
        auxiliary = {k: v for k, v in node.inputs.items() if k not in reserved_keys}
        if auxiliary:
            inputs['input_refs'] = self._evaluate_input_expressions(ctx, auxiliary)

        if 'temperature' in node.inputs:
            inputs['temperature'] = node.inputs['temperature']
        if 'max_tokens' in node.inputs:
            inputs['max_tokens'] = node.inputs['max_tokens']

        return inputs

    def _capture_node_inputs(
        self,
        node: Node,
        state: WorkflowState,
        config: Mapping[str, Any],
        locals_ctx: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Capture actual inputs used by node execution.
        Inputs are evaluated from state at runtime - cannot be predicted at compile time.
        """
        ctx = self._build_eval_context(state, config, locals_ctx)

        if isinstance(node, LLMNode):
            return self._capture_llm_node_inputs(node, ctx)

        if isinstance(node, (ToolNode, MCPNode)):
            return self._evaluate_input_expressions(ctx, node.inputs)

        return {}

    def _write_error_trace(
        self,
        node: Node,
        state: WorkflowState,
        inputs: Dict[str, Any],
        *,
        exc: Exception,
        node_type: str,
    ) -> Dict[str, Any]:
        """Write a partial trace with error info when node execution fails."""
        trace_key = self._get_trace_key(node.id, state)

        return {
            trace_key: {
                'node_id': node.id,
                'node_type': node_type,
                'inputs': inputs,
                'error': {
                    'type': exc.__class__.__name__,
                    'message': str(exc),
                },
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': 'failed',
            }
        }

    def _capture_node_output(
        self,
        node: Node,
        output_dict: Dict[str, Any],
    ) -> Any:
        """
        Extract raw output from node execution result.
        This is the actual result before any transformation.
        """
        if isinstance(node, ToolNode):
            # Output dict contains {node.id: result}
            # Extract the raw result
            if node.id in output_dict:
                return output_dict[node.id]
            # Fallback: return first value
            if output_dict:
                return next(iter(output_dict.values()))
            return None

        elif isinstance(node, LLMNode):
            # Similar - extract from output_dict
            if node.id in output_dict:
                return output_dict[node.id]
            if output_dict:
                return next(iter(output_dict.values()))
            return None

        # For other node types, return the output dict
        return output_dict

# pylint: disable=unused-argument,import-outside-toplevel
# Reason: Context params required by interface; imports done inside methods to avoid circular imports
"""
Runtime node executors – each workflow node type is compiled into a callable
that LangGraph can schedule.

With the registry-based dispatch, execution logic is now in individual node
type files (src/seer/core/nodes/*_node.py). This module provides the NodeRuntime
orchestrator that:
1. Builds LangGraph-compatible runners
2. Manages trigger and context binding
3. Dispatches to registered node types via the node_type_registry
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from langgraph._internal._runnable import RunnableCallable

from seer.core.errors import ExecutionError
from seer.core.expr.evaluator import EvaluationContext
from seer.core.expr.typecheck import TypeEnvironment
from seer.core.registry.mcp_client_registry import MCPClientRegistry
from seer.core.registry.model_registry import ModelRegistry
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.core.runtime.state import INTERNAL_STATE_PREFIX, WorkflowState
from seer.core.schema.models import Node
from seer.core.schema.schema_registry import SchemaRegistry
from seer.logger import get_logger
from seer.runtime_credit_limits import check_runtime_credit_limit

logger = get_logger(__name__)


@dataclass(frozen=True)
class RuntimeServices:
    schema_registry: SchemaRegistry
    tool_registry: ToolRegistry
    model_registry: ModelRegistry
    type_env: TypeEnvironment
    mcp_client_registry: Optional[MCPClientRegistry] = None

    # Auto-resolved connection IDs for nodes in single-account OAuth scenarios.
    # Maps node_id -> connection_id. Set during compilation when user has exactly
    # one connection for a provider and the node doesn't specify connection_id.
    resolved_connections: Dict[str, int] | None = None


class NodeRuntime:
    def __init__(self, services: RuntimeServices) -> None:
        self.services = services
        self._type_schemas = services.type_env.as_dict()
        self._current_trigger: Mapping[str, Any] | None = None
        self._current_context: WorkflowRuntimeContext | None = None
        self._current_vars: Mapping[str, Any] | None = None
        self._loop_body_map: Dict[str, str] = {}  # node_id -> parent_loop_id
        self._nested_loop_parents: Dict[str, str] = {}  # inner_loop_id -> outer_loop_id

    def build_runner(self, node: Node) -> RunnableCallable:
        async def runner_async(
            state: WorkflowState,
            config: Mapping[str, Any] | None = None,
            context: WorkflowRuntimeContext | None = None,
        ) -> Dict[str, Any]:
            return await self._run_node_async(node, state, config or {}, locals_ctx=None, context=context)

        return RunnableCallable(func=None, afunc=runner_async, name=f"node:{node.id}")

    def bind_trigger(self, trigger: Mapping[str, Any] | None) -> None:
        """Bind trigger event envelope for ${trigger.*} resolution."""
        self._current_trigger = dict(trigger) if trigger else None

    def bind_context(self, context: WorkflowRuntimeContext | None) -> None:
        self._current_context = context

    def bind_vars(self, vars_dict: Mapping[str, Any] | None) -> None:
        """Bind global variables for ${vars.*} resolution."""
        self._current_vars = vars_dict

    def set_loop_body_map(self, loop_body_map: Dict[str, str]) -> None:
        """Set mapping from node_id to parent loop_id for nodes inside loops."""
        self._loop_body_map = loop_body_map

    def set_nested_loop_parents(self, nested_loop_parents: Dict[str, str]) -> None:
        """Set mapping from inner_loop_id to outer_loop_id for nested loops.

        This mapping is used for:
        1. Resetting inner loop state when parent iteration changes
        2. Building full iteration paths for trace keys
        """
        self._nested_loop_parents = nested_loop_parents

    # -------------------------------------------------------------------------
    # BUG FIX: Multi-Level Trace Keys for Nested Loops (2024-02 RCA)
    # -------------------------------------------------------------------------
    # PROBLEM: Original trace key generation only added ONE iteration suffix
    #   from the immediate parent loop. For nested loops, this caused collisions:
    #     - outer_loop idx=0, inner_loop idx=0 → _trace_process_iter_0
    #     - outer_loop idx=1, inner_loop idx=0 → _trace_process_iter_0 (COLLISION!)
    #
    # SOLUTION: Build full iteration path from outermost to innermost loop:
    #     - outer_loop idx=0, inner_loop idx=0 → _trace_process_iter_0_iter_0
    #     - outer_loop idx=0, inner_loop idx=1 → _trace_process_iter_0_iter_1
    #     - outer_loop idx=1, inner_loop idx=0 → _trace_process_iter_1_iter_0
    #     - outer_loop idx=1, inner_loop idx=1 → _trace_process_iter_1_iter_1
    # -------------------------------------------------------------------------
    def _get_trace_key(self, node_id: str, state: WorkflowState) -> str:
        """
        Generate trace key for a node, including loop iteration path if inside nested loops.

        Returns:
            - `_trace_{node_id}` if not in a loop
            - `_trace_{node_id}_iter_{N}` if in a single loop
            - `_trace_{node_id}_iter_{N}_iter_{M}...` if in nested loops (outermost first)
        """
        # Check if this node is inside a loop
        parent_loop_id = self._loop_body_map.get(node_id)
        if not parent_loop_id:
            return f"_trace_{node_id}"

        # Build full iteration path from innermost to outermost loop
        # We traverse from innermost (parent_loop_id) up the nesting chain
        iteration_suffixes: list[str] = []
        current_loop_id = parent_loop_id

        while current_loop_id:
            loop_state_key = f"_loop_{current_loop_id}"
            loop_state = state.get(loop_state_key)
            if loop_state and isinstance(loop_state, dict):
                current_index = loop_state.get("current_index", 0)
                iteration_suffixes.append(f"_iter_{current_index}")

            # Walk up to the parent loop (if this loop is nested)
            current_loop_id = self._nested_loop_parents.get(current_loop_id)

        # Reverse to get outermost first (e.g., _iter_0_iter_1 for outer=0, inner=1)
        iteration_suffixes.reverse()

        return f"_trace_{node_id}{''.join(iteration_suffixes)}"

    async def _check_llm_credit_limit_async(self) -> None:
        """
        Run the credit gate in async contexts before an LLM call.
        """
        await check_runtime_credit_limit(self._current_context, logger)

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
            except Exception as e:  # pylint: disable=broad-exception-caught  # Defensive: log tracking errors, don't fail workflow
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
        except Exception as e:  # pylint: disable=broad-exception-caught  # Defensive: scheduling failure shouldn't break workflow
            logger.error("Failed to schedule LLM usage tracking: %s", e)

    # ------------------------------------------------------------------
    # Node handlers - Registry-based dispatch (no fallback)
    # ------------------------------------------------------------------
    async def _run_node_async(
        self,
        node: Node,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        """
        Execute a node using the registry-based dispatch.

        All node types must be registered in the node_type_registry.
        There is no fallback to isinstance checks - if a node type is not
        registered, it's an error.
        """
        from seer.core.nodes.registry import node_type_registry
        from seer.core.nodes.base import NodeExecutionContext

        node_impl = node_type_registry.get(node.type)
        if node_impl is None:
            raise ExecutionError(f"Unsupported node type '{node.type}' - not registered in node_type_registry")

        # Build execution context with all necessary data
        ctx = NodeExecutionContext(
            state=state,
            config=config,
            locals_ctx=locals_ctx,
            runtime_context=context or self._current_context,
            loop_body_map=self._loop_body_map,
            nested_loop_parents=self._nested_loop_parents,
            trigger=self._current_trigger,
            vars=self._current_vars,
        )

        start_time = time.perf_counter()
        success = True
        error_str: str | None = None
        try:
            result = await node_impl.execute_async(node, ctx, self.services)
            return result
        except Exception as exc:
            success = False
            error_str = str(exc)[:500]
            raise
        finally:
            user_email = getattr(ctx.runtime_context.user, "email", None) if (ctx.runtime_context and ctx.runtime_context.user) else None
            if user_email:
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                extra = node_impl.get_analytics_properties(node, ctx)
                # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import at module level
                from seer.analytics.workflow_tracking import capture_workflow_event
                await capture_workflow_event(
                    event="workflow_node_executed",
                    user_email=user_email,
                    properties={
                        "node_id": node.id,
                        "node_type": node.type,
                        "workflow_run_id": ctx.runtime_context.workflow_run_id,
                        "success": success,
                        "latency_ms": latency_ms,
                        "error": error_str,
                        **extra,
                    },
                )

    # ------------------------------------------------------------------
    # Helpers (kept for backward compatibility)
    # ------------------------------------------------------------------
    def _build_eval_context(
        self,
        state: WorkflowState,
        config: Mapping[str, Any],
        locals_ctx: Mapping[str, Any] | None,
    ) -> EvaluationContext:
        """Build evaluation context for expression evaluation."""
        visible_state = {k: v for k, v in state.items() if not k.startswith(INTERNAL_STATE_PREFIX)}
        locals_mapping = locals_ctx or {}
        return EvaluationContext(
            state=visible_state,
            locals=locals_mapping,
            config=config,
            trigger=self._current_trigger,
            vars=self._current_vars,
        )

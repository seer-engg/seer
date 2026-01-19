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
from typing import Any, Dict, List, Mapping, Sequence

from langgraph._internal._runnable import RunnableCallable

from seer.core.errors import ExecutionError
from seer.core.expr.evaluator import (
    EvaluationContext,
    evaluate_condition,
    evaluate_value,
    render_template,
)
from seer.core.expr.typecheck import TypeEnvironment
from seer.core.registry.model_registry import ModelRegistry
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.core.runtime.state import INTERNAL_STATE_PREFIX, WorkflowState
from seer.core.runtime.validate_output import validate_against_schema
from seer.core.schema.models import (
    ForEachNode,
    IfNode,
    LLMNode,
    Node,
    OutputMode,
    TaskKind,
    TaskNode,
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


class NodeRuntime:
    def __init__(self, services: RuntimeServices) -> None:
        self.services = services
        self._type_schemas = services.type_env.as_dict()
        self._current_trigger: Mapping[str, Any] | None = None
        self._current_context: WorkflowRuntimeContext | None = None

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

        from decimal import Decimal
        from seer.observability.credit_calculator import calculate_cost
        from seer.observability.tracking import track_llm_usage

        async def do_track():
            try:
                # Calculate cost
                cost = calculate_cost(
                    model=usage_metadata["model"],
                    input_tokens=usage_metadata["input_tokens"],
                    output_tokens=usage_metadata["output_tokens"],
                    reasoning_tokens=usage_metadata.get("reasoning_tokens", 0),
                )

                # Detect provider from model name
                model = usage_metadata["model"]
                if model.startswith(("gpt-", "o3-", "o1-")):
                    provider = "openai"
                elif model.startswith("claude-"):
                    provider = "anthropic"
                else:
                    provider = "unknown"

                # Track usage
                await track_llm_usage(
                    user=self._current_context.user,
                    provider=provider,
                    model=model,
                    input_tokens=usage_metadata["input_tokens"],
                    output_tokens=usage_metadata["output_tokens"],
                    cost=cost,
                    workflow_run_id=self._current_context.workflow_run_id,
                    operation="workflow_execution",
                    metadata={
                        "reasoning_tokens": usage_metadata.get("reasoning_tokens", 0),
                    },
                )

                logger.debug(
                    "Tracked LLM usage: model=%s, tokens=%d/%d, cost=$%.6f",
                    model,
                    usage_metadata["input_tokens"],
                    usage_metadata["output_tokens"],
                    cost,
                )

            except Exception as e:
                # Log error but don't fail workflow
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
            logger.error(f"Failed to schedule LLM usage tracking: {e}")

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
        if isinstance(node, TaskNode):
            return self._run_task(node, state, config, locals_ctx=locals_ctx)
        if isinstance(node, ToolNode):
            return self._run_tool(node, state, config, locals_ctx=locals_ctx, context=context)
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
        if isinstance(node, TaskNode):
            return self._run_task(node, state, config, locals_ctx=locals_ctx)
        if isinstance(node, LLMNode):
            await self._check_llm_credit_limit_async()
            return self._run_llm(node, state, config, locals_ctx=locals_ctx)
        if isinstance(node, IfNode):
            return await self._run_if_async(node, state, config, locals_ctx=locals_ctx, context=context)
        if isinstance(node, ForEachNode):
            return await self._run_for_each_async(node, state, config, locals_ctx=locals_ctx, context=context)
        raise ExecutionError(f"Unsupported node type '{node.type}'")

    def _run_task(
        self,
        node: TaskNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        ctx = self._build_eval_context(state, config, locals_ctx)
        # Evaluate inputs even if current TaskKind does not use them – future
        # kinds may rely on the resolved values and we want early validation.
        _ = {key: evaluate_value(ctx, value) for key, value in node.in_.items()}

        if node.kind == TaskKind.set:
            result = evaluate_value(ctx, node.value)
        else:
            raise ExecutionError(f"Unsupported task kind '{node.kind}'")

        return self._prepare_output(node.out, result)

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

        # STEP 3: Prepare output (existing logic)
        output = self._prepare_output(node.out, result)

        # STEP 4: Store trace data
        # Use single underscore prefix to avoid LangGraph filtering double-underscore keys
        trace_key = f"_trace_{node.id}"
        output[trace_key] = {
            'node_id': node.id,
            'node_type': 'tool',
            'inputs': inputs,  # Actual runtime inputs
            'output': result,  # Raw tool result (before prepare_output)
            'output_key': node.out,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

        # Diagnostic logging: Verify trace key is in output
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Tool node '{node.id}' output keys: {list(output.keys())}, trace_key present: {trace_key in output}",
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
        tool_def = self.services.tool_registry.get(node.tool)
        runtime_context = context or self._current_context
        handler = getattr(tool_def, "async_handler", None)
        if handler is None:
            result = await asyncio.to_thread(tool_def.handler, inputs, dict(config), runtime_context)
        else:
            result = await handler(inputs, dict(config), runtime_context)

        # STEP 3: Prepare output (existing logic)
        output = self._prepare_output(node.out, result)

        # STEP 4: Store trace data
        # Use single underscore prefix to avoid LangGraph filtering double-underscore keys
        trace_key = f"_trace_{node.id}"
        output[trace_key] = {
            'node_id': node.id,
            'node_type': 'tool',
            'inputs': inputs,  # Actual runtime inputs
            'output': result,  # Raw tool result (before prepare_output)
            'output_key': node.out,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

        # Diagnostic logging: Verify trace key is in output
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Tool node '{node.id}' (async) output keys: {list(output.keys())}, trace_key present: {trace_key in output}",
                extra={"node_id": node.id, "output_keys": list(output.keys()), "trace_key": trace_key}
            )

        return output

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
        prompt = render_template(ctx, node.prompt)
        # Evaluate auxiliary inputs (e.g. grounding snippets)
        auxiliary = {key: evaluate_value(ctx, value) for key, value in node.in_.items()}
        model_def = self.services.model_registry.get(node.model)

        invocation = {
            "prompt": prompt,
            "inputs": auxiliary,
            "config": dict(config),
            "parameters": {
                "temperature": node.temperature,
                "max_tokens": node.max_tokens,
            },
            "meta": node.meta,
        }

        usage_metadata = {}
        if node.output.mode == OutputMode.text:
            if model_def.text_handler is None:
                raise ExecutionError(f"Model '{node.model}' does not support text responses")
            # Handler now returns tuple
            result, usage_metadata = model_def.text_handler(invocation)
            if not isinstance(result, str):
                raise ExecutionError(f"LLM node '{node.id}' expected text response")
        elif node.output.mode == OutputMode.json:
            schema = self._type_schemas.get(node.out or "")
            if schema is None:
                raise ExecutionError(f"No schema recorded for '{node.out}'")
            if model_def.json_handler is None:
                raise ExecutionError(f"Model '{node.model}' does not support structured responses")
            # Handler now returns tuple
            result, usage_metadata = model_def.json_handler(invocation, schema)
            if not isinstance(result, dict):
                raise ExecutionError(f"LLM node '{node.id}' expected JSON response")
        else:
            raise ExecutionError(f"Unsupported output mode '{node.output.mode}' for node '{node.id}'")

        # STEP 2.5: Track usage asynchronously (fire and forget)
        if usage_metadata:
            self._track_llm_usage_async(usage_metadata)

        # STEP 3: Prepare output
        output = self._prepare_output(node.out, result)

        # STEP 4: Store trace data
        # Use single underscore prefix to avoid LangGraph filtering double-underscore keys
        trace_key = f"_trace_{node.id}"
        output[trace_key] = {
            'node_id': node.id,
            'node_type': 'llm',
            'inputs': inputs,  # Prompt template + evaluated input_refs
            'output': result,  # Raw LLM response
            'output_key': node.out,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            # Add usage metadata to trace
            'usage': {
                'model': usage_metadata.get('model', node.model),
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
                f"LLM node '{node.id}' output keys: {list(output.keys())}, trace_key present: {trace_key in output}",
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

    def _prepare_output(self, key: str | None, value: Any) -> Dict[str, Any]:
        if not key:
            return {}
        if key.startswith(INTERNAL_STATE_PREFIX):
            raise ExecutionError(f"State keys starting with '{INTERNAL_STATE_PREFIX}' are reserved")
        schema = self._type_schemas.get(key)
        if schema is not None:
            validate_against_schema(schema, value, schema_id=key)
        return {key: value}

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
        inputs = {'prompt_template': node.prompt, 'model': node.model}

        if node.in_:
            inputs['input_refs'] = self._evaluate_input_expressions(ctx, node.in_)

        if node.temperature is not None:
            inputs['temperature'] = node.temperature
        if node.max_tokens is not None:
            inputs['max_tokens'] = node.max_tokens

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

        if isinstance(node, (ToolNode, TaskNode)):
            return self._evaluate_input_expressions(ctx, node.in_)

        return {}

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
            # Output dict contains {node.out: result}
            # Extract the raw result
            if node.out and node.out in output_dict:
                return output_dict[node.out]
            # Fallback: return first value
            if output_dict:
                return next(iter(output_dict.values()))
            return None

        elif isinstance(node, LLMNode):
            # Similar - extract from output_dict
            if node.out and node.out in output_dict:
                return output_dict[node.out]
            if output_dict:
                return next(iter(output_dict.values()))
            return None

        # For other node types, return the output dict
        return output_dict

"""
Runtime node executors – each workflow node type is compiled into a callable
that LangGraph can schedule. Control flow nodes (if / for_each) execute their
children inline using the same dispatch logic, ensuring consistent semantics
between top-level and nested blocks.
"""

from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from langgraph._internal._runnable import RunnableCallable
from workflow_compiler.errors import ExecutionError
from workflow_compiler.expr.evaluator import EvaluationContext, evaluate_condition, evaluate_value, render_template
from workflow_compiler.expr.typecheck import TypeEnvironment
from workflow_compiler.registry.model_registry import ModelRegistry
from workflow_compiler.registry.tool_registry import ToolRegistry
from workflow_compiler.runtime.context import WorkflowRuntimeContext
from workflow_compiler.runtime.state import INTERNAL_STATE_PREFIX, WorkflowState
from workflow_compiler.runtime.validate_output import validate_against_schema
from workflow_compiler.schema.models import (
    ForEachNode,
    IfNode,
    LLMNode,
    Node,
    OutputMode,
    TaskKind,
    TaskNode,
    ToolNode,
)
from workflow_compiler.schema.schema_registry import SchemaRegistry

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
        self._current_inputs: Mapping[str, Any] = {}
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

    def bind_inputs(self, inputs: Mapping[str, Any]) -> None:
        self._current_inputs = dict(inputs)

    def bind_context(self, context: WorkflowRuntimeContext | None) -> None:
        self._current_context = context

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

        if node.output.mode == OutputMode.text:
            if model_def.text_handler is None:
                raise ExecutionError(f"Model '{node.model}' does not support text responses")
            result = model_def.text_handler(invocation)
            if not isinstance(result, str):
                raise ExecutionError(f"LLM node '{node.id}' expected text response")
        elif node.output.mode == OutputMode.json:
            schema = self._type_schemas.get(node.out or "")
            if schema is None:
                raise ExecutionError(f"No schema recorded for '{node.out}'")
            if model_def.json_handler is None:
                raise ExecutionError(f"Model '{node.model}' does not support structured responses")
            result = model_def.json_handler(invocation, schema)
            if not isinstance(result, dict):
                raise ExecutionError(f"LLM node '{node.id}' expected JSON response")
        else:
            raise ExecutionError(f"Unsupported output mode '{node.output.mode}' for node '{node.id}'")
        
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
        ctx = self._build_eval_context(state, config, locals_ctx)
        branch = node.then if evaluate_condition(ctx, node.condition) else node.else_
        return self._execute_sequence(branch, state, config, locals_ctx=locals_ctx, context=context)

    async def _run_if_async(
        self,
        node: IfNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        ctx = self._build_eval_context(state, config, locals_ctx)
        branch = node.then if evaluate_condition(ctx, node.condition) else node.else_
        return await self._execute_sequence_async(branch, state, config, locals_ctx=locals_ctx, context=context)

    def _run_for_each(
        self,
        node: ForEachNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        ctx = self._build_eval_context(state, config, locals_ctx)
        items_value = evaluate_value(ctx, node.items)
        if not isinstance(items_value, list):
            raise ExecutionError(f"for_each node '{node.id}' items expression must produce a list")

        combined_updates: Dict[str, Any] = {}
        loop_state: WorkflowState = dict(state)
        body_result_key = node.body[-1].out if node.body else None
        if node.out and not body_result_key:
            raise ExecutionError(
                f"for_each node '{node.id}' with out='{node.out}' requires the body to end with a node that writes to state"
            )
        aggregated: List[Any] = []

        for index, item in enumerate(items_value):
            iteration_locals = dict(locals_ctx or {})
            iteration_locals[node.item_var] = item
            iteration_locals[node.index_var] = index
            iteration_updates = self._execute_sequence(
                node.body, loop_state, config, locals_ctx=iteration_locals, context=context
            )
            if iteration_updates:
                loop_state.update(iteration_updates)
                combined_updates.update(iteration_updates)
            if node.out:
                if body_result_key not in loop_state:
                    raise ExecutionError(
                        f"for_each node '{node.id}' expected child '{body_result_key}' to produce output"
                    )
                aggregated.append(loop_state[body_result_key])

        if node.out:
            result = aggregated
            combined_updates.update(self._prepare_output(node.out, result))

        return combined_updates

    async def _run_for_each_async(
        self,
        node: ForEachNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
        context: WorkflowRuntimeContext | None,
    ) -> Dict[str, Any]:
        ctx = self._build_eval_context(state, config, locals_ctx)
        items_value = evaluate_value(ctx, node.items)
        if not isinstance(items_value, list):
            raise ExecutionError(f"for_each node '{node.id}' items expression must produce a list")

        combined_updates: Dict[str, Any] = {}
        loop_state: WorkflowState = dict(state)
        body_result_key = node.body[-1].out if node.body else None
        if node.out and not body_result_key:
            raise ExecutionError(
                f"for_each node '{node.id}' with out='{node.out}' requires the body to end with a node that writes to state"
            )
        aggregated: List[Any] = []

        for index, item in enumerate(items_value):
            iteration_locals = dict(locals_ctx or {})
            iteration_locals[node.item_var] = item
            iteration_locals[node.index_var] = index
            iteration_updates = await self._execute_sequence_async(
                node.body, loop_state, config, locals_ctx=iteration_locals, context=context
            )
            if iteration_updates:
                loop_state.update(iteration_updates)
                combined_updates.update(iteration_updates)
            if node.out:
                if body_result_key not in loop_state:
                    raise ExecutionError(
                        f"for_each node '{node.id}' expected child '{body_result_key}' to produce output"
                    )
                aggregated.append(loop_state[body_result_key])

        if node.out:
            result = aggregated
            combined_updates.update(self._prepare_output(node.out, result))

        return combined_updates

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
            inputs=self._current_inputs,
            locals=locals_mapping,
            config=config,
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
        
        if isinstance(node, ToolNode):
            # Evaluate node.in_ expressions against current state
            # This gives us the ACTUAL values passed to the tool
            inputs = {}
            for key, expr in node.in_.items():
                try:
                    inputs[key] = evaluate_value(ctx, expr)
                except Exception as e:
                    # If evaluation fails, store error info
                    inputs[key] = {"__error__": str(e), "__expression__": expr}
            return inputs
        
        elif isinstance(node, LLMNode):
            # For LLM, capture:
            # 1. Prompt template (from spec)
            # 2. Evaluated input_refs (actual values from state)
            inputs = {
                'prompt_template': node.prompt,  # Template string
            }
            if node.in_:
                evaluated_refs = {}
                for key, expr in node.in_.items():
                    try:
                        evaluated_refs[key] = evaluate_value(ctx, expr)
                    except Exception as e:
                        evaluated_refs[key] = {"__error__": str(e), "__expression__": expr}
                inputs['input_refs'] = evaluated_refs
            
            # Also capture model config
            inputs['model'] = node.model
            if node.temperature is not None:
                inputs['temperature'] = node.temperature
            if node.max_tokens is not None:
                inputs['max_tokens'] = node.max_tokens
            
            return inputs
        
        elif isinstance(node, TaskNode):
            # Evaluate node.in_ expressions
            inputs = {}
            for key, expr in node.in_.items():
                try:
                    inputs[key] = evaluate_value(ctx, expr)
                except Exception as e:
                    inputs[key] = {"__error__": str(e), "__expression__": expr}
            return inputs
        
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



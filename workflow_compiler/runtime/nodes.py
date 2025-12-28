"""
Runtime node executors – each workflow node type is compiled into a callable
that LangGraph can schedule. Control flow nodes (if / for_each) execute their
children inline using the same dispatch logic, ensuring consistent semantics
between top-level and nested blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from workflow_compiler.errors import ExecutionError
from workflow_compiler.expr.evaluator import EvaluationContext, evaluate_condition, evaluate_value, render_template
from workflow_compiler.expr.typecheck import TypeEnvironment
from workflow_compiler.registry.model_registry import ModelRegistry
from workflow_compiler.registry.tool_registry import ToolRegistry
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


RuntimeFn = Callable[[WorkflowState, Mapping[str, Any] | None], Dict[str, Any]]


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

    def build_runner(self, node: Node) -> RuntimeFn:
        def runner(state: WorkflowState, config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
            return self._run_node(node, state, config or {}, locals_ctx=None)

        return runner

    def bind_inputs(self, inputs: Mapping[str, Any]) -> None:
        self._current_inputs = dict(inputs)

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
    ) -> Dict[str, Any]:
        if isinstance(node, TaskNode):
            return self._run_task(node, state, config, locals_ctx=locals_ctx)
        if isinstance(node, ToolNode):
            return self._run_tool(node, state, config, locals_ctx=locals_ctx)
        if isinstance(node, LLMNode):
            return self._run_llm(node, state, config, locals_ctx=locals_ctx)
        if isinstance(node, IfNode):
            return self._run_if(node, state, config, locals_ctx=locals_ctx)
        if isinstance(node, ForEachNode):
            return self._run_for_each(node, state, config, locals_ctx=locals_ctx)
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
    ) -> Dict[str, Any]:
        ctx = self._build_eval_context(state, config, locals_ctx)
        inputs = {key: evaluate_value(ctx, value) for key, value in node.in_.items()}
        tool_def = self.services.tool_registry.get(node.tool)
        result = tool_def.handler(inputs, dict(config))
        return self._prepare_output(node.out, result)

    def _run_llm(
        self,
        node: LLMNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
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
            return self._prepare_output(node.out, result)

        if node.output.mode == OutputMode.json:
            schema = self._type_schemas.get(node.out or "")
            if schema is None:
                raise ExecutionError(f"No schema recorded for '{node.out}'")
            if model_def.json_handler is None:
                raise ExecutionError(f"Model '{node.model}' does not support structured responses")
            result = model_def.json_handler(invocation, schema)
            if not isinstance(result, dict):
                raise ExecutionError(f"LLM node '{node.id}' expected JSON response")
            return self._prepare_output(node.out, result)

        raise ExecutionError(f"Unsupported output mode '{node.output.mode}' for node '{node.id}'")

    def _run_if(
        self,
        node: IfNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        ctx = self._build_eval_context(state, config, locals_ctx)
        branch = node.then if evaluate_condition(ctx, node.condition) else node.else_
        return self._execute_sequence(branch, state, config, locals_ctx=locals_ctx)

    def _run_for_each(
        self,
        node: ForEachNode,
        state: WorkflowState,
        config: Mapping[str, Any],
        *,
        locals_ctx: Mapping[str, Any] | None,
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
                node.body, loop_state, config, locals_ctx=iteration_locals
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
    ) -> Dict[str, Any]:
        sequence_state: WorkflowState = dict(state)
        accumulator: Dict[str, Any] = {}
        for child in nodes:
            updates = self._run_node(child, sequence_state, config, locals_ctx=locals_ctx)
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
            validate_against_schema(schema, value)
        return {key: value}



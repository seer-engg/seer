"""
Helpers for invoking a compiled workflow graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from workflow_compiler.runtime.context import WorkflowRuntimeContext
from workflow_compiler.runtime.nodes import NodeRuntime
from workflow_compiler.runtime.state import INTERNAL_STATE_PREFIX
from workflow_compiler.schema.models import JsonSchema, WorkflowSpec
import mlflow

mlflow.langchain.autolog()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompiledWorkflow:
    spec: WorkflowSpec
    type_env: Mapping[str, JsonSchema]
    graph: Any
    runtime: NodeRuntime

    def invoke(
        self,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        context: WorkflowRuntimeContext | None = None,
    ) -> Mapping[str, Any]:
        self.runtime.bind_inputs(inputs)
        self.runtime.bind_context(context)
        effective_config = dict(config or {})
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "CompiledWorkflow.invoke graph config_keys=%s context_present=%s",
                sorted(effective_config.keys()),
                context is not None,
            )
        invoke_kwargs = {"config": effective_config}
        if context is not None:
            invoke_kwargs["context"] = context
        final_state = self.graph.invoke({}, **invoke_kwargs)
        return {
            key: value
            for key, value in final_state.items()
            if not key.startswith(INTERNAL_STATE_PREFIX)
        }

    async def ainvoke(
        self,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        context: WorkflowRuntimeContext | None = None,
    ) -> Mapping[str, Any]:
        self.runtime.bind_inputs(inputs)
        self.runtime.bind_context(context)
        effective_config = dict(config or {})
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "CompiledWorkflow.ainvoke graph config_keys=%s context_present=%s",
                sorted(effective_config.keys()),
                context is not None,
            )
        invoke_kwargs = {"config": effective_config}
        if context is not None:
            invoke_kwargs["context"] = context
        final_state = await self.graph.ainvoke({}, **invoke_kwargs)
        return {
            key: value
            for key, value in final_state.items()
            if not key.startswith(INTERNAL_STATE_PREFIX)
        }



"""
Helpers for invoking a compiled workflow graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from seer.core.runtime.context import WorkflowRuntimeContext
from seer.core.runtime.nodes import NodeRuntime
from seer.core.runtime.state import INTERNAL_STATE_PREFIX
from seer.core.schema.models import JsonSchema, WorkflowSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompiledWorkflow:
    spec: WorkflowSpec
    type_env: Mapping[str, JsonSchema]
    graph: Any
    runtime: NodeRuntime

    def invoke(
        self,
        config: Mapping[str, Any] | None = None,
        context: WorkflowRuntimeContext | None = None,
        trigger: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.runtime.bind_trigger(trigger)
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
        config: Mapping[str, Any] | None = None,
        context: WorkflowRuntimeContext | None = None,
        trigger: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.runtime.bind_trigger(trigger)
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

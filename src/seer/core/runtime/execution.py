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
from seer.utilities.langfuse_tracing import merge_workflow_langfuse_callbacks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompiledWorkflow:
    spec: WorkflowSpec
    type_env: Mapping[str, JsonSchema]
    graph: Any
    runtime: NodeRuntime

    async def ainvoke(
        self,
        workflow_input: Any = None,
        config: Mapping[str, Any] | None = None,
        context: WorkflowRuntimeContext | None = None,
        trigger: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.runtime.bind_trigger(trigger)
        self.runtime.bind_context(context)
        effective_config = dict(config or {})
        effective_config = merge_workflow_langfuse_callbacks(effective_config)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "CompiledWorkflow.ainvoke graph config_keys=%s context_present=%s",
                sorted(effective_config.keys()),
                context is not None,
            )
        invoke_kwargs = {"config": effective_config}
        if context is not None:
            invoke_kwargs["context"] = context
        # Use provided input (e.g., Command for resume) or empty dict for fresh start
        graph_input = workflow_input if workflow_input is not None else {}

        # Wrap graph invocation with Langfuse user context for trace attribution
        # pylint: disable=import-outside-toplevel  # Reason: lazy loading to match module pattern
        from seer.utilities.langfuse_tracing import langfuse_user_context
        user_id = context.user.user_id if context and context.user else None
        with langfuse_user_context(user_id):
            final_state = await self.graph.ainvoke(graph_input, **invoke_kwargs)

        return {
            key: value
            for key, value in final_state.items()
            if not key.startswith(INTERNAL_STATE_PREFIX) or key == "__interrupt__"
        }

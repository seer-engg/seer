"""
Helpers for invoking a compiled workflow graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from workflow_compiler.runtime.nodes import NodeRuntime
from workflow_compiler.runtime.state import INTERNAL_STATE_PREFIX
from workflow_compiler.schema.models import JsonSchema, WorkflowSpec
import mlflow
mlflow.langchain.autolog()


@dataclass(frozen=True)
class CompiledWorkflow:
    spec: WorkflowSpec
    type_env: Mapping[str, JsonSchema]
    graph: Any
    runtime: NodeRuntime

    def invoke(self, inputs: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        self.runtime.bind_inputs(inputs)
        final_state = self.graph.invoke({}, config or {})
        return {
            key: value
            for key, value in final_state.items()
            if not key.startswith(INTERNAL_STATE_PREFIX)
        }



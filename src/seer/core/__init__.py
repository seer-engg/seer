"""
Public entrypoint for compiling workflow specs into runnable LangGraph graphs.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from seer.core.compiler.context import CompilerContext
from seer.core.compiler.emit_langgraph import emit_langgraph
from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.type_env import build_type_environment
from seer.core.compiler.validate_refs import validate_references
from seer.core.runtime.execution import CompiledWorkflow
from seer.core.runtime.nodes import NodeRuntime, RuntimeServices


async def compile_workflow(
    payload: Any,
    context: CompilerContext,
    *,
    checkpointer: Optional[AsyncPostgresSaver] = None,
) -> CompiledWorkflow:
    """
    Compile a workflow specification into a runnable LangGraph workflow.
    """

    spec = parse_workflow_spec(payload)
    type_env = build_type_environment(
        spec,
        schema_registry=context.schema_registry,
        tool_registry=context.tool_registry,
    )
    validate_references(spec, type_env)
    plan = build_execution_plan(spec)
    runtime = NodeRuntime(
        RuntimeServices(
            schema_registry=context.schema_registry,
            tool_registry=context.tool_registry,
            model_registry=context.model_registry,
            type_env=type_env,
        )
    )
    graph = await emit_langgraph(plan, runtime, checkpointer=checkpointer)
    return CompiledWorkflow(
        spec=spec,
        type_env=type_env.as_dict(),
        graph=graph,
        runtime=runtime,
    )

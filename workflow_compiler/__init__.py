"""
Public entrypoint for compiling workflow specs into runnable LangGraph graphs.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from workflow_compiler.compiler.context import CompilerContext
from workflow_compiler.compiler.emit_langgraph import emit_langgraph
from workflow_compiler.compiler.lower_control_flow import build_execution_plan
from workflow_compiler.compiler.parse import parse_workflow_spec
from workflow_compiler.compiler.type_env import build_type_environment
from workflow_compiler.compiler.validate_refs import validate_references
from workflow_compiler.runtime.execution import CompiledWorkflow
from workflow_compiler.runtime.nodes import NodeRuntime, RuntimeServices


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



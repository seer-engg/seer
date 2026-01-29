from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from seer.core.compiler.context import CompilerContext
from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.type_env import build_type_environment_async
from seer.core.compiler.validate_refs import validate_references
from seer.core.schema.models import WorkflowSpec

if TYPE_CHECKING:
    from seer.core.runtime.execution import CompiledWorkflow


async def compile_parsed_workflow(
    spec: WorkflowSpec,
    context: CompilerContext,
    *,
    checkpointer: Optional[AsyncPostgresSaver] = None,
) -> "CompiledWorkflow":
    """
    Shared compilation pipeline for already-validated WorkflowSpec objects.

    This builds the type environment, validates references, lowers the control
    flow to an execution plan, and emits a LangGraph graph with a runtime.
    """
    type_env = await build_type_environment_async(
        spec,
        schema_registry=context.schema_registry,
        tool_registry=context.tool_registry,
        mcp_client_registry=context.mcp_client_registry,
    )
    validate_references(spec, type_env)
    plan = build_execution_plan(spec)

    # Import locally to avoid circular dependency with runtime package initialization.
    from seer.core.runtime.nodes import NodeRuntime, RuntimeServices  # pylint: disable=import-outside-toplevel
    from seer.core.runtime.execution import CompiledWorkflow  # pylint: disable=import-outside-toplevel
    from seer.core.compiler.emit_langgraph import emit_langgraph  # pylint: disable=import-outside-toplevel

    runtime = NodeRuntime(
        RuntimeServices(
            schema_registry=context.schema_registry,
            tool_registry=context.tool_registry,
            model_registry=context.model_registry,
            type_env=type_env,
            mcp_client_registry=context.mcp_client_registry,
        )
    )

    graph = await emit_langgraph(plan, runtime, checkpointer=checkpointer)
    return CompiledWorkflow(
        spec=spec,
        type_env=type_env.as_dict(),
        graph=graph,
        runtime=runtime,
    )

"""
Public entrypoint for compiling workflow specs into runnable LangGraph graphs.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from seer.core.compiler.compile_pipeline import compile_parsed_workflow
from seer.core.compiler.context import CompilerContext
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.runtime.execution import CompiledWorkflow


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
    return await compile_parsed_workflow(spec, context, checkpointer=checkpointer)

"""
Internal create_artifact tool factory for AgentNode.

This tool is NOT registered in the public tool registry. It is injected
directly into the agent's tool list when enable_artifacts is True.

The agent calls create_artifact(html_content, filename, format) to convert
HTML to PDF or DOCX and store the result via WorkflowFileSystem.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from seer.core.errors import ExecutionError
from seer.core.nodes.artifacts.converters import FORMAT_MIME, html_to_docx, html_to_pdf

if TYPE_CHECKING:
    from seer.core.nodes.base import NodeExecutionContext

logger = logging.getLogger(__name__)

ARTIFACT_TOOL_NAME = "create_artifact"


class _CreateArtifactInput(BaseModel):
    """Input schema for the create_artifact tool."""

    html_content: str = Field(description="Full HTML document to convert into the output file")
    filename: str = Field(description="Desired output filename, e.g. 'report.pdf' or 'summary.docx'")
    format: str = Field(description="Output format: 'pdf' or 'docx'")


def make_create_artifact_tool(ctx: "NodeExecutionContext", node_id: str) -> StructuredTool:
    """
    Build a StructuredTool that converts HTML and stores the result via WorkflowFileSystem.

    The tool is bound to the given execution context so it can resolve the
    user and run ID needed for file storage.

    Args:
        ctx: The node execution context for this agent invocation.
        node_id: The ID of the agent node (used as source_node_id in file records).

    Returns:
        StructuredTool ready to be appended to the agent's tool list.
    """

    async def _execute(html_content: str, filename: str, format: str) -> str:  # pylint: disable=redefined-builtin  # Reason: 'format' matches the public tool schema name
        """Convert HTML to the requested format and store as a workflow file."""
        fmt = format.lower().strip()
        if fmt not in FORMAT_MIME:
            raise ExecutionError(f"create_artifact: unsupported format '{format}'. Use 'pdf' or 'docx'.")

        # Convert HTML → bytes
        try:
            if fmt == "pdf":
                file_bytes = html_to_pdf(html_content)
            else:
                file_bytes = html_to_docx(html_content)
        except Exception as e:
            raise ExecutionError(f"create_artifact: conversion to {fmt} failed: {e}") from e

        # Store via WorkflowFileSystem
        if not ctx.runtime_context or not ctx.runtime_context.has_file_system:
            raise ExecutionError("Artifact storage not configured")

        if not ctx.runtime_context.user:
            raise ExecutionError("create_artifact: no user context for file storage")

        run_id = ctx.runtime_context.workflow_run_id or ""
        mime_type = FORMAT_MIME[fmt]

        file_ref = await ctx.runtime_context.file_system.store_file_with_record(
            user=ctx.runtime_context.user,
            run_id=run_id,
            filename=filename,
            data=file_bytes,
            mime_type=mime_type,
            source_tool=ARTIFACT_TOOL_NAME,
            source_node_id=node_id,
            organization_id=ctx.runtime_context.organization_id,
        )

        logger.debug(
            "create_artifact: stored %s (%d bytes) for node %s",
            filename, len(file_bytes), node_id,
        )

        return json.dumps(file_ref.to_dict(), default=str)

    return StructuredTool.from_function(
        coroutine=_execute,
        name=ARTIFACT_TOOL_NAME,
        description=(
            "Convert HTML content to a file artifact (PDF or DOCX) and store it. "
            "Call this when you need to produce a downloadable document from HTML."
        ),
        args_schema=_CreateArtifactInput,
    )

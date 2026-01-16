"""Workflow execution tools for MCP server."""

import json
from typing import Any, Dict, List, Optional

from mcp.types import TextContent, Tool  # pylint: disable=no-name-in-module # Reason: external MCP SDK

from api.workflows import models as api_models
from api.workflows.services.execution import (
    run_saved_workflow,
    get_run_status,
    get_run_history,
)
from api.workflows.services.history import list_workflow_runs
from shared.database import User


async def seer_run_workflow(
    user: User,
    workflow_id: str,
    inputs: Optional[Dict[str, Any]] = None,
    test_mode: bool = False,
) -> List[TextContent]:
    """Run a workflow.

    Args:
        user: Authenticated user
        workflow_id: Workflow ID (e.g., "wf_123")
        inputs: Optional input data for the workflow
        test_mode: Whether to run in test mode (default: False)

    Returns:
        List of text content with execution result
    """
    payload = api_models.RunFromWorkflowRequest(
        inputs=inputs or {},
        test_mode=test_mode,
    )

    result = await run_saved_workflow(user, workflow_id, payload)

    return [
        TextContent(
            type="text",
            text=f"""Started workflow execution!

Run ID: {result.run_id}
Workflow ID: {result.workflow_id}
Status: {result.status}
Created At: {result.created_at}

Use seer_get_execution to check the status and retrieve results.""",
        )
    ]


async def seer_get_execution(
    user: User,
    run_id: str,
) -> List[TextContent]:
    """Get execution details and results.

    Args:
        user: Authenticated user
        run_id: Run ID (e.g., "run_123")

    Returns:
        List of text content with execution details
    """
    result = await get_run_status(user, run_id)

    status_info = f"""Run ID: {result.run_id}
Status: {result.status}
Workflow ID: {result.workflow_id}
Created At: {result.created_at}"""

    if result.started_at:
        status_info += f"\nStarted At: {result.started_at}"
    if result.finished_at:
        status_info += f"\nFinished At: {result.finished_at}"

    error_info = ""
    if result.last_error:
        error_info = f"\n\nError:\n{result.last_error}"

    return [
        TextContent(
            type="text",
            text=f"""Execution Details:

{status_info}{error_info}

Use seer_get_execution_history to see the full execution trace.""",
        )
    ]


async def seer_get_execution_history(
    user: User,
    run_id: str,
) -> List[TextContent]:
    """Get detailed execution history with step-by-step results.

    Args:
        user: Authenticated user
        run_id: Run ID (e.g., "run_123")

    Returns:
        List of text content with execution history
    """
    result = await get_run_history(user, run_id)

    # Format the history for readable output
    summary = result.summary
    history_json = json.dumps(summary.model_dump() if hasattr(summary, 'model_dump') else summary, indent=2)

    return [
        TextContent(
            type="text",
            text=f"""Execution History for Run {run_id}:

Status: {result.status}
Workflow ID: {result.workflow_id}

Detailed History:
{history_json}""",
        )
    ]


async def seer_list_executions(
    user: User,
    workflow_id: str,
    limit: int = 20,
) -> List[TextContent]:
    """List workflow executions.

    Args:
        user: Authenticated user
        workflow_id: Workflow ID (e.g., "wf_123")
        limit: Maximum number of executions to return (default: 20, max: 100)

    Returns:
        List of text content with executions
    """
    if limit < 1 or limit > 100:
        limit = min(max(limit, 1), 100)

    result = await list_workflow_runs(user, workflow_id, limit=limit)

    if not result.runs:
        return [TextContent(type="text", text=f"No executions found for workflow {workflow_id}.")]

    runs_text = "\n\n".join(
        [
            f"""Run ID: {r.run_id}
Status: {r.status}
Created: {r.created_at}
Started: {r.started_at or "Not started"}
Finished: {r.finished_at or "Not finished"}
Error: {r.error or "None"}"""
            for r in result.runs
        ]
    )

    return [
        TextContent(
            type="text",
            text=f"""Found {len(result.runs)} execution(s) for workflow {workflow_id}:

{runs_text}""",
        )
    ]


def get_execution_tools() -> List[Tool]:
    """Get execution tool definitions for MCP."""
    return [
        Tool(
            name="seer_run_workflow",
            description="Run a workflow in Seer",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID (e.g., 'wf_123')",
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Optional input data for the workflow",
                    },
                    "test_mode": {
                        "type": "boolean",
                        "description": "Whether to run in test mode (default: false)",
                        "default": False,
                    },
                },
                "required": ["workflow_id"],
            },
        ),
        Tool(
            name="seer_get_execution",
            description="Get execution status and details",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID (e.g., 'run_123')",
                    },
                },
                "required": ["run_id"],
            },
        ),
        Tool(
            name="seer_get_execution_history",
            description="Get detailed execution history with step-by-step trace",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID (e.g., 'run_123')",
                    },
                },
                "required": ["run_id"],
            },
        ),
        Tool(
            name="seer_list_executions",
            description="List all executions for a specific workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID (e.g., 'wf_123')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of executions to return (1-100, default: 20)",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": ["workflow_id"],
            },
        ),
    ]

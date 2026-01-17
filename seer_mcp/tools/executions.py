"""Workflow execution tools for MCP server."""

import json
from typing import Any, Dict, List, Optional

from mcp.types import TextContent, Tool  # pylint: disable=no-name-in-module # Reason: external MCP SDK

from api.workflows import models as api_models
from api.workflows.services.execution import (
    run_saved_workflow,
    list_workflow_runs,
)
from api.workflows.services.history import (
    get_run_status,
    get_run_history,
)
from seer_mcp.config import get_config, MCPMode
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
    config = get_config()

    if config.mode == MCPMode.LOCAL:
        # LOCAL mode: Direct service call
        payload = api_models.RunFromWorkflowRequest(
            inputs=inputs or {},
            test_mode=test_mode,
        )
        result = await run_saved_workflow(user, workflow_id, payload)
        result_dict = {
            "run_id": result.run_id,
            "workflow_id": result.workflow_id,
            "status": result.status,
            "created_at": str(result.created_at),
        }
    else:
        # CLOUD mode: API client call
        from seer_mcp.server import get_api_client  # pylint: disable=import-outside-toplevel,cyclic-import # Reason: Avoid circular import

        client = get_api_client()
        if not client:
            raise ValueError("API client not initialized")
        result_dict = await client.run_workflow(workflow_id, inputs, test_mode)

    return [
        TextContent(
            type="text",
            text=f"""Started workflow execution!

Run ID: {result_dict['run_id']}
Workflow ID: {result_dict['workflow_id']}
Status: {result_dict['status']}
Created At: {result_dict['created_at']}

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
    config = get_config()

    if config.mode == MCPMode.LOCAL:
        # LOCAL mode: Direct service call
        result = await get_run_status(user, run_id)
        result_dict = {
            "run_id": result.run_id,
            "status": result.status,
            "workflow_id": result.workflow_id,
            "created_at": str(result.created_at),
            "started_at": str(result.started_at) if result.started_at else None,
            "finished_at": str(result.finished_at) if result.finished_at else None,
            "last_error": result.last_error,
        }
    else:
        # CLOUD mode: API client call
        from seer_mcp.server import get_api_client  # pylint: disable=import-outside-toplevel,cyclic-import # Reason: Avoid circular import

        client = get_api_client()
        if not client:
            raise ValueError("API client not initialized")
        result_dict = await client.get_execution(run_id)

    status_info = f"""Run ID: {result_dict['run_id']}
Status: {result_dict['status']}
Workflow ID: {result_dict['workflow_id']}
Created At: {result_dict['created_at']}"""

    if result_dict.get("started_at"):
        status_info += f"\nStarted At: {result_dict['started_at']}"
    if result_dict.get("finished_at"):
        status_info += f"\nFinished At: {result_dict['finished_at']}"

    error_info = ""
    if result_dict.get("last_error"):
        error_info = f"\n\nError:\n{result_dict['last_error']}"

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
    config = get_config()

    if config.mode == MCPMode.LOCAL:
        # LOCAL mode: Direct service call
        result = await get_run_history(user, run_id)
        summary = result.summary
        history_data = summary.model_dump() if hasattr(summary, 'model_dump') else summary
        result_dict = {
            "status": result.status,
            "workflow_id": result.workflow_id,
            "summary": history_data,
        }
    else:
        # CLOUD mode: API client call
        from seer_mcp.server import get_api_client  # pylint: disable=import-outside-toplevel,cyclic-import # Reason: Avoid circular import

        client = get_api_client()
        if not client:
            raise ValueError("API client not initialized")
        result_dict = await client.get_execution_history(run_id)

    history_json = json.dumps(result_dict.get("summary", {}), indent=2)

    return [
        TextContent(
            type="text",
            text=f"""Execution History for Run {run_id}:

Status: {result_dict['status']}
Workflow ID: {result_dict['workflow_id']}

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
    config = get_config()

    if limit < 1 or limit > 100:
        limit = min(max(limit, 1), 100)

    if config.mode == MCPMode.LOCAL:
        # LOCAL mode: Direct service call
        result = await list_workflow_runs(user, workflow_id, limit=limit)
        runs = result.runs
    else:
        # CLOUD mode: API client call
        from seer_mcp.server import get_api_client  # pylint: disable=import-outside-toplevel,cyclic-import # Reason: Avoid circular import

        client = get_api_client()
        if not client:
            raise ValueError("API client not initialized")
        result_dict = await client.list_executions(workflow_id, limit=limit)
        runs = result_dict.get("items", [])

    if not runs:
        return [TextContent(type="text", text=f"No executions found for workflow {workflow_id}.")]

    runs_text = "\n\n".join(
        [
            f"""Run ID: {r.get('run_id') if isinstance(r, dict) else r.run_id}
Status: {r.get('status') if isinstance(r, dict) else r.status}
Created: {r.get('created_at') if isinstance(r, dict) else r.created_at}
Started: {(r.get('started_at') if isinstance(r, dict) else r.started_at) or "Not started"}
Finished: {(r.get('finished_at') if isinstance(r, dict) else r.finished_at) or "Not finished"}
Error: {(r.get('error') if isinstance(r, dict) else r.error) or "None"}"""
            for r in runs
        ]
    )

    return [
        TextContent(
            type="text",
            text=f"""Found {len(runs)} execution(s) for workflow {workflow_id}:

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

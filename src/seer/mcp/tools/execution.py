"""
MCP execution tools for running workflows and checking status.

Delegates to existing workflow execution services.
"""
# pylint: disable=cyclic-import # Reason: mcp server module registers tools via imports

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from seer.mcp.server import mcp
from seer.mcp.tools.workflows import _get_mcp_user, _ensure_db
from seer.mcp.tracking import track_mcp_tool
from seer.logger import get_logger

logger = get_logger(__name__)


@mcp.tool()
@track_mcp_tool("run_workflow")
async def run_workflow(
    workflow_id: str,
    inputs: Optional[Dict[str, Any]] = None,
    version: Optional[int] = None,
) -> str:
    """
    Execute a workflow and return the run ID.

    If the workflow has triggers, creates one run per trigger with sample data.
    Otherwise, creates a single manual run.

    Args:
        workflow_id: The workflow ID to execute
        inputs: Optional input variables for the workflow
        version: Optional specific version to run (default: draft or latest published)

    Returns:
        JSON with run_id(s), status, and workflow metadata
    """
    try:
        await _ensure_db()
        user = await _get_mcp_user()

        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest

        request = RunFromWorkflowRequest(
            version=version,
            inputs=inputs or {},
            config={},
        )

        response = await run_saved_workflow(user, workflow_id, request)

        # Single run response
        return json.dumps({
            "run_id": response.run_id,
            "status": response.status,
            "workflow_id": response.workflow_id,
            "created_at": response.created_at.isoformat(),
            "started_at": response.started_at.isoformat() if response.started_at else None,
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error running workflow: %s", e)
        return json.dumps({
            "error": "execution_failed",
            "message": str(e)
        })


@mcp.tool()
@track_mcp_tool("get_run_status")
async def get_run_status(
    workflow_id: str,  # pylint: disable=unused-argument # Reason: Required for MCP tool API signature consistency
    run_id: str,
) -> str:
    """
    Get the current status of a workflow run.

    Args:
        workflow_id: The workflow ID
        run_id: The run ID to check

    Returns:
        JSON with run status, timestamps, and any errors
    """
    try:
        await _ensure_db()
        user = await _get_mcp_user()

        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.history import get_run_status as get_run_status_service

        response = await get_run_status_service(user, run_id)

        result = {
            "run_id": response.run_id,
            "status": response.status,
            "workflow_id": response.workflow_id,
            "created_at": response.created_at.isoformat(),
        }

        if response.started_at:
            result["started_at"] = response.started_at.isoformat()
        if response.finished_at:
            result["finished_at"] = response.finished_at.isoformat()
        if response.progress:
            result["progress"] = {
                "completed": response.progress.completed,
                "total": response.progress.total,
            }
        if response.current_node_id:
            result["current_node_id"] = response.current_node_id
        if response.last_error:
            result["last_error"] = response.last_error

        return json.dumps(result, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error getting run status: %s", e)
        return json.dumps({
            "error": "not_found",
            "message": str(e)
        })


@mcp.tool()
@track_mcp_tool("list_runs")
async def list_runs(
    workflow_id: str,
    limit: int = 50,
) -> str:
    """
    List recent runs for a workflow.

    Args:
        workflow_id: The workflow ID
        limit: Maximum number of runs to return (default: 50)

    Returns:
        JSON with list of runs and their statuses
    """
    try:
        await _ensure_db()
        user = await _get_mcp_user()

        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.execution import list_workflow_runs

        response = await list_workflow_runs(user, workflow_id, limit=limit)

        runs = []
        for run in response.runs:
            run_data = {
                "run_id": run.run_id,
                "status": run.status,
                "created_at": run.created_at.isoformat(),
            }
            if run.started_at:
                run_data["started_at"] = run.started_at.isoformat()
            if run.finished_at:
                run_data["finished_at"] = run.finished_at.isoformat()
            if run.error:
                run_data["error"] = run.error
            runs.append(run_data)

        return json.dumps({
            "workflow_id": response.workflow_id,
            "runs": runs,
            "total": len(runs),
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error listing runs: %s", e)
        return json.dumps({
            "workflow_id": workflow_id,
            "runs": [],
            "error": str(e)
        })


@mcp.tool()
@track_mcp_tool("get_run_history")
async def get_run_history(
    workflow_id: str,  # pylint: disable=unused-argument # Reason: Required for MCP tool API signature consistency
    run_id: str,
) -> str:
    """
    Get detailed execution history for a workflow run.

    Returns node-by-node trace information including inputs, outputs, and timing.

    Args:
        workflow_id: The workflow ID
        run_id: The run ID to get history for

    Returns:
        JSON with execution history including node traces
    """
    try:
        await _ensure_db()
        user = await _get_mcp_user()

        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.history import get_run_history as get_run_history_service

        response = await get_run_history_service(user, run_id)

        return json.dumps({
            "run_id": response.run_id,
            "history": response.history,
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error getting run history: %s", e)
        return json.dumps({
            "run_id": run_id,
            "history": [],
            "error": str(e)
        })

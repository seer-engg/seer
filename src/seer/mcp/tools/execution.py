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


async def _get_latest_trigger_event(
    workflow_pk: int,
    trigger_id: Optional[str] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Fetch the most recent trigger event for a workflow.

    Args:
        workflow_pk: The workflow's primary key (integer)
        trigger_id: Optional specific trigger ID for multi-trigger workflows

    Returns:
        tuple of (event_envelope, trigger_id_used, error_message)
        - event_envelope: The event dict if found, None otherwise
        - trigger_id_used: Which trigger was selected
        - error_message: Error string if no event available, None if success
    """
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.database.workflow_models import TriggerSubscription, TriggerEvent

    # Find trigger subscription(s) for this workflow
    query = TriggerSubscription.filter(workflow_id=workflow_pk)
    if trigger_id:
        query = query.filter(trigger_id=trigger_id)

    subscription = await query.first()

    if not subscription:
        if trigger_id:
            return None, None, f"No trigger found with id '{trigger_id}'"
        # No triggers = manual workflow, proceed normally
        return None, None, None

    # Get most recent event for this trigger
    event = await TriggerEvent.filter(
        subscription_id=subscription.id
    ).order_by("-received_at").first()

    if not event:
        return None, subscription.trigger_id, (
            f"Workflow has trigger '{subscription.trigger_id}' but no events received yet. "
            "Send a test event first or use the web UI."
        )

    return event.event, subscription.trigger_id, None


@mcp.tool()
@track_mcp_tool("run_workflow")
async def run_workflow(
    workflow_id: str,
    inputs: Optional[Dict[str, Any]] = None,
    version: Optional[int] = None,
    trigger_id: Optional[str] = None,
) -> str:
    """
    Execute a workflow and return the run ID.

    For workflows with triggers, automatically selects the most recent trigger event.
    For workflows without triggers, creates a manual run.

    Args:
        workflow_id: The workflow ID to execute
        inputs: Optional input variables for the workflow
        version: Optional specific version to run (default: draft or latest published)
        trigger_id: Optional trigger ID for multi-trigger workflows (uses first trigger if not specified)

    Returns:
        JSON with run_id, status, workflow metadata, and trigger info if auto-selected
    """
    try:
        await _ensure_db()
        user = await _get_mcp_user()

        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest
        from seer.database.workflow_models import parse_workflow_public_id

        # Auto-select latest trigger event if workflow has triggers
        workflow_pk = parse_workflow_public_id(workflow_id)
        event_envelope, used_trigger_id, error = await _get_latest_trigger_event(
            workflow_pk, trigger_id
        )

        if error:
            return json.dumps({
                "error": "no_trigger_event",
                "message": error,
                "workflow_id": workflow_id,
                "trigger_id": used_trigger_id,
            })

        request = RunFromWorkflowRequest(
            version=version,
            inputs=inputs or {},
            config={},
            trigger_event_override=event_envelope,
            trigger_id=trigger_id,
        )

        response = await run_saved_workflow(user, workflow_id, request)

        result = {
            "run_id": response.run_id,
            "status": response.status,
            "workflow_id": response.workflow_id,
            "created_at": response.created_at.isoformat(),
            "started_at": response.started_at.isoformat() if response.started_at else None,
        }

        if event_envelope and used_trigger_id:
            result["trigger_id_used"] = used_trigger_id
            result["auto_selected_event"] = True

        return json.dumps(result, indent=2)

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

"""
MCP workflow CRUD tools for creating and managing workflows.

Delegates to existing workflow lifecycle services.
"""
# pylint: disable=cyclic-import # Reason: mcp server module registers tools via imports

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from seer.mcp.server import mcp
from seer.mcp.auth import get_mcp_authenticated_user
from seer.mcp.tracking import track_mcp_tool
from seer.database import User, init_db
from seer.logger import get_logger
from seer.tools.workflow_validation import run_full_validation

logger = get_logger(__name__)

# System user ID in the database (Postgres primary key)
SYSTEM_USER_ID = 1


async def _ensure_db() -> None:
    """Ensure database is initialized."""
    try:
        # pylint: disable=import-outside-toplevel # Reason: Lazy load to avoid circular imports at module level
        from tortoise import Tortoise
        # pylint: disable=protected-access # Reason: Tortoise doesn't expose public init check
        if not Tortoise._inited:
            await init_db()
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Graceful fallback
        logger.warning("Database initialization check failed: %s", e)


async def _get_system_user() -> User:
    """
    Get the system user for MCP operations.

    Used as fallback when no authenticated user is available (e.g., stdio transport).
    The system user is the pre-seeded user with database ID 1.
    """
    await _ensure_db()

    user = await User.get_or_none(id=SYSTEM_USER_ID)
    if user is None:
        raise RuntimeError(f"System user (id={SYSTEM_USER_ID}) not found in database. Ensure the database is properly seeded.")
    return user


async def _get_mcp_user() -> User:
    """
    Get the authenticated user or fall back to system user.

    For HTTP transport with Clerk auth: Returns the authenticated Clerk user.
    For stdio transport or when auth is not configured: Returns the system user (id=1).
    """
    await _ensure_db()

    # Try to get authenticated user from context (set by MCPAuthMiddleware)
    verified_token = get_mcp_authenticated_user()
    if verified_token:
        # Get or create database user from Clerk user
        user, created = await User.get_or_create(
            user_id=verified_token.user_id,
            defaults={
                "email": verified_token.email,
                "first_name": verified_token.first_name,
                "last_name": verified_token.last_name,
                "claims": verified_token.claims,
            }
        )
        if created:
            logger.info("Created user from MCP auth: %s", verified_token.user_id)
        return user

    # Fall back to system user (stdio transport or no auth configured)
    return await _get_system_user()


@mcp.tool()
@track_mcp_tool("list_workflows")
async def list_workflows(
    limit: int = 50,
    cursor: Optional[str] = None,
) -> str:
    """
    List all workflows for the current user.

    Returns paginated list of workflows with basic metadata.

    Args:
        limit: Maximum number of workflows to return (default: 50, max: 100)
        cursor: Pagination cursor from previous response

    Returns:
        JSON with list of workflows and next_cursor for pagination
    """
    try:
        user = await _get_mcp_user()

        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.lifecycle import list_workflows as list_workflows_service

        response = await list_workflows_service(user, limit=limit, cursor=cursor)

        items = []
        for item in response.items:
            items.append({
                "workflow_id": item.workflow_id,
                "name": item.name,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
            })

        return json.dumps({
            "workflows": items,
            "total": len(items),
            "next_cursor": response.next_cursor,
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error listing workflows: %s", e)
        return json.dumps({
            "workflows": [],
            "error": str(e)
        })


@mcp.tool()
@track_mcp_tool("get_workflow")
async def get_workflow(workflow_id: str) -> str:
    """
    Get a workflow by its ID.

    Returns the full workflow specification including nodes, edges, and triggers.

    Args:
        workflow_id: The workflow ID (e.g., "wf_abc123")

    Returns:
        JSON with workflow details including full spec
    """
    try:
        user = await _get_mcp_user()

        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.lifecycle import get_workflow as get_workflow_service

        response = await get_workflow_service(user, workflow_id)

        return json.dumps({
            "workflow_id": response.workflow_id,
            "name": response.name,
            "spec": response.spec.model_dump(mode="json"),
            "created_at": response.created_at.isoformat(),
            "updated_at": response.updated_at.isoformat(),
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error getting workflow: %s", e)
        return json.dumps({
            "error": "not_found",
            "message": str(e)
        })


@mcp.tool()
@track_mcp_tool("publish_workflow")
async def publish_workflow(workflow_id: str) -> str:
    """
    Publish the draft version of a workflow, making it active.

    This promotes the current draft to a released version.
    The workflow will start responding to triggers after publishing.

    Args:
        workflow_id: The workflow ID to publish

    Returns:
        JSON with published workflow details
    """
    try:
        user = await _get_mcp_user()

        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.lifecycle import publish_workflow as publish_workflow_service
        from seer.api.workflows.models import WorkflowPublishRequest

        request = WorkflowPublishRequest()
        response = await publish_workflow_service(user, workflow_id, request)

        return json.dumps({
            "workflow_id": response.workflow_id,
            "name": response.name,
            "message": "Workflow published successfully",
            "updated_at": response.updated_at.isoformat(),
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error publishing workflow: %s", e)
        return json.dumps({
            "error": "publish_failed",
            "message": str(e)
        })


def _build_auto_fix_info(schema_fixes: list) -> Dict[str, Any]:
    """Build auto-fix information for response."""
    return {
        "trigger_schemas_updated": schema_fixes,
        "recommendation": (
            "Event schemas were auto-corrected to use canonical schemas from the registry. "
            "If you have expressions referencing trigger data (e.g., ${trigger.data.subject}), "
            "verify they use correct field paths from the 'available_fields' shown above."
        )
    }


@mcp.tool()
@track_mcp_tool("validate_and_upsert_workflow")
async def validate_and_upsert_workflow(
    name: str,
    spec: Dict[str, Any],
    workflow_id: Optional[str] = None,
    summary: Optional[str] = None,
) -> str:
    """
    Validate and create or update a workflow with comprehensive validation.

    If workflow_id is provided, updates the existing workflow's draft.
    If workflow_id is not provided, creates a new workflow.

    Performs comprehensive validation chain before persisting:
    1. Pydantic schema validation
    2. Tool existence check against registry
    3. Trigger key validation against trigger registry
    4. Auto-fix trigger event_schemas with canonical schemas from registry
    5. Full compilation validation

    The workflow is created/updated in draft status. Use publish_workflow to make it active.

    Args:
        name: Name for the workflow (e.g., "Welcome Email on Signup")
        spec: Workflow specification as a JSON object with:
              - version: Must be string "2" (NOT "1.0", NOT "2.0", exactly "2")
              - nodes: Array of node objects (required)
              - edges: Array of edge objects (optional)
              - triggers: Array of trigger objects (optional)
        workflow_id: Optional workflow ID to update. If not provided, creates a new workflow.
        summary: Optional natural language description of what the workflow does

    Returns:
        JSON with status, message, validated spec, workflow_id, and any auto-fixes applied
    """
    try:
        user = await _get_mcp_user()
        validation = await run_full_validation(user, spec)

        if not validation.success:
            response = {
                "status": "error",
                "error_type": validation.error.error_type,
                "message": validation.error.message,
            }
            if validation.error.hint:
                response["hint"] = validation.error.hint
            return json.dumps(response)

        # All validations passed - create or update workflow
        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services import lifecycle as wf_lifecycle
        from seer.api.workflows.models import WorkflowCreateRequest, WorkflowDraftPatchRequest

        final_spec = validation.validated_spec

        if workflow_id:
            resp = await wf_lifecycle.patch_workflow_draft(user, workflow_id, WorkflowDraftPatchRequest(spec=final_spec))
            result = {"status": "ok", "message": "Workflow updated successfully", "workflow_id": resp.workflow_id,
                      "name": resp.name, "spec": resp.spec.model_dump(mode="json"), "updated_at": resp.updated_at.isoformat()}
        else:
            resp = await wf_lifecycle.create_workflow(user, WorkflowCreateRequest(name=name, spec=final_spec))
            result = {"status": "ok", "message": "Workflow created successfully", "workflow_id": resp.workflow_id,
                      "name": resp.name, "spec": resp.spec.model_dump(mode="json"), "created_at": resp.created_at.isoformat()}

        if validation.schema_fixes:
            result["auto_fixes"] = _build_auto_fix_info(validation.schema_fixes)
        if summary:
            result["summary"] = summary

        return json.dumps(result, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error upserting workflow: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
@track_mcp_tool("analyze_workflow")
async def analyze_workflow(workflow_id: str) -> str:
    """
    Analyze the structure and composition of a workflow.

    Returns a JSON string describing the workflow's blocks, connections, and configuration.
    Useful for understanding workflow complexity and debugging.

    Args:
        workflow_id: The workflow ID to analyze (e.g., "wf_abc123")

    Returns:
        JSON with total_blocks, total_connections, block_types breakdown, blocks array, connections array
    """
    try:
        user = await _get_mcp_user()

        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.lifecycle import get_workflow as get_workflow_service
        from seer.services.workflows.analysis import build_workflow_analysis

        response = await get_workflow_service(user, workflow_id)
        analysis = build_workflow_analysis(workflow_id, response.name, response.spec)

        return json.dumps(analysis, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error analyzing workflow: %s", e)
        return json.dumps({
            "error": "analysis_failed",
            "message": str(e)
        })


@mcp.tool()
@track_mcp_tool("delete_workflow")
async def delete_workflow(workflow_id: str) -> str:
    """
    Delete a workflow and all its versions.

    WARNING: This permanently deletes the workflow and cannot be undone.

    Args:
        workflow_id: The workflow ID to delete

    Returns:
        JSON with deletion confirmation
    """
    try:
        user = await _get_mcp_user()

        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.lifecycle import delete_workflow as delete_workflow_service

        await delete_workflow_service(user, workflow_id)

        return json.dumps({
            "deleted": True,
            "workflow_id": workflow_id,
            "message": "Workflow deleted successfully"
        })

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error deleting workflow: %s", e)
        return json.dumps({
            "deleted": False,
            "error": str(e)
        })

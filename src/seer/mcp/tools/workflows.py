"""
MCP workflow CRUD tools for creating and managing workflows.

Delegates to existing workflow lifecycle services.
"""
# pylint: disable=cyclic-import # Reason: mcp server module registers tools via imports

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from seer.mcp.server import mcp
from seer.mcp.auth import get_mcp_authenticated_user
from seer.database import User, init_db
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.errors import ValidationPhaseError
from seer.logger import get_logger

logger = get_logger(__name__)

# Service user ID for MCP operations (configurable via environment)
MCP_SERVICE_USER_ID = os.environ.get("SEER_MCP_USER_ID", "mcp-service-user")


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


async def _get_service_user() -> User:
    """
    Get or create the MCP service user for workflow operations.

    Used as fallback when no authenticated user is available (e.g., stdio transport).
    """
    await _ensure_db()

    user, created = await User.get_or_create(
        user_id=MCP_SERVICE_USER_ID,
        defaults={
            "email": "mcp@seer.local",
            "first_name": "MCP",
            "last_name": "Service",
        }
    )
    if created:
        logger.info("Created MCP service user: %s", MCP_SERVICE_USER_ID)
    return user


async def _get_mcp_user() -> User:
    """
    Get the authenticated user or fall back to service user.

    For HTTP transport with Clerk auth: Returns the authenticated Clerk user.
    For stdio transport or when auth is not configured: Returns the service user.
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

    # Fall back to service user (stdio transport or no auth configured)
    return await _get_service_user()


@mcp.tool()
async def create_workflow(
    name: str,
    spec: Dict[str, Any],
) -> str:
    """
    Create a new workflow with the given name and specification.

    The workflow is created in draft status. Use publish_workflow to make it active.

    Args:
        name: Name for the workflow (e.g., "Welcome Email on Signup")
        spec: Workflow specification as a JSON object with version, nodes, edges, and triggers

    Returns:
        JSON with workflow_id, name, spec, and timestamps
    """
    try:
        # Validate spec first
        try:
            validated_spec = parse_workflow_spec(spec)
        except ValidationPhaseError as exc:
            return json.dumps({
                "error": "validation_failed",
                "message": str(exc),
                "hint": "Check that your spec follows the workflow schema"
            })

        user = await _get_mcp_user()

        # Import here to avoid circular imports
        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.lifecycle import create_workflow as create_workflow_service
        from seer.api.workflows.models import WorkflowCreateRequest

        request = WorkflowCreateRequest(name=name, spec=validated_spec)
        response = await create_workflow_service(user, request)

        return json.dumps({
            "workflow_id": response.workflow_id,
            "name": response.name,
            "spec": response.spec.model_dump(mode="json"),
            "created_at": response.created_at.isoformat(),
            "updated_at": response.updated_at.isoformat(),
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error creating workflow: %s", e)
        return json.dumps({
            "error": "create_failed",
            "message": str(e)
        })


@mcp.tool()
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


def _validate_tool_references(validated_spec) -> list[str]:
    """Check that tool nodes reference valid tools in the registry."""
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.tools.base import get_tool

    errors = []
    for node in validated_spec.nodes:
        if node.type == "tool":
            tool_name = getattr(node, "tool", None)
            if tool_name and not get_tool(tool_name):
                errors.append(f"Tool '{tool_name}' not found in registry")
    return errors


def _validate_trigger_references(validated_spec) -> list[str]:
    """Check that triggers reference valid trigger keys in the registry."""
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.core.registry.trigger_registry import trigger_registry

    errors = []
    if validated_spec.triggers:
        for trigger in validated_spec.triggers:
            if not trigger_registry.maybe_get(trigger.key):
                errors.append(f"Trigger '{trigger.key}' not found in registry")
    return errors


async def _validate_compilation(user, spec: Dict[str, Any]) -> Optional[str]:
    """
    Run full compilation validation.

    Returns error message if compilation fails, None if successful.
    """
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.core.runtime.global_compiler import WorkflowCompilerSingleton

    try:
        compiler = WorkflowCompilerSingleton.instance()
        await compiler.compile(user, spec, checkpointer=None)
        return None
    except Exception as compile_error:  # pylint: disable=broad-exception-caught # Reason: Catch all compilation errors
        return str(compile_error)


@mcp.tool()
async def validate_workflow(spec: Dict[str, Any]) -> str:
    """
    Validate a workflow specification without creating it.

    Use this to check if a workflow spec is valid before creating it.
    Returns validation errors if the spec is invalid.

    Args:
        spec: Workflow specification as a JSON object

    Returns:
        JSON with validation result (ok: true/false) and any errors
    """
    try:
        # First, validate against Pydantic schema
        try:
            validated_spec = parse_workflow_spec(spec)
        except ValidationPhaseError as exc:
            return json.dumps({
                "ok": False,
                "error_type": "schema_validation",
                "message": str(exc),
                "hint": "Check that your spec follows the workflow schema"
            })

        # Check tools and triggers exist
        tool_errors = _validate_tool_references(validated_spec)
        trigger_errors = _validate_trigger_references(validated_spec)
        all_errors = tool_errors + trigger_errors

        if all_errors:
            return json.dumps({
                "ok": False,
                "error_type": "reference_validation",
                "errors": all_errors,
                "hint": "Use search_tools() and list_triggers() to find valid names"
            })

        # Full compilation validation
        user = await _get_mcp_user()
        compile_error = await _validate_compilation(user, spec)

        if compile_error:
            return json.dumps({
                "ok": False,
                "error_type": "compilation",
                "message": compile_error,
            })

        return json.dumps({
            "ok": True,
            "message": "Workflow spec is valid",
            "spec": validated_spec.model_dump(mode="json")
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error validating workflow: %s", e)
        return json.dumps({
            "ok": False,
            "error": str(e)
        })


@mcp.tool()
async def update_workflow_draft(
    workflow_id: str,
    spec: Dict[str, Any],
) -> str:
    """
    Update the draft version of a workflow with a new specification.

    This updates the draft spec without publishing it.
    Use publish_workflow to make changes active.

    Args:
        workflow_id: The workflow ID to update
        spec: New workflow specification

    Returns:
        JSON with updated workflow details
    """
    try:
        # Validate spec first
        try:
            validated_spec = parse_workflow_spec(spec)
        except ValidationPhaseError as exc:
            return json.dumps({
                "error": "validation_failed",
                "message": str(exc),
            })

        user = await _get_mcp_user()

        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.lifecycle import patch_workflow_draft
        from seer.api.workflows.models import WorkflowDraftPatchRequest

        request = WorkflowDraftPatchRequest(spec=validated_spec)
        response = await patch_workflow_draft(user, workflow_id, request)

        return json.dumps({
            "workflow_id": response.workflow_id,
            "name": response.name,
            "spec": response.spec.model_dump(mode="json"),
            "updated_at": response.updated_at.isoformat(),
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error updating workflow draft: %s", e)
        return json.dumps({
            "error": "update_failed",
            "message": str(e)
        })


@mcp.tool()
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


@mcp.tool()
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

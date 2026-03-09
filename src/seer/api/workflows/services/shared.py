from __future__ import annotations

# pylint: disable=duplicate-code  # Reason: Shared workflow version helpers intentionally overlap with lifecycle module
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from seer.api.core.errors import COMPILE_PROBLEM, VALIDATION_PROBLEM
from seer.api.core.errors import raise_problem as _raise_problem
from seer.api.workflows import models as api_models
from seer.core.errors import WorkflowCompilerError
from seer.database import (
    Organization,
    OrganizationMembership,
    User,
    Workflow,
    WorkflowVersion,
    WorkflowVersionStatus,
    WorkflowVisibility,
    parse_workflow_public_id,
)
from seer.database.organization_models import OrganizationRole, WorkflowAssignment
from seer.core.schema.models import WorkflowSpec
from seer.core.runtime.global_compiler import WorkflowCompilerSingleton


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def get_published_version(workflow: Workflow) -> Optional[WorkflowVersion]:
    """Get the published (RELEASED) version for a workflow."""
    return await WorkflowVersion.filter(
        workflow=workflow,
        status=WorkflowVersionStatus.RELEASED
    ).first()


def _spec_to_dict(spec: WorkflowSpec) -> Dict[str, Any]:
    return spec.model_dump(mode="json")


def _hash_spec(spec_dict: Dict[str, Any]) -> str:
    serialized = json.dumps(spec_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


async def _get_draft_version(
    workflow: Workflow,
    create_if_missing: bool = True,
    user: Optional[User] = None,
) -> Optional[WorkflowVersion]:
    """
    Get existing DRAFT version, optionally create if missing.

    If create_if_missing=True and no DRAFT exists:
    - Copies from published version if exists
    - Creates empty spec if no published version
    """
    draft = await WorkflowVersion.filter(
        workflow=workflow,
        status=WorkflowVersionStatus.DRAFT
    ).first()

    if draft or not create_if_missing:
        return draft

    # Create on-demand: copy from published or use empty spec
    published = await get_published_version(workflow)
    if published:
        spec_dict = json.loads(json.dumps(published.spec))
    else:
        spec_dict = {"version": "2.0", "nodes": [], "edges": []}

    spec_hash = _hash_spec(spec_dict)
    return await WorkflowVersion.create(
        workflow=workflow,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec_dict,
        created_by=user,
        updated_by=user,
        spec_hash=spec_hash,
        version_number=0,
    )


async def _update_draft_version(
    version: WorkflowVersion,
    spec_dict: Dict[str, Any],
    user: User,
) -> None:
    """Update DRAFT version in-place (enforces mutability rules)."""
    if version.status != WorkflowVersionStatus.DRAFT:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Cannot modify non-draft version",
            detail="Only DRAFT versions can be edited",
            status=400,
        )

    version.spec = spec_dict
    version.updated_by = user
    version.updated_at = _now()
    version.spec_hash = _hash_spec(spec_dict)
    await version.save()
    await Workflow.filter(id=version.workflow_id).update(updated_at=_now())


async def _get_workflow(user: User, workflow_id: str) -> Workflow:
    try:
        pk = parse_workflow_public_id(workflow_id)
    except ValueError:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid workflow id",
            detail="Workflow id is invalid",
            status=400,
        )
    workflow = await Workflow.filter(id=pk, user=user).first()
    if workflow is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Workflow not found",
            detail=f"Workflow '{workflow_id}' not found",
            status=404,
        )
    return workflow


async def _can_view_workflow(
    user: User,
    workflow: Workflow,
    membership: Optional[OrganizationMembership] = None,
) -> bool:
    """
    Check if user can view a workflow based on organization membership and visibility.

    Args:
        user: The user attempting to view
        workflow: The workflow being viewed
        membership: User's membership in the workflow's organization (optional)

    Returns:
        True if user can view the workflow
    """
    # If workflow has no organization (legacy), only owner can view
    if workflow.organization_id is None:
        return workflow.user_id == user.id

    # If no membership provided, fetch it
    if membership is None:
        membership = await OrganizationMembership.get_or_none(
            organization_id=workflow.organization_id,
            user=user,
        )
        if membership is None:
            return False

    # Owner and Admin can see everything in their org
    if membership.role in (OrganizationRole.OWNER, OrganizationRole.ADMIN):
        return True

    # Get visibility (default to TEAM for backward compatibility)
    visibility = getattr(workflow, "visibility", WorkflowVisibility.TEAM)

    if membership.role == OrganizationRole.USER:
        if visibility == WorkflowVisibility.TEAM:
            return True
        if visibility == WorkflowVisibility.PRIVATE:
            return workflow.user_id == user.id
        if visibility == WorkflowVisibility.ASSIGNED:
            return await WorkflowAssignment.filter(workflow=workflow, user=user).exists()

    if membership.role == OrganizationRole.CONSULTANT:
        # Consultants can only view assigned workflows
        return await WorkflowAssignment.filter(workflow=workflow, user=user).exists()

    return False


async def _can_manage_workflow(
    user: User,
    workflow: Workflow,
    membership: Optional[OrganizationMembership] = None,
) -> bool:
    """
    Check if user can edit/delete a workflow.

    Args:
        user: The user attempting the action
        workflow: The workflow being managed
        membership: User's membership in the workflow's organization (optional)

    Returns:
        True if user can manage the workflow
    """
    # If workflow has no organization (legacy), only owner can manage
    if workflow.organization_id is None:
        return workflow.user_id == user.id

    # If no membership provided, fetch it
    if membership is None:
        membership = await OrganizationMembership.get_or_none(
            organization_id=workflow.organization_id,
            user=user,
        )
        if membership is None:
            return False

    # Owner/Admin can manage all workflows in the org
    if membership.role in (OrganizationRole.OWNER, OrganizationRole.ADMIN):
        return True

    # Users can manage TEAM-visible workflows (not just their own)
    if membership.role == OrganizationRole.USER:
        visibility = getattr(workflow, "visibility", WorkflowVisibility.TEAM)
        if visibility == WorkflowVisibility.TEAM:
            return True
        if visibility == WorkflowVisibility.PRIVATE:
            return workflow.user_id == user.id
        if visibility == WorkflowVisibility.ASSIGNED:
            return await WorkflowAssignment.filter(workflow=workflow, user=user).exists()

    # Consultants cannot manage workflows
    return False


async def _get_workflow_org_scoped(
    user: User,
    workflow_id: str,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
    require_manage: bool = False,
) -> Workflow:
    """
    Get a workflow with organization-scoped access control.

    Args:
        user: The authenticated user
        workflow_id: The workflow public ID (e.g., "wf_123")
        organization: The current organization context (optional)
        membership: User's membership in the organization (optional)
        require_manage: If True, require manage permission (for edit/delete)

    Returns:
        The workflow if access is allowed

    Raises:
        HTTP 404 if workflow not found or access denied
    """
    try:
        pk = parse_workflow_public_id(workflow_id)
    except ValueError:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid workflow id",
            detail="Workflow id is invalid",
            status=400,
        )

    # First, try to find by organization (if provided)
    if organization:
        workflow = await Workflow.filter(id=pk, organization=organization).first()
    else:
        # Fall back to user-scoped (legacy behavior)
        workflow = await Workflow.filter(id=pk, user=user).first()

    if workflow is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Workflow not found",
            detail=f"Workflow '{workflow_id}' not found",
            status=404,
        )

    # Check access permissions
    if require_manage:
        has_access = await _can_manage_workflow(user, workflow, membership)
    else:
        has_access = await _can_view_workflow(user, workflow, membership)

    if not has_access:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Workflow not found",
            detail=f"Workflow '{workflow_id}' not found",
            status=404,
        )

    return workflow


def raise_compiler_error(exc: WorkflowCompilerError, problem_title: str, type_uri: str = COMPILE_PROBLEM) -> None:
    """Convert WorkflowCompilerError to API problem response and raise it."""
    problem_errors = [
        api_models.ProblemError(
            code=err.code,
            message=err.message,
            node_id=err.node_id,
            location=err.location,
            expression=err.expression,
        )
        for err in exc.errors
    ] if exc.errors else None

    _raise_problem(
        type_uri=type_uri,
        title=problem_title,
        detail=str(exc),
        status=400,
        errors=problem_errors,
    )


async def validate_workflow_spec(
    user: User,
    spec: WorkflowSpec,
    organization_id: Optional[int] = None,
) -> None:
    """
    Validate workflow spec by running the compiler pipeline.
    Raises HTTP 400 if validation fails.

    Args:
        user: The user validating the workflow
        spec: The workflow spec to validate
        organization_id: Optional organization ID for resolving shared connections
    """
    from seer.api.agents.checkpointer import get_checkpointer  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import at module load
    compiler = WorkflowCompilerSingleton.instance()
    spec_dict = _spec_to_dict(spec)
    checkpointer = await get_checkpointer()
    try:
        await compiler.compile(user, spec_dict, checkpointer=checkpointer, organization_id=organization_id)
    except WorkflowCompilerError as exc:
        raise_compiler_error(exc, "Workflow validation failed")

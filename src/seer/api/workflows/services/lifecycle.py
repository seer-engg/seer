# pylint: disable=duplicate-code,too-many-lines
# Reason: Shared workflow version mutation snippets are reused in services.shared;
# module contains multiple closely related CRUD operations that would be awkward to split
"""Workflow CRUD operations and version management."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
from typing import Any, Dict, Optional
from pydantic import ValidationError
from tortoise.exceptions import DoesNotExist

from seer.api.workflows import models as api_models
from seer.api.workflows.services.shared import (
    VALIDATION_PROBLEM,
    _get_draft_version,
    _get_workflow,
    _get_workflow_org_scoped,
    _hash_spec,
    _now,
    _raise_problem,
    _spec_to_dict,
    _update_draft_version,
    get_published_version,
    validate_workflow_spec,
)
from seer.database.organization_models import OrganizationRole, OrganizationType
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
from seer.core.schema.models import WorkflowSpec
from seer.database.workflow_models import TriggerSubscription, WorkflowRun, WorkflowRunStatus
from seer.services.collaboration import CollaborationEventType, publish_collaboration_event
from seer.tools.base import list_tools as list_all_tools

# ===== Helper Functions =====


async def _enrich_trigger_specs_with_subscriptions(workflow: Workflow, spec: WorkflowSpec) -> None:
    """
    Enrich trigger specs with subscription data in ui_meta field.

    Adds webhook_url, secret_token, form_url to trigger.ui_meta by fetching
    TriggerSubscription records.
    """
    if not spec.triggers:
        return

    # Import here to avoid circular dependency
    # pylint: disable=import-outside-toplevel
    from seer.api.workflows.services.triggers import (
        _build_webhook_url,
        _build_form_url,
        _should_emit_webhook_url,
    )

    # Fetch all subscriptions for this workflow in one query (efficient)
    subscriptions = await TriggerSubscription.filter(workflow=workflow).all()
    subscription_map = {sub.trigger_id: sub for sub in subscriptions}

    # Enrich each trigger spec's ui_meta
    for trigger in spec.triggers:
        subscription = subscription_map.get(trigger.id)

        # If no subscription exists (e.g., draft only), skip enrichment
        if not subscription:
            continue

        # Build webhook URL if applicable
        webhook_url = None
        if _should_emit_webhook_url(subscription.trigger_key):
            webhook_url = _build_webhook_url(subscription.id, subscription.trigger_key)

        # Build form URL if applicable
        form_url = None
        if subscription.trigger_key == "form.hosted":
            form_url = _build_form_url(subscription)

        # Add enrichment data to ui_meta
        trigger.ui_meta["subscription_id"] = subscription.id
        trigger.ui_meta["secret_token"] = subscription.secret_token
        if webhook_url:
            trigger.ui_meta["webhook_url"] = webhook_url
        if form_url:
            trigger.ui_meta["form_url"] = form_url
        if subscription.created_at:
            trigger.ui_meta["created_at"] = subscription.created_at.isoformat()
        if subscription.updated_at:
            trigger.ui_meta["updated_at"] = subscription.updated_at.isoformat()


def _extract_integrations(spec_dict: Optional[Dict[str, Any]]) -> list[str]:
    """Extract deduplicated integration types from workflow spec tool nodes."""
    if not spec_dict:
        return []
    nodes = spec_dict.get("nodes", [])
    tool_names = {n.get("tool", "") for n in nodes if n.get("type") == "tool" and n.get("tool")}
    if not tool_names:
        return []
    # Build lookup from registry
    tool_integration_map = {}
    for t in list_all_tools():
        if t.name in tool_names:
            integration = getattr(t, "integration_type", None) or t.name.split("_")[0]
            tool_integration_map[t.name] = integration
    # Fallback for tools not in registry
    integrations = set()
    for name in tool_names:
        if name in tool_integration_map:
            integrations.add(tool_integration_map[name])
        else:
            integrations.add(name.split("_")[0])
    return sorted(integrations)


async def _workflow_summary(workflow: Workflow, draft_version: Optional[WorkflowVersion] = None) -> api_models.WorkflowSummary:
    """
    Create a workflow summary.

    If draft_version is not provided, it will be fetched from the database.
    Pass it explicitly when available to avoid extra queries.
    """
    updated_at = draft_version.updated_at if draft_version else workflow.updated_at
    spec_dict = draft_version.spec if draft_version else None
    integrations = _extract_integrations(spec_dict)

    return api_models.WorkflowSummary(
        workflow_id=workflow.workflow_id,
        name=workflow.name,
        created_at=workflow.created_at,
        updated_at=updated_at,
        is_published=workflow.is_published,
        is_active=workflow.is_active,
        integrations=integrations,
    )


async def _publish_workflow_event(
    workflow: Workflow,
    *,
    event_type: CollaborationEventType,
    actor: User,
    payload: dict[str, Any] | None = None,
) -> None:
    await publish_collaboration_event(
        organization_id=workflow.organization_id,
        event_type=event_type,
        resource_type="workflow",
        resource_id=workflow.workflow_id,
        actor=actor,
        payload=payload,
    )


async def toggle_workflow_published(
    user: User,
    workflow_id: str,
    is_published: bool,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.WorkflowSummary:
    """Toggle the is_published flag on a workflow."""
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership, require_manage=True)
    workflow.is_published = is_published
    await workflow.save(update_fields=["is_published", "updated_at"])
    await _publish_workflow_event(
        workflow,
        event_type=CollaborationEventType.WORKFLOW_PUBLISHED if is_published else CollaborationEventType.WORKFLOW_UNPUBLISHED,
        actor=user,
        payload={"is_published": is_published},
    )
    return await _workflow_summary(workflow)


async def toggle_workflow_active(
    user: User,
    workflow_id: str,
    is_active: bool,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.WorkflowSummary:
    """Toggle the is_active flag on a workflow."""
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership, require_manage=True)
    workflow.is_active = is_active
    await workflow.save(update_fields=["is_active", "updated_at"])
    await _publish_workflow_event(
        workflow,
        event_type=CollaborationEventType.WORKFLOW_ACTIVE_CHANGED,
        actor=user,
        payload={"is_active": is_active},
    )
    return await _workflow_summary(workflow)


def _serialize_version_list_item(
    version: WorkflowVersion,
    *,
    latest_version_id: Optional[int],
    published_version_id: Optional[int],
) -> api_models.WorkflowVersionListItem:
    return api_models.WorkflowVersionListItem(
        version_id=version.id,
        status=version.status.value if isinstance(version.status, WorkflowVersionStatus) else version.status,
        version_number=version.version_number,
        created_from_draft_revision=version.created_from_draft_revision,
        created_at=version.created_at,
        is_latest=version.id == latest_version_id if latest_version_id else False,
        is_published=version.id == published_version_id if published_version_id else False,
    )


async def _workflow_response(workflow: Workflow) -> api_models.WorkflowResponse:
    draft_version = await _get_draft_version(workflow, create_if_missing=False)

    # If no draft exists (e.g., after publishing), use published version spec
    if draft_version is None:
        published_version_obj = await get_published_version(workflow)
        if published_version_obj:
            raw_spec = published_version_obj.spec or {}
        else:
            # No draft and no published version - this shouldn't happen for normal workflows
            _raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Missing draft",
                detail="Workflow has no draft or published version",
                status=500,
            )
            return {}  # Unreachable, but satisfies type checker
    else:
        raw_spec = draft_version.spec or {}

    spec_version_raw = raw_spec.get("version")
    try:
        spec_version = Decimal(str(spec_version_raw))
    except (InvalidOperation, TypeError):
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Unsupported workflow spec version",
            detail=f"Workflow spec version '{spec_version_raw}' is invalid; minimum supported version is 2.",
            status=400,
        )
    if spec_version < Decimal(2):
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Unsupported workflow spec version",
            detail=f"Workflow spec version '{spec_version_raw}' is not supported; minimum supported version is 2.",
            status=400,
        )
    spec = WorkflowSpec.model_validate(raw_spec)

    # Enrich trigger specs with subscription data in ui_meta
    await _enrich_trigger_specs_with_subscriptions(workflow, spec)

    return api_models.WorkflowResponse(
        workflow_id=workflow.workflow_id,
        name=workflow.name,
        created_at=workflow.created_at,
        updated_at=workflow.updated_at,
        spec=spec,
    )


def _parse_workflow_cursor(cursor: Optional[str]) -> Optional[int]:
    if cursor is None:
        return None
    try:
        if cursor.startswith("wf_"):
            return parse_workflow_public_id(cursor)
        return int(cursor)
    except ValueError:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid cursor",
            detail="Cursor parameter is invalid",
            status=400,
        )
        return None  # Unreachable, but satisfies pylint


async def create_workflow(
    user: User,
    payload: api_models.WorkflowCreateRequest,
    organization: Optional[Organization] = None,
) -> api_models.WorkflowResponse:
    """
    Create a new workflow.

    Args:
        user: The authenticated user (becomes the workflow creator)
        payload: Workflow creation request
        organization: If provided, assigns the workflow to this organization

    Returns:
        The created workflow response
    """
    # Workflow limit check moved to UsageLimitMiddleware
    # Fallback to user's personal org if no org context provided (e.g. MCP-created workflows)
    if not organization:
        organization = await Organization.get_or_none(owner=user, type=OrganizationType.PERSONAL)

    spec_dict = _spec_to_dict(payload.spec)
    workflow = await Workflow.create(
        user=user,
        name=payload.name,
        organization=organization,
        visibility=WorkflowVisibility.TEAM if organization else WorkflowVisibility.PRIVATE,
    )
    await WorkflowVersion.create(
        workflow=workflow,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec_dict,
        created_by=user,
        updated_by=user,
        spec_hash=_hash_spec(spec_dict),
        version_number=0,
    )

    await _publish_workflow_event(
        workflow,
        event_type=CollaborationEventType.WORKFLOW_CREATED,
        actor=user,
        payload={"name": workflow.name},
    )

    return await _workflow_response(workflow)


async def list_workflows(
    user: User,
    *,
    limit: int = 50,
    cursor: Optional[str] = None,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.WorkflowListResponse:
    """
    List workflows accessible to the user.

    Args:
        user: The authenticated user
        limit: Maximum number of workflows to return
        cursor: Pagination cursor
        organization: If provided, list org-scoped workflows with visibility checks
        membership: User's membership in the organization (for permission checks)

    Returns:
        Paginated list of workflow summaries
    """
    limit = max(1, min(limit, 100))
    cursor_pk = _parse_workflow_cursor(cursor)

    # Build base query based on organization context
    if organization:
        # Organization-scoped: filter by org
        query = Workflow.filter(organization=organization)

        # For Users and Consultants, apply visibility filters
        if membership and membership.role not in (OrganizationRole.OWNER, OrganizationRole.ADMIN):
            # Users and consultants can see TEAM visibility, own PRIVATE, or ASSIGNED workflows
            if membership.role in (OrganizationRole.USER, OrganizationRole.CONSULTANT):
                from tortoise.expressions import Q  # pylint: disable=import-outside-toplevel  # Reason: conditional import for complex query
                query = query.filter(
                    Q(visibility=WorkflowVisibility.TEAM) |
                    Q(visibility=WorkflowVisibility.PRIVATE, user=user) |
                    Q(visibility=WorkflowVisibility.ASSIGNED, assignments__user=user)
                )
    else:
        # Legacy user-scoped behavior
        query = Workflow.filter(user=user)

    if cursor_pk:
        query = query.filter(id__lt=cursor_pk)

    records = await query.order_by("-id").limit(limit + 1)

    # Fetch all DRAFT versions for these workflows
    workflow_ids = [r.id for r in records[:limit]]
    drafts_by_workflow = {}
    if workflow_ids:
        draft_versions = await WorkflowVersion.filter(
            workflow_id__in=workflow_ids,
            status=WorkflowVersionStatus.DRAFT
        ).all()
        for dv in draft_versions:
            drafts_by_workflow[dv.workflow_id] = dv

    # Build summaries
    items = []
    for record in records[:limit]:
        draft_version = drafts_by_workflow.get(record.id)
        items.append(await _workflow_summary(record, draft_version))

    next_cursor = items[-1].workflow_id if len(records) > limit and items else None
    return api_models.WorkflowListResponse(items=items, next_cursor=next_cursor)


async def get_workflow(
    user: User,
    workflow_id: str,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.WorkflowResponse:
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership)
    return await _workflow_response(workflow)


async def list_workflow_versions(
    user: User,
    workflow_id: str,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.WorkflowVersionListResponse:
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership)
    versions = (
        await WorkflowVersion.filter(workflow=workflow)
        .order_by("-created_at")
        .all()
    )
    published_version_obj = await get_published_version(workflow)
    published_version_id = published_version_obj.id if published_version_obj else None
    latest_version_id = versions[0].id if versions else None
    items = [
        _serialize_version_list_item(
            version,
            latest_version_id=latest_version_id,
            published_version_id=published_version_id,
        )
        for version in versions
    ]
    return api_models.WorkflowVersionListResponse(
        workflow_id=workflow.workflow_id,
        versions=items,
    )


async def update_workflow(
    user: User,
    workflow_id: str,
    payload: api_models.WorkflowUpdateRequest,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.WorkflowResponse:
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership, require_manage=True)
    changed_fields: list[str] = []
    if payload.name is not None:
        workflow.name = payload.name
        changed_fields.append("name")
    await workflow.save()
    await _publish_workflow_event(
        workflow,
        event_type=CollaborationEventType.WORKFLOW_UPDATED,
        actor=user,
        payload={"name": workflow.name, "changed_fields": changed_fields or ["updated_at"]},
    )
    return await _workflow_response(workflow)


async def apply_workflow_from_spec(
    user: User,
    workflow_id: str,
    spec_payload: Dict[str, Any],
) -> api_models.WorkflowResponse:
    """
    Replace an existing workflow's spec with a validated WorkflowSpec payload.
    """
    workflow = await _get_workflow(user, workflow_id)
    try:
        spec = WorkflowSpec.model_validate(spec_payload)
    except (ValidationError, TypeError, ValueError) as exc:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid workflow spec",
            detail=str(exc),
            status=400,
        )

    draft_version = await _get_draft_version(workflow, create_if_missing=True, user=user)
    if not draft_version:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Failed to create draft",
            detail="Could not create or retrieve draft version",
            status=500,
        )
    await _update_draft_version(draft_version, _spec_to_dict(spec), user)
    return await _workflow_response(workflow)


async def patch_workflow_draft(
    user: User,
    workflow_id: str,
    payload: api_models.WorkflowDraftPatchRequest,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.WorkflowResponse:
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership, require_manage=True)

    # Get or create DRAFT version
    draft_version = await _get_draft_version(workflow, create_if_missing=True, user=user)
    if not draft_version:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Failed to create draft",
            detail="Could not create or retrieve draft version",
            status=500,
        )

    # Update draft version in-place
    spec_dict = _spec_to_dict(payload.spec)
    await _update_draft_version(draft_version, spec_dict, user)
    await _publish_workflow_event(
        workflow,
        event_type=CollaborationEventType.WORKFLOW_DRAFT_UPDATED,
        actor=user,
        payload={"changed_fields": ["spec"], "draft_version_id": draft_version.id},
    )

    return await _workflow_response(workflow)


async def restore_workflow_version(  # pylint: disable=too-many-positional-arguments  # Reason: All args are required for org-scoped version restore; splitting would obscure the call site
    user: User,
    workflow_id: str,
    version_id: int,
    payload: api_models.WorkflowVersionRestoreRequest,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.WorkflowResponse:
    del payload  # payload reserved for future extensions (e.g., metadata), kept for API compatibility
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership, require_manage=True)
    try:
        version = await WorkflowVersion.get(id=version_id, workflow=workflow)
    except DoesNotExist:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Version not found",
            detail=f"Version '{version_id}' does not belong to workflow '{workflow_id}'",
            status=404,
        )

    # Get or create DRAFT version
    draft_version = await _get_draft_version(workflow, create_if_missing=True, user=user)
    if not draft_version:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Failed to create draft",
            detail="Could not create or retrieve draft version",
            status=500,
        )

    # Update draft to match restored version
    spec_dict = json.loads(json.dumps(version.spec or {}))
    await _update_draft_version(draft_version, spec_dict, user)
    await _publish_workflow_event(
        workflow,
        event_type=CollaborationEventType.WORKFLOW_VERSION_RESTORED,
        actor=user,
        payload={"version_id": version.id, "draft_version_id": draft_version.id},
    )

    return await _workflow_response(workflow)


async def _next_release_number(workflow: Workflow) -> int:
    latest = (
        await WorkflowVersion.filter(workflow=workflow, version_number__isnull=False)
        .order_by("-version_number")
        .first()
    )
    if latest is None or latest.version_number is None:
        return 1
    return latest.version_number + 1


async def publish_workflow(
    user: User,
    workflow_id: str,
    payload: api_models.WorkflowPublishRequest,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.WorkflowResponse:
    del payload  # payload reserved for future publish options, kept for API compatibility
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership, require_manage=True)

    # Get existing DRAFT version (must exist to publish)
    draft_version = await _get_draft_version(workflow, create_if_missing=False)
    if not draft_version:
        published = await get_published_version(workflow)
        if published:
            _raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Workflow already published",
                detail="No pending changes to publish. Edit the workflow before publishing again.",
                status=400,
            )
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="No draft to publish",
            detail="Cannot publish workflow without a draft version",
            status=400,
        )

    # Validate spec
    spec = WorkflowSpec.model_validate(draft_version.spec)

    # Run full compilation validation (same checks as /runs)
    # Pass organization_id for shared connection validation
    await validate_workflow_spec(user, spec, organization_id=workflow.organization_id)

    # Sync trigger subscriptions
    # pylint: disable=import-outside-toplevel
    from seer.api.workflows.services.triggers import sync_trigger_subscriptions
    await sync_trigger_subscriptions(user, workflow, spec, skip_validation=False)

    # Archive previous release
    previous_release = await get_published_version(workflow)
    if previous_release:
        await WorkflowVersion.filter(id=previous_release.id).update(
            status=WorkflowVersionStatus.ARCHIVED
        )

    # Promote DRAFT to RELEASED
    release_number = await _next_release_number(workflow)
    await WorkflowVersion.filter(id=draft_version.id).update(
        status=WorkflowVersionStatus.RELEASED,
        version_number=release_number,
    )

    # Update workflow timestamp
    await Workflow.filter(id=workflow.id).update(
        updated_at=_now(),
    )

    # Note: No new DRAFT is created here (on-demand creation)

    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership, require_manage=True)
    await _publish_workflow_event(
        workflow,
        event_type=CollaborationEventType.WORKFLOW_PUBLISHED,
        actor=user,
        payload={"release_version_number": release_number},
    )
    return await _workflow_response(workflow)


async def delete_workflow(
    user: User,
    workflow_id: str,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> None:
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership, require_manage=True)
    deleted_workflow_id = workflow.workflow_id
    deleted_workflow_name = workflow.name
    deleted_org_id = workflow.organization_id

    # Cancel active runs to release DB locks
    active_runs = await WorkflowRun.filter(
        workflow=workflow,
        status__in=[WorkflowRunStatus.RUNNING, WorkflowRunStatus.QUEUED, WorkflowRunStatus.INTERRUPTED],
    )
    for run in active_runs:
        run.status = WorkflowRunStatus.CANCELLED
        await run.save(update_fields=["status"])

    # Nullify FK on workflow_runs (constraint dropped in migration 6, but column remains)
    await WorkflowRun.filter(workflow=workflow).update(workflow_id=None)

    # Delete trigger subscriptions before workflow
    await TriggerSubscription.filter(workflow=workflow).delete()

    await workflow.delete()
    await publish_collaboration_event(
        organization_id=deleted_org_id,
        event_type=CollaborationEventType.WORKFLOW_DELETED,
        resource_type="workflow",
        resource_id=deleted_workflow_id,
        actor=user,
        payload={"name": deleted_workflow_name},
    )


async def export_workflow(
    user: User,
    workflow_id: str,
    include_triggers: bool = True,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> Dict[str, Any]:
    """
    Export workflow and optionally triggers as portable JSON.
    """


    # 1. Fetch workflow and draft
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership)
    draft_version = await _get_draft_version(workflow, create_if_missing=False)

    if not draft_version:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="No draft found",
            detail="Workflow has no draft to export",
            status=404,
        )
        return {}  # Unreachable, but satisfies type checker

    # 2. Serialize workflow spec
    spec_dict = draft_version.spec  # Already JSON

    # 3. Fetch triggers from the spec (already embedded)
    triggers_data = spec_dict.get("triggers", []) if include_triggers else []

    # 4. Build export JSON
    return {
        "version": "1.0",
        "workflow": {
            "name": workflow.name,
            "spec": spec_dict,
        },
        "triggers": triggers_data,
        "metadata": {
            "exported_at": _now().isoformat(),
            "exported_by": user.email if hasattr(user, 'email') else None,
            "original_workflow_id": workflow.workflow_id,
            "seer_version": "1.0",
        }
    }


async def _ensure_unique_name(user: User, base_name: str) -> str:
    """Append (1), (2), etc. if name conflicts."""
    name = base_name
    counter = 1

    while await Workflow.filter(user=user, name=name).exists():
        name = f"{base_name} ({counter})"
        counter += 1

    return name


async def import_workflow(
    user: User,
    payload: api_models.WorkflowImportRequest,
) -> api_models.WorkflowResponse:
    """
    Import workflow from exported JSON.
    """
    import_data = payload.import_data

    # 1. Validate schema version
    if import_data.get("version") != "1.0":
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Unsupported import version",
            detail=f"Unsupported import version: {import_data.get('version')}",
            status=400,
        )

    # 2. Validate workflow spec
    spec_payload = import_data["workflow"]["spec"]
    # Backward compatibility: merge triggers array if provided separately.
    if payload.import_triggers and not spec_payload.get("triggers") and import_data.get("triggers"):
        spec_payload = dict(spec_payload)
        spec_payload["triggers"] = import_data["triggers"]

    try:
        spec = WorkflowSpec.model_validate(spec_payload)
    except ValidationError as e:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid workflow spec",
            detail=f"Invalid workflow spec: {e}",
            status=400,
        )
    except KeyError as e:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Missing required field",
            detail=f"Missing required field in import data: {e}",
            status=400,
        )

    # 3. Create new workflow (with optional name override)
    workflow_name = payload.name or import_data["workflow"]["name"]
    workflow_name = await _ensure_unique_name(user, workflow_name)

    workflow = await Workflow.create(
        user=user,
        name=workflow_name,
    )

    # 4. Create draft with spec
    spec_dict = spec.model_dump(mode="json")
    await WorkflowVersion.create(
        workflow=workflow,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec_dict,
        created_by=user,
        updated_by=user,
        spec_hash=_hash_spec(spec_dict),
        version_number=0,
    )

    await _publish_workflow_event(
        workflow,
        event_type=CollaborationEventType.WORKFLOW_CREATED,
        actor=user,
        payload={"name": workflow.name, "source": "import"},
    )

    # 5. Return new workflow
    return await _workflow_response(workflow)

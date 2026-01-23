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
    _ensure_draft_version,
    _get_draft_version,
    _get_workflow,
    _hash_spec,
    _now,
    _raise_problem,
    _spec_to_dict,
    _update_draft_version,
    get_published_version,
)
from seer.database import (
    User,
    Workflow,
    WorkflowVersion,
    WorkflowVersionStatus,
    parse_workflow_public_id,
)
from seer.core.schema.models import WorkflowSpec

# ===== Helper Functions =====


async def _workflow_summary(workflow: Workflow, draft_version: Optional[WorkflowVersion] = None) -> api_models.WorkflowSummary:
    """
    Create a workflow summary.

    If draft_version is not provided, it will be fetched from the database.
    Pass it explicitly when available to avoid extra queries.
    """
    return api_models.WorkflowSummary(
        workflow_id=workflow.workflow_id,
        name=workflow.name,
        created_at=workflow.created_at,
        updated_at=workflow.updated_at,
    )


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


async def create_workflow(user: User, payload: api_models.WorkflowCreateRequest) -> api_models.WorkflowResponse:
    # Workflow limit check moved to UsageLimitMiddleware
    spec_dict = _spec_to_dict(payload.spec)
    workflow = await Workflow.create(
        user=user,
        name=payload.name,
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

    return await _workflow_response(workflow)


async def list_workflows(
    user: User,
    *,
    limit: int = 50,
    cursor: Optional[str] = None,
) -> api_models.WorkflowListResponse:
    limit = max(1, min(limit, 100))
    cursor_pk = _parse_workflow_cursor(cursor)

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


async def get_workflow(user: User, workflow_id: str) -> api_models.WorkflowResponse:
    workflow = await _get_workflow(user, workflow_id)
    return await _workflow_response(workflow)


async def list_workflow_versions(user: User, workflow_id: str) -> api_models.WorkflowVersionListResponse:
    workflow = await _get_workflow(user, workflow_id)
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
) -> api_models.WorkflowResponse:
    workflow = await _get_workflow(user, workflow_id)
    if payload.name is not None:
        workflow.name = payload.name
    await workflow.save()
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
    except Exception as exc:
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
) -> api_models.WorkflowResponse:
    workflow = await _get_workflow(user, workflow_id)

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

    return await _workflow_response(workflow)


async def restore_workflow_version(
    user: User,
    workflow_id: str,
    version_id: int,
    payload: api_models.WorkflowVersionRestoreRequest,
) -> api_models.WorkflowResponse:
    workflow = await _get_workflow(user, workflow_id)
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
) -> api_models.WorkflowResponse:
    workflow = await _get_workflow(user, workflow_id)

    # Get existing DRAFT version (must exist to publish)
    draft_version = await _get_draft_version(workflow, create_if_missing=False)
    if not draft_version:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="No draft to publish",
            detail="Cannot publish workflow without a draft version",
            status=400,
        )

    # Validate spec
    spec = WorkflowSpec.model_validate(draft_version.spec)

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

    workflow = await _get_workflow(user, workflow_id)
    return await _workflow_response(workflow)


async def delete_workflow(user: User, workflow_id: str) -> None:
    workflow = await _get_workflow(user, workflow_id)
    await workflow.delete()


async def export_workflow(
    user: User,
    workflow_id: str,
    include_triggers: bool = True,
) -> Dict[str, Any]:
    """
    Export workflow and optionally triggers as portable JSON.
    """


    # 1. Fetch workflow and draft
    workflow = await _get_workflow(user, workflow_id)
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

    # 5. Return new workflow
    return await _workflow_response(workflow)

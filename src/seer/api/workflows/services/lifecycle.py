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
    _get_workflow,
    _now,
    _raise_problem,
    _spec_to_dict,
)
from seer.database import (
    User,
    Workflow,
    WorkflowDraft,
    WorkflowVersion,
    WorkflowVersionStatus,
    parse_workflow_public_id,
)
from seer.core.schema.models import WorkflowSpec

# ===== Helper Functions =====


def _workflow_summary(workflow: Workflow) -> api_models.WorkflowSummary:
    draft: Optional[WorkflowDraft] = getattr(workflow, "draft", None)
    draft_revision = draft.revision if draft else 0
    return api_models.WorkflowSummary(
        workflow_id=workflow.workflow_id,
        name=workflow.name,
        description=workflow.description,
        draft_revision=draft_revision,
        created_at=workflow.created_at,
        updated_at=workflow.updated_at,
    )


def _serialize_version_summary(
    version: Optional[WorkflowVersion],
) -> Optional[api_models.WorkflowVersionSummary]:
    if not version:
        return None
    return api_models.WorkflowVersionSummary(
        version_id=version.id,
        status=version.status.value if isinstance(version.status, WorkflowVersionStatus) else version.status,
        version_number=version.version_number,
        created_from_draft_revision=version.created_from_draft_revision,
        created_at=version.created_at,
    )


def _serialize_version_list_item(
    version: WorkflowVersion,
    *,
    latest_version_id: Optional[int],
    published_version_id: Optional[int],
) -> api_models.WorkflowVersionListItem:
    summary = _serialize_version_summary(version)
    if summary is None:
        raise RuntimeError("Failed to serialize workflow version")
    return api_models.WorkflowVersionListItem(
        **summary.model_dump(),
        is_latest=version.id == latest_version_id if latest_version_id else False,
        is_published=version.id == published_version_id if published_version_id else False,
    )


async def _recent_version(workflow: Workflow) -> Optional[WorkflowVersion]:
    return await WorkflowVersion.filter(workflow=workflow).order_by("-created_at").first()


async def _workflow_response(workflow: Workflow) -> api_models.WorkflowResponse:
    draft: Optional[WorkflowDraft] = getattr(workflow, "draft", None)
    if draft is None:
        draft = await WorkflowDraft.get_or_none(workflow=workflow)
    if draft is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Missing draft",
            detail="Workflow draft state not initialized",
            status=500,
        )
    raw_spec = draft.spec or {}
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
    published_version_obj: Optional[WorkflowVersion] = getattr(workflow, "published_version", None)
    if published_version_obj and not isinstance(published_version_obj, WorkflowVersion):
        published_version_obj = None
    latest_version = await _recent_version(workflow)
    return api_models.WorkflowResponse(
        workflow_id=workflow.workflow_id,
        name=workflow.name,
        description=workflow.description,
        draft_revision=draft.revision,
        created_at=workflow.created_at,
        updated_at=workflow.updated_at,
        spec=spec,
        tags=list(workflow.tags or []),
        meta=api_models.WorkflowMeta(last_compile_ok=(workflow.meta or {}).get("last_compile_ok", False)),
        published_version=_serialize_version_summary(published_version_obj),
        latest_version=_serialize_version_summary(latest_version),
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
        description=payload.description,
        tags=list(payload.tags or []),
        meta={"last_compile_ok": False},
    )
    await WorkflowDraft.create(
        workflow=workflow,
        spec=spec_dict,
        revision=1,
        updated_by=user,
    )
    await workflow.fetch_related("draft")

    return await _workflow_response(workflow)


async def list_workflows(
    user: User,
    *,
    limit: int = 50,
    cursor: Optional[str] = None,
) -> api_models.WorkflowListResponse:
    limit = max(1, min(limit, 100))
    cursor_pk = _parse_workflow_cursor(cursor)

    query = Workflow.filter(user=user).prefetch_related("draft")
    if cursor_pk:
        query = query.filter(id__lt=cursor_pk)

    records = await query.order_by("-id").limit(limit + 1)
    items = [_workflow_summary(record) for record in records[:limit]]
    next_cursor = items[-1].workflow_id if len(records) > limit and items else None
    return api_models.WorkflowListResponse(items=items, next_cursor=next_cursor)


async def get_workflow(user: User, workflow_id: str) -> api_models.WorkflowResponse:
    workflow = await _get_workflow(user, workflow_id)
    return await _workflow_response(workflow)


async def list_workflow_versions(user: User, workflow_id: str) -> api_models.WorkflowVersionListResponse:
    workflow = await _get_workflow(user, workflow_id)
    draft = workflow.draft or await WorkflowDraft.get(workflow=workflow)
    versions = (
        await WorkflowVersion.filter(workflow=workflow)
        .order_by("-created_at")
        .all()
    )
    published_version_obj: Optional[WorkflowVersion] = getattr(workflow, "published_version", None)
    published_version_id = published_version_obj.id if isinstance(published_version_obj, WorkflowVersion) else None
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
        draft_revision=draft.revision,
        versions=items,
        latest_version_id=latest_version_id,
        published_version_id=published_version_id,
    )


async def update_workflow(
    user: User,
    workflow_id: str,
    payload: api_models.WorkflowUpdateRequest,
) -> api_models.WorkflowResponse:
    workflow = await _get_workflow(user, workflow_id)
    if payload.name is not None:
        workflow.name = payload.name
    if payload.description is not None:
        workflow.description = payload.description
    if payload.tags is not None:
        workflow.tags = list(payload.tags)
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

    draft = await WorkflowDraft.get(workflow=workflow)
    draft.spec = _spec_to_dict(spec)
    draft.revision += 1
    draft.updated_by = user
    await draft.save()
    await Workflow.filter(id=workflow.id).update(updated_at=_now())
    await workflow.fetch_related("draft")
    return await _workflow_response(workflow)


async def patch_workflow_draft(
    user: User,
    workflow_id: str,
    payload: api_models.WorkflowDraftPatchRequest,
) -> api_models.WorkflowResponse:
    workflow = await _get_workflow(user, workflow_id)
    draft = workflow.draft or await WorkflowDraft.get(workflow=workflow)
    # TODO: discuss if want to have revision check here
    # if payload.base_revision is not None and payload.base_revision != draft.revision:
    #     _raise_problem(
    #         type_uri=VALIDATION_PROBLEM,
    #         title="Draft revision mismatch",
    #         detail="Draft has changed since last fetch",
    #         status=409,
    #     )

    # For now, we allow patching the draft without checking the revision
    draft.revision = max(draft.revision, payload.base_revision or 0) + 1

    spec = payload.spec
    draft.spec = _spec_to_dict(spec)
    draft.updated_by = user
    await draft.save()
    await Workflow.filter(id=workflow.id).update(updated_at=_now())
    await workflow.fetch_related("draft")
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
    draft = workflow.draft or await WorkflowDraft.get(workflow=workflow)
    if payload.base_revision is not None and payload.base_revision != draft.revision:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Draft revision mismatch",
            detail="Draft has changed since last fetch",
            status=409,
        )
    draft.spec = json.loads(json.dumps(version.spec or {}))
    draft.revision += 1
    draft.updated_by = user
    await draft.save()
    await Workflow.filter(id=workflow.id).update(updated_at=_now())
    await workflow.fetch_related("draft", "published_version")
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
    version = await _ensure_draft_version(workflow, user)

    previous_release = getattr(workflow, "published_version", None)
    if previous_release and isinstance(previous_release, WorkflowVersion):
        await WorkflowVersion.filter(id=previous_release.id).update(status=WorkflowVersionStatus.ARCHIVED)

    release_number = await _next_release_number(workflow)
    await WorkflowVersion.filter(id=version.id).update(
        status=WorkflowVersionStatus.RELEASED,
        version_number=release_number,
    )
    workflow.published_version = version
    await Workflow.filter(id=workflow.id).update(
        published_version_id=version.id,
        updated_at=_now(),
    )
    # Refresh status for response
    version.status = WorkflowVersionStatus.RELEASED
    version.version_number = release_number

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
    draft = workflow.draft or await WorkflowDraft.get_or_none(workflow=workflow)

    if not draft:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="No draft found",
            detail="Workflow has no draft to export",
            status=404,
        )

    # 2. Serialize workflow spec
    spec_dict = draft.spec  # Already JSON

    # 3. Fetch triggers from the spec (already embedded)
    triggers_data = spec_dict.get("triggers", []) if include_triggers else []

    # 4. Build export JSON
    return {
        "version": "1.0",
        "workflow": {
            "name": workflow.name,
            "description": workflow.description,
            "tags": workflow.tags or [],
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
        description=import_data["workflow"].get("description"),
        tags=import_data["workflow"].get("tags", []),
        meta={"last_compile_ok": False},
    )

    # 4. Create draft with spec
    await WorkflowDraft.create(
        workflow=workflow,
        spec=spec.model_dump(mode="json"),
        revision=1,
        updated_by=user,
    )

    # 5. Return new workflow
    await workflow.fetch_related("draft")
    return await _workflow_response(workflow)

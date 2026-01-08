"""Workflow CRUD operations and version management."""

from __future__ import annotations

import json
from typing import Optional

from api.workflows import models as api_models
from api.workflows.services.shared import (
    VALIDATION_PROBLEM,
    _now,
    _raise_problem,
    _spec_to_dict,
    _get_workflow
)

from shared.database.workflow_models import (
    User,
    Workflow,
    WorkflowDraft,
    WorkflowVersion,
    WorkflowVersionStatus,
    parse_workflow_public_id,
)
from tortoise.exceptions import DoesNotExist
from workflow_compiler.schema.models import WorkflowSpec
from typing import Dict, Any

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
    spec = WorkflowSpec.model_validate(draft.spec)
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


async def create_workflow(user: User, payload: api_models.WorkflowCreateRequest) -> api_models.WorkflowResponse:
    spec_dict = _spec_to_dict(payload.spec)
    workflow = await Workflow.create(
        user=user,
        name=payload.name,
        description=payload.description,
        tags=list(payload.tags or []),
        meta={"last_compile_ok": False},
    )
    draft = await WorkflowDraft.create(
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
    if payload.base_revision is not None and payload.base_revision != draft.revision:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Draft revision mismatch",
            detail="Draft has changed since last fetch",
            status=409,
        )
    spec = payload.spec
    draft.spec = _spec_to_dict(spec)
    draft.revision += 1
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
    try:
        version = await WorkflowVersion.get(id=payload.version_id, workflow=workflow)
    except DoesNotExist:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Version not found",
            detail=f"Version '{payload.version_id}' does not belong to workflow '{workflow_id}'",
            status=404,
        )

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

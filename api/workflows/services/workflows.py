"""Workflow CRUD operations and version management."""

from __future__ import annotations

import json
from typing import Optional

from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.workflows import models as api_models
from api.workflows.services._shared import (
    VALIDATION_PROBLEM,
    _now,
    _raise_problem,
    _spec_to_dict,
)
from shared.database.base import async_session_maker
from shared.database.models import (
    User,
    Workflow,
    WorkflowDraft,
    WorkflowVersion,
    WorkflowVersionStatus,
    parse_workflow_public_id,
)
from workflow_compiler.schema.models import WorkflowSpec

# ===== Helper Functions =====


def _workflow_summary(workflow: Workflow) -> api_models.WorkflowSummary:
    """Serialize workflow to summary response."""
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
    """Serialize workflow version to summary response."""
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
    """Serialize workflow version to list item with metadata flags."""
    summary = _serialize_version_summary(version)
    if summary is None:
        raise RuntimeError("Failed to serialize workflow version")
    return api_models.WorkflowVersionListItem(
        **summary.model_dump(),
        is_latest=version.id == latest_version_id if latest_version_id else False,
        is_published=version.id == published_version_id if published_version_id else False,
    )


async def _recent_version(workflow: Workflow, session: AsyncSession) -> Optional[WorkflowVersion]:
    """Get the most recently created version of a workflow."""
    stmt = select(WorkflowVersion).where(WorkflowVersion.workflow_id == workflow.id).order_by(WorkflowVersion.created_at.desc())
    result = await session.execute(stmt)
    return result.scalars().first()


async def _workflow_response(workflow: Workflow, session: AsyncSession) -> api_models.WorkflowResponse:
    """Build complete workflow response with spec and version metadata."""
    draft: Optional[WorkflowDraft] = getattr(workflow, "draft", None)
    if draft is None:
        stmt = select(WorkflowDraft).where(WorkflowDraft.workflow_id == workflow.id)
        result = await session.execute(stmt)
        draft = result.scalar_one_or_none()
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
    latest_version = await _recent_version(workflow, session)
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
    """Parse pagination cursor to workflow internal ID."""
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


async def _get_workflow(user: User, workflow_id: str, session: AsyncSession) -> Workflow:
    """
    Get workflow by public ID for user.

    Exported for use by other service modules (execution, triggers).
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
    stmt = (
        select(Workflow)
        .where(Workflow.id == pk, Workflow.user_id == user.id)
        .options(selectinload(Workflow.draft), selectinload(Workflow.published_version))
    )
    result = await session.execute(stmt)
    workflow = result.scalar_one_or_none()
    if workflow is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Workflow not found",
            detail=f"Workflow '{workflow_id}' not found",
            status=404,
        )
    return workflow


async def _next_release_number(workflow: Workflow, session: AsyncSession) -> int:
    """Calculate next release version number for workflow."""
    stmt = (
        select(WorkflowVersion)
        .where(WorkflowVersion.workflow_id == workflow.id, WorkflowVersion.version_number.isnot(None))
        .order_by(WorkflowVersion.version_number.desc())
    )
    result = await session.execute(stmt)
    latest = result.scalars().first()
    if latest is None or latest.version_number is None:
        return 1
    return latest.version_number + 1


# ===== Public API Functions =====


async def create_workflow(user: User, payload: api_models.WorkflowCreateRequest) -> api_models.WorkflowResponse:
    """Create a new workflow with initial draft."""
    spec_dict = _spec_to_dict(payload.spec)
    async with async_session_maker() as session:
        workflow = Workflow(
            user_id=user.id,
            name=payload.name,
            description=payload.description,
            tags=list(payload.tags or []),
            meta={"last_compile_ok": False},
        )
        session.add(workflow)
        await session.commit()
        await session.refresh(workflow)

        draft = WorkflowDraft(
            workflow_id=workflow.id,
            spec=spec_dict,
            revision=1,
            updated_by_id=user.id,
        )
        session.add(draft)
        await session.commit()
        await session.refresh(draft)

        # Attach draft to workflow for response generation
        workflow.draft = draft
        return await _workflow_response(workflow, session)


async def list_workflows(
    user: User,
    *,
    limit: int = 50,
    cursor: Optional[str] = None,
) -> api_models.WorkflowListResponse:
    """List workflows for user with pagination."""
    limit = max(1, min(limit, 100))
    cursor_pk = _parse_workflow_cursor(cursor)

    async with async_session_maker() as session:
        stmt = select(Workflow).where(Workflow.user_id == user.id).options(selectinload(Workflow.draft))
        if cursor_pk:
            stmt = stmt.where(Workflow.id < cursor_pk)

        stmt = stmt.order_by(Workflow.id.desc()).limit(limit + 1)
        result = await session.execute(stmt)
        records = result.scalars().all()
        items = [_workflow_summary(record) for record in records[:limit]]
        next_cursor = items[-1].workflow_id if len(records) > limit and items else None
        return api_models.WorkflowListResponse(items=items, next_cursor=next_cursor)


async def get_workflow(user: User, workflow_id: str) -> api_models.WorkflowResponse:
    """Get a specific workflow by ID."""
    async with async_session_maker() as session:
        workflow = await _get_workflow(user, workflow_id, session)
        return await _workflow_response(workflow, session)


async def list_workflow_versions(user: User, workflow_id: str) -> api_models.WorkflowVersionListResponse:
    """List all versions of a workflow."""
    async with async_session_maker() as session:
        workflow = await _get_workflow(user, workflow_id, session)
        draft = workflow.draft
        if not draft:
            stmt = select(WorkflowDraft).where(WorkflowDraft.workflow_id == workflow.id)
            result = await session.execute(stmt)
            draft = result.scalar_one()

        stmt = select(WorkflowVersion).where(WorkflowVersion.workflow_id == workflow.id).order_by(WorkflowVersion.created_at.desc())
        result = await session.execute(stmt)
        versions = result.scalars().all()

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
    """Update workflow metadata (name, description, tags)."""
    async with async_session_maker() as session:
        workflow = await _get_workflow(user, workflow_id, session)
        if payload.name is not None:
            workflow.name = payload.name
        if payload.description is not None:
            workflow.description = payload.description
        if payload.tags is not None:
            workflow.tags = list(payload.tags)
        session.add(workflow)
        await session.commit()
        await session.refresh(workflow)
        return await _workflow_response(workflow, session)


async def apply_workflow_from_spec(
    user: User,
    workflow_id: str,
    spec_payload: dict,
) -> api_models.WorkflowResponse:
    """
    Replace an existing workflow's spec with a validated WorkflowSpec payload.
    """
    async with async_session_maker() as session:
        workflow = await _get_workflow(user, workflow_id, session)
        try:
            spec = WorkflowSpec.model_validate(spec_payload)
        except Exception as exc:
            _raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Invalid workflow spec",
                detail=str(exc),
                status=400,
            )

        stmt = select(WorkflowDraft).where(WorkflowDraft.workflow_id == workflow.id)
        result = await session.execute(stmt)
        draft = result.scalar_one()

        draft.spec = _spec_to_dict(spec)
        draft.revision += 1
        draft.updated_by_id = user.id
        session.add(draft)
        await session.commit()

        workflow.updated_at = _now()
        session.add(workflow)
        await session.commit()
        await session.refresh(workflow)

        # Reload draft for response
        workflow.draft = draft
        return await _workflow_response(workflow, session)


async def patch_workflow_draft(
    user: User,
    workflow_id: str,
    payload: api_models.WorkflowDraftPatchRequest,
) -> api_models.WorkflowResponse:
    """Patch workflow draft spec with optimistic locking."""
    async with async_session_maker() as session:
        workflow = await _get_workflow(user, workflow_id, session)
        draft = workflow.draft
        if not draft:
            stmt = select(WorkflowDraft).where(WorkflowDraft.workflow_id == workflow.id)
            result = await session.execute(stmt)
            draft = result.scalar_one()

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
        draft.updated_by_id = user.id
        session.add(draft)
        await session.commit()

        workflow.updated_at = _now()
        session.add(workflow)
        await session.commit()
        await session.refresh(workflow)

        # Reload draft for response
        workflow.draft = draft
        return await _workflow_response(workflow, session)


async def restore_workflow_version(
    user: User,
    workflow_id: str,
    version_id: int,
    payload: api_models.WorkflowVersionRestoreRequest,
) -> api_models.WorkflowResponse:
    """Restore workflow draft from a specific version."""
    async with async_session_maker() as session:
        workflow = await _get_workflow(user, workflow_id, session)
        stmt = select(WorkflowVersion).where(WorkflowVersion.id == version_id, WorkflowVersion.workflow_id == workflow.id)
        result = await session.execute(stmt)
        version = result.scalar_one_or_none()
        if version is None:
            _raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Version not found",
                detail=f"Version '{version_id}' does not belong to workflow '{workflow_id}'",
                status=404,
            )

        draft = workflow.draft
        if not draft:
            stmt = select(WorkflowDraft).where(WorkflowDraft.workflow_id == workflow.id)
            result = await session.execute(stmt)
            draft = result.scalar_one()

        if payload.base_revision is not None and payload.base_revision != draft.revision:
            _raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Draft revision mismatch",
                detail="Draft has changed since last fetch",
                status=409,
            )
        draft.spec = json.loads(json.dumps(version.spec or {}))
        draft.revision += 1
        draft.updated_by_id = user.id
        session.add(draft)
        await session.commit()

        workflow.updated_at = _now()
        session.add(workflow)
        await session.commit()

        # Reload for response
        await session.refresh(workflow)
        stmt = (
            select(Workflow)
            .where(Workflow.id == workflow.id)
            .options(selectinload(Workflow.draft), selectinload(Workflow.published_version))
        )
        result = await session.execute(stmt)
        workflow = result.scalar_one()
        return await _workflow_response(workflow, session)


async def publish_workflow(
    user: User,
    workflow_id: str,
    payload: api_models.WorkflowPublishRequest,
) -> api_models.WorkflowResponse:
    """Publish a workflow version as the latest release."""
    async with async_session_maker() as session:
        workflow = await _get_workflow(user, workflow_id, session)
        stmt = select(WorkflowVersion).where(WorkflowVersion.id == payload.version_id, WorkflowVersion.workflow_id == workflow.id)
        result = await session.execute(stmt)
        version = result.scalar_one_or_none()
        if version is None:
            _raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Version not found",
                detail=f"Version '{payload.version_id}' does not belong to workflow '{workflow_id}'",
                status=404,
            )

        previous_release = getattr(workflow, "published_version", None)
        if previous_release and isinstance(previous_release, WorkflowVersion):
            previous_release.status = WorkflowVersionStatus.ARCHIVED
            session.add(previous_release)
            await session.commit()

        release_number = await _next_release_number(workflow, session)
        version.status = WorkflowVersionStatus.RELEASED
        version.version_number = release_number
        session.add(version)
        await session.commit()

        workflow.published_version_id = version.id
        workflow.updated_at = _now()
        session.add(workflow)
        await session.commit()

        # Reload workflow for response
        workflow = await _get_workflow(user, workflow_id, session)
        return await _workflow_response(workflow, session)


async def delete_workflow(user: User, workflow_id: str) -> None:
    """Delete a workflow and all associated data."""
    async with async_session_maker() as session:
        workflow = await _get_workflow(user, workflow_id, session)
        await session.delete(workflow)
        await session.commit()

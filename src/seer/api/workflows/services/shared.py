from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from seer.api.core.errors import VALIDATION_PROBLEM
from seer.api.core.errors import raise_problem as _raise_problem
from seer.database import (
    User,
    Workflow,
    WorkflowRun,
    WorkflowVersion,
    WorkflowVersionStatus,
    parse_workflow_public_id,
)
from seer.core.schema.models import WorkflowSpec


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


async def _ensure_draft_version(
    workflow: Workflow,
    user: User,
    skip_validation: bool = False
) -> WorkflowVersion:
    """
    Get the existing DRAFT version without creating a new one.
    This function is kept for backward compatibility during transition.
    """
    draft_version = await _get_draft_version(workflow, create_if_missing=True, user=user)

    if not draft_version:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="No draft version",
            detail="Workflow has no draft version",
            status=500,
        )

    spec = WorkflowSpec.model_validate(draft_version.spec or {})

    # Sync trigger subscriptions declared in the spec so polling/webhooks stay in sync.
    # pylint: disable=import-outside-toplevel
    from seer.api.workflows.services.triggers import sync_trigger_subscriptions

    await sync_trigger_subscriptions(user, workflow, spec, skip_validation=skip_validation)

    return draft_version



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

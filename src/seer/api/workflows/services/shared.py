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
    WorkflowDraft,
    WorkflowRun,
    WorkflowVersion,
    WorkflowVersionStatus,
    parse_workflow_public_id,
)
from seer.core.schema.models import WorkflowSpec


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _spec_to_dict(spec: WorkflowSpec) -> Dict[str, Any]:
    return spec.model_dump(mode="json")


def _hash_spec(spec_dict: Dict[str, Any]) -> str:
    serialized = json.dumps(spec_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


async def _ensure_draft_version(
    workflow: Workflow,
    user: User,
    skip_validation: bool = False
) -> WorkflowVersion:
    draft = await WorkflowDraft.get(workflow=workflow)
    spec = WorkflowSpec.model_validate(draft.spec or {})
    spec_dict = spec.model_dump(mode="json")
    spec_hash = _hash_spec(spec_dict)
    # Sync trigger subscriptions declared in the spec so polling/webhooks stay in sync.
    # pylint: disable=import-outside-toplevel
    from seer.api.workflows.services.triggers import sync_trigger_subscriptions

    await sync_trigger_subscriptions(user, workflow, spec, skip_validation=skip_validation)
    existing = (
        await WorkflowVersion.filter(
            workflow=workflow,
            spec_hash=spec_hash,
            status=WorkflowVersionStatus.DRAFT,
            created_from_draft_revision=draft.revision,
        )
        .order_by("-created_at")
        .first()
    )
    if existing:
        return existing
    latest_version = (
        await WorkflowVersion.filter(workflow=workflow).order_by("-version_number").first()
    )
    return await WorkflowVersion.create(
        workflow=workflow,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec_dict,
        created_from_draft_revision=draft.revision,
        created_by=user,
        manifest=None,
        spec_hash=spec_hash,
        version_number=(latest_version.version_number + 1) if latest_version else 0,
    )



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
    workflow = await Workflow.filter(id=pk, user=user).prefetch_related("draft", "published_version").first()
    if workflow is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Workflow not found",
            detail=f"Workflow '{workflow_id}' not found",
            status=404,
        )
    return workflow

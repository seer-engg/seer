"""
Approval API — public callback endpoints for HITL workflow approvals.

These endpoints are called from the approval web page (or deep link)
to approve/reject a paused workflow run. No auth required since the
run_id acts as an unguessable token.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from seer.database import WorkflowRun, WorkflowRunStatus
from seer.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/approvals", tags=["approvals"])


class ApprovalRequest(BaseModel):
    """Approve or reject a paused workflow run."""
    approved: bool
    message: Optional[str] = None


class ApprovalResponse(BaseModel):
    run_id: str
    status: str
    message: str


@router.get("/{run_id}")
async def get_approval_details(run_id: str) -> Dict[str, Any]:
    """
    Get cart/interrupt details for an approval page.

    Returns the pending interrupt data so the approval UI can render
    cart contents, prices, etc.
    """
    from seer.database.workflow_models import parse_run_public_id  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import

    try:
        run_pk = parse_run_public_id(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id}") from exc

    run = await WorkflowRun.get_or_none(id=run_pk)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status != WorkflowRunStatus.INTERRUPTED:
        return {
            "run_id": run.run_id,
            "status": run.status.value,
            "interrupt_data": None,
            "message": "This run is not awaiting approval",
        }

    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "interrupt_data": run.pending_interrupt_data,
    }


@router.post("/{run_id}", response_model=ApprovalResponse)
async def submit_approval(run_id: str, payload: ApprovalRequest) -> ApprovalResponse:
    """
    Approve or reject a paused workflow. Resumes execution.

    This is the callback endpoint hit by the approval web page or
    Ntfy action button.
    """
    from seer.database.workflow_models import parse_run_public_id  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import

    try:
        run_pk = parse_run_public_id(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id}") from exc

    run = await WorkflowRun.get_or_none(id=run_pk)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status != WorkflowRunStatus.INTERRUPTED:
        raise HTTPException(
            status_code=409,
            detail=f"Run is not awaiting approval (status: {run.status.value})",
        )

    await run.fetch_related("user", "workflow")

    # Resume the workflow with approval/rejection response
    from seer.services.workflows.execution import resume_workflow_run  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import

    responses = {
        "approved": payload.approved,
        "message": payload.message or ("Approved" if payload.approved else "Rejected"),
    }

    try:
        await resume_workflow_run(run.user, run.run_id, responses)
    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: Return structured error
        logger.exception("Approval resume failed for run '%s'", run_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    action = "approved" if payload.approved else "rejected"
    logger.info("Run '%s' %s via approval endpoint", run_id, action)

    return ApprovalResponse(
        run_id=run.run_id,
        status=action,
        message=f"Cart {action} successfully",
    )

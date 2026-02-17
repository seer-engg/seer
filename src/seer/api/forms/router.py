"""Form API endpoints for public form hosting."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, status

from seer.api.forms.validation import validate_form_data
from seer.api.webhooks.services import handle_generic_webhook
from seer.database import TriggerSubscription, WorkflowRun, WorkflowRunStatus
from seer.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/forms", tags=["forms"])


@router.get("/resolve/{suffix}")
async def resolve_form(suffix: str) -> Dict[str, Any]:
    """
    Resolve form configuration by suffix for public rendering.

    Args:
        suffix: Form URL suffix (e.g., "contact-form")

    Returns:
        Form configuration with fields, title, description, and styling

    Raises:
        HTTPException: If form not found or not enabled
    """
    subscription = await TriggerSubscription.filter(
        form_suffix=suffix,
        enabled=True,
    ).first()

    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Form not found",
        )

    form_config = subscription.form_config or {}

    response: Dict[str, Any] = {
        "form_id": subscription.id,
        "title": form_config.get("title", "Form"),
        "description": form_config.get("description"),
        "fields": subscription.form_fields or [],
        "submit_button_text": form_config.get("submitButtonText", "Submit"),
        "success_message": form_config.get("successMessage", "Thank you for your submission!"),
        "styling": form_config.get("styling", {}),
    }

    # Include HITL display items if present (for HITL forms)
    hitl_display = form_config.get("_hitl_display")
    if hitl_display:
        response["display_items"] = hitl_display

    return response


@router.post("/submit/{suffix}")
async def submit_form(suffix: str, request: Request) -> Dict[str, Any]:
    """
    Validate and submit form data.

    Args:
        suffix: Form URL suffix
        request: FastAPI request containing form data

    Returns:
        Success response with event ID

    Raises:
        HTTPException: If form not found, validation fails, or submission fails
    """
    try:
        # Find the subscription by suffix
        subscription = await TriggerSubscription.filter(
            form_suffix=suffix,
            enabled=True,
        ).first()

        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Form not found",
            )

        # Parse request data
        try:
            data = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON data",
            ) from exc

        # Validate form data
        form_fields = subscription.form_fields or []
        validation_errors = validate_form_data(data, form_fields)

        if validation_errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "errors": validation_errors,
                    "message": "Form validation failed",
                },
            )

        # Check if this is an HITL form (has _hitl_run_id in form_config)
        form_config = subscription.form_config or {}
        hitl_run_id = form_config.get("_hitl_run_id")

        if hitl_run_id:
            # This is an HITL form - resume the workflow instead of triggering new run
            return await _handle_hitl_form_submission(subscription, data, hitl_run_id)

        # Regular form - trigger webhook to start new workflow run
        # Skip secret verification — form submissions are public endpoints
        event = await handle_generic_webhook(
            subscription.id,
            payload=data,
            headers=dict(request.headers),
            secret=None,
            provider_event_id=None,
            skip_secret_verification=True,
        )

        return {
            "ok": True,
            "event_id": event.id if event else None,
            "message": "Form submitted successfully",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error submitting form: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit form",
        ) from exc


async def _handle_hitl_form_submission(
    subscription: TriggerSubscription,
    data: Dict[str, Any],
    run_id: str,
) -> Dict[str, Any]:
    """
    Handle HITL form submission by resuming the interrupted workflow.

    HITL forms are one-time use - after successful submission, the form is disabled
    to prevent duplicate workflow resumes.

    Args:
        subscription: The TriggerSubscription for the HITL form
        data: The form submission data (user responses)
        run_id: The public run ID (run_XXX format) from form_config._hitl_run_id

    Returns:
        Success response dict

    Raises:
        HTTPException: If workflow run not found or not in INTERRUPTED state
    """
    from seer.database.workflow_models import parse_run_public_id  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import
    from seer.services.workflows.execution import resume_workflow_run  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import

    logger.info(
        "Processing HITL form submission for run '%s'",
        run_id,
        extra={"run_id": run_id, "subscription_id": subscription.id},
    )

    # Parse and fetch the workflow run
    try:
        run_pk = parse_run_public_id(run_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid run_id format: {run_id}",
        ) from exc

    run = await WorkflowRun.get_or_none(id=run_pk)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow run not found",
        )

    # Check run status
    if run.status != WorkflowRunStatus.INTERRUPTED:
        if run.status in (WorkflowRunStatus.SUCCEEDED, WorkflowRunStatus.FAILED):
            # Workflow already completed - return success but note it was already processed
            logger.warning(
                "HITL form submitted for already-completed run '%s' (status: %s)",
                run_id,
                run.status,
                extra={"run_id": run_id, "status": run.status},
            )
            return {
                "ok": True,
                "message": "This workflow has already been completed.",
                "already_completed": True,
            }
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow is not waiting for input (status: {run.status})",
        )

    # Fetch related user for resume_workflow_run
    await subscription.fetch_related("user")

    # Resume the workflow with form data as responses
    try:
        await resume_workflow_run(
            user=subscription.user,
            run_id=run_id,
            responses=data,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to resume workflow from HITL form",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resume workflow",
        ) from exc

    # Disable the form (one-time use) to prevent duplicate submissions
    await TriggerSubscription.filter(id=subscription.id).update(enabled=False)

    logger.info(
        "HITL form successfully resumed workflow '%s'",
        run_id,
        extra={"run_id": run_id},
    )

    form_config = subscription.form_config or {}
    success_message = form_config.get(
        "successMessage",
        "Your response has been recorded. The workflow will continue."
    )

    return {
        "ok": True,
        "message": success_message,
        "workflow_resumed": True,
    }

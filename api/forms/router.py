"""Public form submission endpoints for Form Trigger."""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from api.triggers import services as trigger_services

router = APIRouter(prefix="/v1/forms", tags=["forms"])


@router.post("/{subscription_id}/submit", status_code=status.HTTP_202_ACCEPTED)
async def submit_form(
    subscription_id: int,
    request: Request,
):
    """
    Public endpoint for form submission that triggers a workflow.

    This endpoint is called when a user submits a public form created
    via a Form Trigger. The form data is passed to the workflow as inputs.

    No authentication required - this is a public endpoint for external users.
    """
    # Parse form data (could be JSON or form-encoded)
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        form_data = await request.json()
    elif "application/x-www-form-urlencoded" in content_type or \
            "multipart/form-data" in content_type:
        form_obj = await request.form()
        form_data = dict(form_obj)
    else:
        # Default to JSON
        form_data = await request.json()

    event = await trigger_services.handle_form_submission(
        subscription_id=subscription_id,
        form_data=form_data,
        headers=request.headers,
        provider_event_id=None,  # Could use request ID or generate UUID
    )

    return {
        "ok": True,
        "event_id": event.id,
        "message": "Form submitted successfully. Your workflow has been triggered.",
    }


@router.get("/{subscription_id}")
async def get_form_info(subscription_id: int):
    """
    Get form information for rendering the public form.

    Returns form fields, workflow name, and other metadata needed
    to display the form to end users.
    """
    form_info = await trigger_services.get_form_trigger_info(subscription_id)
    return form_info


__all__ = ["router"]

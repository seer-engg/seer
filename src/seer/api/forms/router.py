"""Form API endpoints for public form hosting."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, status
from tortoise.exceptions import DoesNotExist

from seer.api.forms.validation import validate_form_data
from seer.api.webhooks.services import handle_generic_webhook
from seer.database import TriggerSubscription
from seer.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/forms", tags=["forms"])


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
    try:
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

        return {
            "form_id": subscription.id,
            "title": form_config.get("title", "Form"),
            "description": form_config.get("description"),
            "fields": subscription.form_fields or [],
            "submit_button_text": form_config.get("submitButtonText", "Submit"),
            "success_message": form_config.get("successMessage", "Thank you for your submission!"),
            "styling": form_config.get("styling", {}),
        }

    except DoesNotExist as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Form not found",
        ) from exc
    except Exception as exc:
        logger.error("Error resolving form: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load form",
        ) from exc


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

        # Trigger webhook with form data
        # The webhook handler will process this like a normal webhook event
        event = await handle_generic_webhook(
            subscription.id,
            payload=data,
            headers=dict(request.headers),
            secret=None,
            provider_event_id=None,
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
        )

"""Form data validation logic."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def validate_email(value: str) -> bool:
    """Validate email format."""
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(email_pattern, value))


def validate_url(value: str) -> bool:
    """Validate URL format."""
    url_pattern = (
        r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\."
        r"[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"
    )
    return bool(re.match(url_pattern, value))


def _validate_field_value(value: Any, field_type: str, display_label: str) -> List[str]:
    if field_type == "email":
        if isinstance(value, str) and value and not validate_email(value):
            return [f"{display_label} must be a valid email address"]
    elif field_type == "url":
        if isinstance(value, str) and value and not validate_url(value):
            return [f"{display_label} must be a valid URL"]
    elif field_type == "number":
        try:
            float(value)
        except (ValueError, TypeError):
            return [f"{display_label} must be a valid number"]
    return []


def validate_form_data(
    data: Dict[str, Any],
    form_fields: List[Dict[str, Any]],
) -> List[str]:
    """
    Validate submitted form data against field definitions.

    Args:
        data: Form submission data
        form_fields: Field configurations from TriggerSubscription

    Returns:
        List of error messages (empty if valid)
    """
    errors: List[str] = []

    for field in form_fields:
        field_name = field.get("name")
        display_label = field.get("displayLabel", field_name)
        required = field.get("required", False)
        field_type = field.get("type", "text")

        if required and (field_name not in data or not data[field_name]):
            errors.append(f"{display_label} is required")
            continue

        if field_name not in data or data[field_name] is None:
            continue

        errors.extend(_validate_field_value(data[field_name], field_type, display_label))

    return errors

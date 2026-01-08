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
    url_pattern = r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"
    return bool(re.match(url_pattern, value))


def validate_form_data(  # noqa: C901, pylint: disable=too-complex
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

        # Check required fields
        if required and (field_name not in data or not data[field_name]):
            errors.append(f"{display_label} is required")
            continue

        # Skip validation if field is not present and not required
        if field_name not in data or data[field_name] is None:
            continue

        value = data[field_name]

        # Type-specific validation
        if field_type == "email":
            if isinstance(value, str) and value and not validate_email(value):
                errors.append(f"{display_label} must be a valid email address")
        elif field_type == "url":
            if isinstance(value, str) and value and not validate_url(value):
                errors.append(f"{display_label} must be a valid URL")
        elif field_type == "number":
            try:
                float(value)
            except (ValueError, TypeError):
                errors.append(f"{display_label} must be a valid number")

    return errors

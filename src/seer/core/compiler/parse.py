"""
Stage 1 — Parse JSON into a strongly typed WorkflowSpec.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from seer.core.errors import ValidationPhaseError
from seer.core.schema.models import WorkflowSpec


def parse_workflow_spec(payload: Any) -> WorkflowSpec:
    """
    Accepts either a JSON string or an object compatible with the WorkflowSpec
    definition and returns a validated WorkflowSpec instance.
    """

    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValidationPhaseError(f"Invalid workflow JSON payload: {exc}") from exc
    elif isinstance(payload, Mapping):
        data = payload
    else:
        raise ValidationPhaseError(
            f"Unsupported payload type {type(payload).__name__}; expected str or Mapping"
        )

    try:
        return WorkflowSpec.model_validate(data)
    except Exception as exc:  # Pydantic raises ValidationError
        raise ValidationPhaseError(f"Workflow spec validation failed: {exc}") from exc

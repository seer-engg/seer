"""
Auto-fix trigger event_schemas by replacing with canonical schemas from the registry.

This module provides functions to detect and fix incorrect trigger event_schemas
in workflow specs, ensuring they match the canonical schemas defined in the
trigger registry.

Used by both Nexus agent tools and MCP tools for workflow validation.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional

from seer.core.registry.trigger_registry import trigger_registry


def _schemas_differ(spec_schema: Optional[Dict], canonical_schema: Dict) -> bool:
    """Check if spec schema differs from canonical and needs replacement."""
    if not spec_schema:  # None or empty dict
        return True
    return spec_schema != canonical_schema


def _extract_available_fields(canonical_schema: Dict) -> Dict[str, List[str]]:
    """Extract available field names from canonical schema for LLM feedback."""
    envelope_fields = list(canonical_schema.get("properties", {}).keys())
    data_schema = canonical_schema.get("properties", {}).get("data", {})
    data_fields = list(data_schema.get("properties", {}).keys())
    return {"envelope": envelope_fields, "data": data_fields}


def _build_example_expressions(trigger_id: str, canonical_schema: Dict) -> List[str]:
    """Build example expressions for LLM feedback."""
    examples = []
    data_schema = canonical_schema.get("properties", {}).get("data", {})
    data_props = data_schema.get("properties", {})

    for field in list(data_props.keys())[:3]:
        nested = data_props.get(field, {})
        if nested.get("type") == "object" and nested.get("properties"):
            nested_field = list(nested["properties"].keys())[0]
            examples.append(f"${{{trigger_id}.data.{field}.{nested_field}}}")
        else:
            examples.append(f"${{{trigger_id}.data.{field}}}")

    return examples


def fix_trigger_event_schemas(spec_dict: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Auto-fix trigger event_schemas by replacing with canonical schemas from registry.

    This function compares each trigger's event_schema in the workflow spec against
    the canonical schema from the trigger registry. If they differ (or the spec's
    schema is empty), it replaces the spec's schema with the canonical one.

    Args:
        spec_dict: The workflow spec dictionary to fix

    Returns:
        Tuple of (spec_dict with fixes applied, list of fix records for feedback)

    Example feedback record:
        {
            "trigger_id": "new_email",
            "trigger_key": "poll.gmail.email_received",
            "reason": "Empty event_schema replaced with canonical schema",
            "available_fields": {"envelope": [...], "data": [...]},
            "example_expressions": ["${new_email.data.message_id}", ...]
        }
    """
    fixes = []
    triggers = spec_dict.get("triggers", [])

    for trigger in triggers:
        trigger_key = trigger.get("key")
        if not trigger_key:
            continue

        canonical_trigger = trigger_registry.maybe_get(trigger_key)
        if not canonical_trigger or not canonical_trigger.schemas.event:
            continue  # Unknown trigger or no canonical schema

        canonical_schema = canonical_trigger.schemas.event
        spec_schema = trigger.get("event_schema")

        if _schemas_differ(spec_schema, canonical_schema):
            # Determine reason for fix
            if not spec_schema:
                reason = "Empty event_schema replaced with canonical schema"
            else:
                reason = "Incorrect event_schema replaced with canonical schema"

            # Apply fix (modifies trigger dict in place)
            trigger["event_schema"] = deepcopy(canonical_schema)

            # Record fix for feedback
            fixes.append({
                "trigger_id": trigger.get("id"),
                "trigger_key": trigger_key,
                "reason": reason,
                "available_fields": _extract_available_fields(canonical_schema),
                "example_expressions": _build_example_expressions(trigger.get("id", "trigger"), canonical_schema)
            })

    return spec_dict, fixes

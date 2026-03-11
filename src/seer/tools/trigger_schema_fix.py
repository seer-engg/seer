"""
Auto-fix trigger event_schemas by replacing with canonical schemas from the registry.

This module provides functions to detect and fix incorrect trigger event_schemas
in workflow specs, ensuring they match the canonical schemas defined in the
trigger registry.

Used by both Nexus agent tools and MCP tools for workflow validation.
"""

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

from seer.core.registry.trigger_registry import trigger_registry


def _schemas_differ(spec_schema: Optional[Dict], canonical_schema: Dict) -> bool:
    """Check if spec schema differs from canonical and needs replacement."""
    if not spec_schema:  # None or empty dict
        return True
    return spec_schema != canonical_schema


# Known envelope fields in trigger event schemas
_ENVELOPE_FIELDS = {"id", "trigger_key", "provider", "account_id", "occurred_at", "received_at", "data", "raw", "type"}


def _detect_misplaced_properties(spec_schema: Optional[Dict], canonical_schema: Optional[Dict]) -> List[str]:
    """
    Detect properties at root level that likely belong inside 'data'.

    When agents define custom fields at the wrong structural level (e.g., at root
    instead of nested under data.properties), this function identifies them so
    we can provide helpful feedback.

    Args:
        spec_schema: The agent's original event_schema
        canonical_schema: The canonical schema from trigger registry

    Returns:
        List of property names that appear to be misplaced (at root instead of data)
    """
    if not spec_schema:
        return []

    # Get root-level property names from spec
    spec_props = set(spec_schema.get("properties", {}).keys())

    # Get canonical envelope property names
    canonical_props = set((canonical_schema or {}).get("properties", {}).keys())

    # Properties at root that aren't part of canonical envelope are likely misplaced
    misplaced = spec_props - canonical_props - _ENVELOPE_FIELDS

    return sorted(list(misplaced))


def _merge_custom_data_properties(canonical_schema: Dict, spec_schema: Optional[Dict]) -> Dict:
    """
    Merge custom data.properties from spec into canonical schema.

    For triggers with dynamic data (form.hosted, webhook.generic), preserves
    agent-defined field schemas while keeping the canonical envelope structure.

    Args:
        canonical_schema: The canonical schema from trigger registry
        spec_schema: The agent's original event_schema (may contain custom data properties)

    Returns:
        Merged schema with canonical envelope and preserved custom data.properties
    """
    if not spec_schema:
        return deepcopy(canonical_schema)

    result = deepcopy(canonical_schema)

    # Extract custom data.properties from spec
    spec_data = spec_schema.get("properties", {}).get("data", {})
    custom_props = spec_data.get("properties", {})

    if not custom_props:
        return result

    # Merge into canonical schema's data property
    canonical_data = result.get("properties", {}).get("data", {})
    if canonical_data.get("additionalProperties") is True:
        # Preserve additionalProperties: true but add explicit properties
        if "properties" not in canonical_data:
            canonical_data["properties"] = {}
        canonical_data["properties"].update(deepcopy(custom_props))
        result["properties"]["data"] = canonical_data

    return result


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

    # Handle declared properties
    for field in list(data_props.keys())[:3]:
        nested = data_props.get(field, {})
        if nested.get("type") == "object" and nested.get("properties"):
            nested_field = list(nested["properties"].keys())[0]
            examples.append(f"${{{trigger_id}.data.{field}.{nested_field}}}")
        else:
            examples.append(f"${{{trigger_id}.data.{field}}}")

    # Handle additionalProperties: true (dynamic fields like form.hosted)
    if not data_props and data_schema.get("additionalProperties") is True:
        examples.append(f"${{{trigger_id}.data.<field_name>}}")
        examples.append("# Replace <field_name> with names from provider_config.fields")

    return examples


def _infer_data_fields_from_nodes(spec_dict: Dict[str, Any], trigger_id: str) -> set[str]:
    """Extract field names referenced as ${trigger_id.data.X} in any node."""
    pattern = re.compile(r"\$\{" + re.escape(trigger_id) + r"\.data\.(\w+)\}")
    fields: set[str] = set()

    def scan(value: Any) -> None:
        if isinstance(value, str):
            fields.update(pattern.findall(value))
        elif isinstance(value, dict):
            for v in value.values():
                scan(v)
        elif isinstance(value, list):
            for v in value:
                scan(v)

    for node in spec_dict.get("nodes", []):
        scan(node)
    return fields


def _determine_fix_reason(spec_schema: Optional[Dict]) -> str:
    """Determine the reason string for a schema fix."""
    if not spec_schema:
        return "Empty event_schema replaced with canonical schema"
    spec_data_props = spec_schema.get("properties", {}).get("data", {}).get("properties", {})
    if spec_data_props:
        return "Event schema fixed, custom data fields preserved"
    return "Incorrect event_schema replaced with canonical schema"


def _inject_inferred_fields(merged_schema: Dict[str, Any], spec_dict: Dict[str, Any], trigger_id: str) -> None:
    """For dynamic-data triggers, infer fields from node references and inject into schema."""
    data_schema = merged_schema.get("properties", {}).get("data", {})
    if data_schema.get("additionalProperties") is not True:
        return
    inferred = _infer_data_fields_from_nodes(spec_dict, trigger_id)
    if not inferred:
        return
    if "properties" not in data_schema:
        data_schema["properties"] = {}
    for field in inferred:
        if field not in data_schema["properties"]:
            data_schema["properties"][field] = {"type": "string"}


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
            # Detect misplaced properties before applying fix
            misplaced_props = _detect_misplaced_properties(spec_schema, canonical_schema)

            reason = _determine_fix_reason(spec_schema)

            # Apply fix - merge custom data.properties if present
            merged_schema = _merge_custom_data_properties(canonical_schema, spec_schema)
            trigger["event_schema"] = merged_schema

            _inject_inferred_fields(merged_schema, spec_dict, trigger.get("id", ""))

            # Build fix record with enhanced feedback for misplaced properties
            fix_record: Dict[str, Any] = {
                "trigger_id": trigger.get("id"),
                "trigger_key": trigger_key,
                "reason": reason,
                "available_fields": _extract_available_fields(merged_schema),
                "example_expressions": _build_example_expressions(trigger.get("id", "trigger"), merged_schema),
            }

            # Add warning and guidance if properties were misplaced (and thus stripped)
            if misplaced_props:
                fix_record["stripped_properties"] = misplaced_props
                fix_record["warning"] = (
                    f"Found properties at root level that likely belong inside 'data': {misplaced_props}. "
                    "These properties were stripped. Custom form fields must be nested under data.properties."
                )
                # Build correct structure hint with the actual stripped property names
                fix_record["correct_structure_hint"] = {
                    "event_schema": {
                        "properties": {
                            "data": {
                                "type": "object",
                                "properties": {prop: {"type": "string"} for prop in misplaced_props},
                            }
                        }
                    }
                }

            fixes.append(fix_record)

    return spec_dict, fixes

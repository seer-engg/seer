"""Utility module for inferring workflow input schemas from trigger event schemas.

This module is separate to avoid circular imports between api.workflows.services
and api.triggers.services.
"""

from __future__ import annotations

from typing import Any, Dict

from seer.core.registry.trigger_registry import TriggerDefinition
from seer.core.schema.models import InputDef, InputType


def infer_input_contract_from_event_schema(
    trigger_def: TriggerDefinition,
) -> Dict[str, InputDef]:
    """
    Auto-infer workflow inputs from trigger's event schema.

    Extracts properties from event_schema.properties.data.properties and converts them
    to InputDef objects. Flattens nested objects (e.g., data.from.email → from_email).

    Args:
        trigger_def: TriggerDefinition with event_schema

    Returns:
        Dict[str, InputDef]: Mapping of input names to InputDef objects

    Example:
        Gmail trigger with data.subject, data.from.email →
        {
            "subject": InputDef(type="string", description="Email subject"),
            "from_email": InputDef(type="string", description="Sender email address"),
        }
    """
    input_contract: Dict[str, InputDef] = {}

    # Extract data properties from event schema
    event_schema = trigger_def.schemas.event or {}
    properties = event_schema.get("properties", {})
    data_schema = properties.get("data", {})
    data_props = data_schema.get("properties", {})

    def _map_json_schema_type_to_input_type(json_type: Any) -> InputType:
        """Map JSON Schema type to InputType enum."""
        if isinstance(json_type, list):
            # Handle array types like ["string", "null"]
            # Use the first non-null type
            for t in json_type:
                if t != "null":
                    json_type = t
                    break
            else:
                json_type = "string"  # Default if all are null

        type_map = {
            "string": InputType.string,
            "integer": InputType.integer,
            "number": InputType.number,
            "boolean": InputType.boolean,
            "object": InputType.object,
            "array": InputType.array,
        }
        return type_map.get(json_type, InputType.string)

    def _flatten_properties(
        props: Dict[str, Any],
        prefix: str = "",
        path: str = "",
    ) -> None:
        """Recursively flatten nested object properties."""
        for key, schema in props.items():
            field_path = f"{path}.{key}" if path else key
            field_name = f"{prefix}_{key}" if prefix else key

            schema_type = schema.get("type")

            # If it's an object with properties, flatten it
            if schema_type == "object" and "properties" in schema:
                _flatten_properties(
                    schema["properties"],
                    prefix=field_name,
                    path=field_path,
                )
            else:
                # Create an InputDef for this field
                description = schema.get("description")
                if not description:
                    # Generate description from field path
                    description = f"From event: {field_path}"

                input_type = _map_json_schema_type_to_input_type(schema_type)

                input_contract[field_name] = InputDef(
                    type=input_type,
                    description=description,
                    required=False,  # Auto-inferred inputs are optional by default
                )

    # Start flattening from data properties
    _flatten_properties(data_props)

    return input_contract

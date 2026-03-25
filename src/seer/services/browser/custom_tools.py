# pylint: disable=too-many-arguments,too-many-positional-arguments
# Reason: browser_use integration requires matching their action handler signatures
"""
Custom browser-use tools with structured data submission action.

Provides a submit_result action that bypasses the complex StructuredOutputAction
validation, leveraging the LLM's tool-calling capabilities for reliable
structured output.

The key insight is that models like Kimi-k2.5 are excellent at tool-calling
but struggle with the complex action union schemas that browser-use uses for
structured output. By using a simple tool instead, we get reliable extraction.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from browser_use.agent.views import ActionResult
from browser_use.tools.service import Tools
from pydantic import BaseModel, create_model

from seer.logger import get_logger

logger = get_logger(__name__)


# Type mapping from JSON schema types to Python types
_JSON_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}


def _json_type_to_python(schema: Dict[str, Any]) -> type:
    """
    Map JSON schema type to Python type for Pydantic model generation.

    Args:
        schema: JSON schema definition for a single field

    Returns:
        Python type corresponding to the JSON schema type
    """
    json_type = schema.get("type", "string")

    # Handle array type recursively
    if json_type == "array":
        items_schema = schema.get("items", {})
        item_type = _json_type_to_python(items_schema)
        return List[item_type]  # type: ignore[valid-type]

    # Handle object type as Dict
    if json_type == "object":
        return Dict[str, Any]

    return _JSON_TYPE_MAP.get(json_type, str)


def create_submit_result_model(
    schema: Dict[str, Any],
    model_name: str = "SubmitResultParams",
) -> Type[BaseModel]:
    """
    Create a Pydantic model from JSON schema for submit_result action params.

    This creates a flat, simple model that LLMs can easily fill via tool-calling,
    avoiding the complex nested union types that cause parsing failures.

    Args:
        schema: JSON schema dict with "type", "properties", and optional "required"
        model_name: Name for the generated Pydantic model class

    Returns:
        Dynamically generated Pydantic model class

    Example:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"}
            },
            "required": ["name"]
        }
        Model = create_submit_result_model(schema)
        # Model has name (required str) and price (optional float)
    """
    if schema.get("type") != "object":
        # For non-object schemas, wrap in a "data" field
        return create_model(model_name, data=(Any, ...))

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    field_definitions: Dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        field_type = _json_type_to_python(field_schema)
        if field_name in required:
            field_definitions[field_name] = (field_type, ...)
        else:
            field_definitions[field_name] = (Optional[field_type], None)

    return create_model(model_name, **field_definitions)


class CustomBrowserTools(Tools):
    """
    Extended Tools with submit_result action for reliable structured output.

    This subclass adds a custom submit_result action that lets the LLM submit
    structured data via tool-calling instead of the complex done action with
    StructuredOutputAction wrapper.

    Key difference from standard browser-use structured output:
    - Standard: Agent calls done(StructuredOutputAction[T](data=T(...), success=...))
    - Custom: Agent calls submit_result(field1=..., field2=..., ...)

    The simpler parameter structure significantly improves extraction reliability
    for models like Kimi that excel at tool-calling but struggle with complex
    nested schemas.

    Usage:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        tools = CustomBrowserTools(extraction_schema=schema)
        # ... run agent with tools=tools ...
        data = tools.get_extracted_data()
    """

    def __init__(
        self,
        extraction_schema: Optional[Dict[str, Any]] = None,
        exclude_actions: Optional[List[str]] = None,
    ):
        """
        Initialize custom tools with optional extraction schema.

        Args:
            extraction_schema: JSON schema for structured data extraction.
                If provided, registers a submit_result action with matching params.
            exclude_actions: List of default action names to exclude.
        """
        # Don't pass output_model to parent - we handle structured output ourselves
        # This avoids the complex StructuredOutputAction wrapper
        super().__init__(output_model=None, exclude_actions=exclude_actions)

        self._extraction_schema = extraction_schema
        self._extracted_data: Dict[str, Any] = {}

        if extraction_schema:
            self._register_submit_result_action(extraction_schema)

    def _register_submit_result_action(self, schema: Dict[str, Any]) -> None:
        """
        Register submit_result action based on extraction schema.

        Creates a dynamically-typed action that accepts parameters matching
        the schema fields. When called, it stores the validated data and
        signals task completion.

        Args:
            schema: JSON schema defining expected output structure
        """
        param_model = create_submit_result_model(schema, "SubmitResultParams")
        field_names = list(schema.get("properties", {}).keys())
        fields_desc = ", ".join(field_names) if field_names else "the required data"

        # Store reference to self for closure
        tools_instance = self

        @self.registry.action(
            f"Submit the extracted structured data ({fields_desc}). "
            "Call this action when you have gathered all required information from the page.",
            param_model=param_model,
        )
        async def submit_result(params: param_model) -> ActionResult:  # type: ignore[valid-type]
            """Submit extracted data and complete the task."""
            # Store the validated data via public method
            data = params.model_dump()
            tools_instance.set_extracted_data(data)

            logger.info("submit_result called with data: %s", data)

            return ActionResult(
                is_done=True,
                success=True,
                extracted_content=json.dumps(data, ensure_ascii=False),
                long_term_memory="Structured data submitted successfully via submit_result.",
            )

    def set_extracted_data(self, data: Dict[str, Any]) -> None:
        """
        Set the extracted data (used by submit_result action).

        Args:
            data: The extracted data to store.
        """
        self._extracted_data = data

    def get_extracted_data(self) -> Dict[str, Any]:
        """
        Get the data submitted via submit_result action.

        Returns:
            Dict containing the extracted data, or empty dict if not yet submitted.
        """
        return self._extracted_data

    def has_extracted_data(self) -> bool:
        """
        Check if data has been submitted via submit_result.

        Returns:
            True if submit_result was called with data.
        """
        return bool(self._extracted_data)
